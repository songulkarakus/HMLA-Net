"""Data: indexing, hash-based de-duplication, stratified splits, transforms, external validation."""
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split, StratifiedKFold

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

Sample = Tuple[str, int]


# --------------------------------------------------------------------------- #
# Indexing + duplicate (leakage) removal
# --------------------------------------------------------------------------- #
def _file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_index(data_dir: str, dedup: bool = True) -> Tuple[List[Sample], List[str], int]:
    """<data_dir>/<plant> leaf/<class>/*.jpg  ->  (samples, class_names, n_dup)."""
    root = Path(data_dir)
    class_dirs = sorted([p for p in root.glob("*/*") if p.is_dir()])
    if not class_dirs:
        raise SystemExit(f"No class folders in: {root} (expected: <plant> leaf/<class>/*.jpg)")
    class_names = [p.name for p in class_dirs]

    samples: List[Sample] = []
    seen, n_dup = set(), 0
    for label, cdir in enumerate(class_dirs):
        for f in cdir.iterdir():
            if f.suffix.lower() not in IMG_EXTS:
                continue
            if dedup:
                h = _file_md5(f)
                if h in seen:
                    n_dup += 1
                    continue
                seen.add(h)
            samples.append((str(f), label))
    return samples, class_names, n_dup


def plant_index(class_names: List[str]) -> Tuple[Optional[List[int]], List[str]]:
    """Derives the plant group from the class name: 'mango anthracnose' -> 'mango'.

    Used by the hierarchical (plant -> disease) head of the proposed model.
    If no grouping can be derived (a single plant) it returns (None, []) and the
    caller disables the hierarchy.
    """
    prefixes = [(n.split()[0].lower() if n.split() else n.lower()) for n in class_names]
    plant_names = sorted(set(prefixes))
    if len(plant_names) < 2:
        return None, []
    idx = {p: i for i, p in enumerate(plant_names)}
    return [idx[p] for p in prefixes], plant_names


def stratified_split(samples: List[Sample], seed: int,
                     val_frac: float, test_frac: float) -> Dict[str, List[Sample]]:
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    tr_p, tmp_p, tr_l, tmp_l = train_test_split(
        paths, labels, test_size=val_frac + test_frac,
        stratify=labels, random_state=seed)
    rel = test_frac / (val_frac + test_frac)
    va_p, te_p, va_l, te_l = train_test_split(
        tmp_p, tmp_l, test_size=rel, stratify=tmp_l, random_state=seed)
    return {
        "train": list(zip(tr_p, tr_l)),
        "val": list(zip(va_p, va_l)),
        "test": list(zip(te_p, te_l)),
    }


def stratified_kfold_splits(samples: List[Sample], n_splits: int, seed: int,
                            val_frac: float = 0.15) -> List[Dict[str, List[Sample]]]:
    """Stratified K-fold: each fold becomes the test set; val is carved from the rest. Returns K splits."""
    labels = [s[1] for s in samples]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for tr_idx, te_idx in skf.split(np.zeros(len(samples)), labels):
        tr = [samples[i] for i in tr_idx]
        te = [samples[i] for i in te_idx]
        tr_p = [s[0] for s in tr]
        tr_l = [s[1] for s in tr]
        tr2_p, va_p, tr2_l, va_l = train_test_split(
            tr_p, tr_l, test_size=val_frac, stratify=tr_l, random_state=seed)
        folds.append({"train": list(zip(tr2_p, tr2_l)),
                      "val": list(zip(va_p, va_l)),
                      "test": te})
    return folds


# --------------------------------------------------------------------------- #
# Dataset + transforms
# --------------------------------------------------------------------------- #
class LeafDataset(Dataset):
    def __init__(self, samples: List[Sample], transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_transforms(img_size: int, train: bool, random_erasing: bool = True):
    if train:
        tfs = [
            T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.333)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(0.2, 0.2, 0.2, 0.05),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
        if random_erasing:
            tfs.append(T.RandomErasing(p=0.25, scale=(0.02, 0.15)))
        return T.Compose(tfs)
    return T.Compose([
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def make_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_loaders(splits, cfg, device):
    """Builds the train/val/test DataLoaders."""
    train_tf = build_transforms(cfg.img_size, True, cfg.use_random_erasing)
    eval_tf = build_transforms(cfg.img_size, False)
    train_ds = LeafDataset(splits["train"], train_tf)
    val_ds = LeafDataset(splits["val"], eval_tf)
    test_ds = LeafDataset(splits["test"], eval_tf)

    kw = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers,
              pin_memory=(device == "cuda"),
              persistent_workers=cfg.num_workers > 0)

    train_labels = [s[1] for s in splits["train"]]
    if cfg.use_sampler:
        train_loader = DataLoader(train_ds, sampler=make_weighted_sampler(train_labels), **kw)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------- #
# EXTERNAL VALIDATION
#   - Works with an independent data set + a class-mapping JSON.
#   - mapping: { "their_folder_name": "our_class_name", ... }
#     Only images that map onto one of our 21 classes are used.
# --------------------------------------------------------------------------- #
def build_external_index(ext_dir: str, class_names: List[str], mapping_json: str,
                         max_per_class: int = None, seed: int = 42
                         ) -> Tuple[List[Sample], Dict]:
    """Projects an external data set onto our label space (optional per-class cap)."""
    root = Path(ext_dir)
    with open(mapping_json, "r", encoding="utf-8-sig") as f:  # tolerant of a BOM
        mapping = json.load(f)
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    # the mapping targets must be among our classes
    for their, ours in mapping.items():
        if ours is not None and ours not in name_to_idx:
            raise ValueError(f"Mapping target '{ours}' is not one of our classes.")

    # collect per class first (several source folders may map onto the same class,
    # e.g. Early_blight + Late_blight -> 'tomato blight')
    by_class = {}
    folders_by_name = {}
    for folder in [p for p in root.rglob("*") if p.is_dir()]:
        ours = mapping.get(folder.name)
        if ours is None:
            continue
        folders_by_name.setdefault(folder.name, []).append(str(folder))
        for f in folder.iterdir():
            if f.suffix.lower() in IMG_EXTS:
                by_class.setdefault(ours, []).append(str(f))

    # PITFALL: PlantVillage ships sibling folders color/ grayscale/ segmented/. If
    # ext_dir points at the parent folder, the SAME leaf is collected in three
    # renderings and the external validation silently inflates / becomes meaningless.
    # If the same folder NAME appears under several paths, warn and record the evidence.
    duplicated = {n: ps for n, ps in folders_by_name.items() if len(ps) > 1}
    if duplicated:
        print("WARNING: an external-data folder name was found under several paths — "
              "most likely different renderings of the same images are being counted:")
        for n, ps in list(duplicated.items())[:5]:
            print(f"  '{n}' -> {ps}")
        print("  Fix: point --external_dir at a single rendering "
              "(e.g. '.../plantvillage dataset/color').")

    rng = np.random.default_rng(seed)
    samples: List[Sample] = []
    per_class = Counter()
    for ours, paths in by_class.items():
        if max_per_class and len(paths) > max_per_class:
            idx = rng.permutation(len(paths))[:max_per_class]
            paths = [paths[i] for i in idx]
        label = name_to_idx[ours]
        for p in paths:
            samples.append((p, label))
            per_class[ours] += 1

    info = {"n_images": len(samples), "per_class": dict(per_class),
            "covered_classes": sorted(per_class.keys()),
            "max_per_class": max_per_class,
            "source_folders": folders_by_name,
            "duplicate_folder_names": duplicated}
    return samples, info


def make_external_loader(samples: List[Sample], cfg, device) -> DataLoader:
    eval_tf = build_transforms(cfg.img_size, False)
    ds = LeafDataset(samples, eval_tf)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                      num_workers=cfg.num_workers, pin_memory=(device == "cuda"))
