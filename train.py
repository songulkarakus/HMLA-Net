"""
Leaf Disease Classification - SIMPLE SINGLE-MODEL PATH (CAFormer-S18)
====================================================================
4 plants / 21 classes, ~7,255 images. PyTorch + timm.

NOTE: this is a simplified path for training a single model quickly. The PROPOSED
model (HMLA-Net) that produces the paper's results is NOT here — it lives in
`leafdx/models.py` and is run through `run_all.py` / `run_benchmark.py`.

Recipe:
  - Content-hash based duplicate (leakage) removal
  - Stratified 70 / 15 / 15 split
  - ImageNet-21k pretrained CAFormer-S18 (plain classifier, no extra components)
  - Two-phase training: (1) head only, (2) fine-tune the whole network with a low LR
  - Strong augmentation + RandomErasing (prevents memorising the clean background)
  - Class-imbalance compensation with WeightedRandomSampler
  - Label smoothing, AdamW, cosine LR, AMP (mixed precision)
  - Early stopping + saving the best validation model
  - Per-class F1 report + confusion matrix on the test set
  - Optional ONNX export

Run (Windows, GPU):
    python train.py --data_dir "multiple leaf dataset" --batch_size 16
"""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def build_samples(data_dir: Path, dedup: bool):
    """Folder layout: <data_dir>/<plant> leaf/<class>/*.jpg -> 21 classes."""
    class_dirs = sorted([p for p in data_dir.glob("*/*") if p.is_dir()])
    if not class_dirs:
        raise SystemExit(f"No class folders found: {data_dir}\n"
                         f"Expected layout: <plant> leaf/<class>/*.jpg")
    class_names = [p.name for p in class_dirs]

    samples, seen, n_dup = [], set(), 0
    for label, cdir in enumerate(class_dirs):
        files = [f for f in cdir.iterdir() if f.suffix.lower() in IMG_EXTS]
        for f in tqdm(files, desc=f"Scanning: {cdir.name}", leave=False):
            if dedup:
                h = file_md5(f)
                if h in seen:
                    n_dup += 1
                    continue
                seen.add(h)
            samples.append((str(f), label))
    return samples, class_names, n_dup


class LeafDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def build_transforms(img_size: int):
    train_tf = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.333)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(30),
        T.ColorJitter(0.2, 0.2, 0.2, 0.05),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    eval_tf = T.Compose([
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def make_weighted_sampler(labels):
    counts = Counter(labels)
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# --------------------------------------------------------------------------- #
# Training / evaluation steps
# --------------------------------------------------------------------------- #
def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=True):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_tgts = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, tgts in tqdm(loader, desc="train" if train else "eval", leave=False):
            imgs, tgts = imgs.to(device, non_blocking=True), tgts.to(device, non_blocking=True)
            with autocast("cuda", enabled=use_amp):
                out = model(imgs)
                loss = criterion(out, tgts)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * imgs.size(0)
            preds = out.argmax(1)
            correct += (preds == tgts).sum().item()
            n += imgs.size(0)
            if not train:
                all_preds.append(preds.cpu())
                all_tgts.append(tgts.cpu())

    avg_loss = total_loss / n
    acc = correct / n
    if not train:
        return avg_loss, acc, torch.cat(all_preds).numpy(), torch.cat(all_tgts).numpy()
    return avg_loss, acc


def train_phase(model, train_loader, val_loader, criterion, optimizer, scheduler,
                device, scaler, use_amp, epochs, start_epoch, best, out_dir,
                class_names, img_size, model_name, history, patience=None):
    epochs_no_improve = 0
    for ep in range(epochs):
        gep = start_epoch + ep
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device,
                                    optimizer, scaler, use_amp)
        va_loss, va_acc, _, _ = run_epoch(model, val_loader, criterion, device,
                                          use_amp=use_amp)
        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        history.append(dict(epoch=gep, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, lr=lr))
        print(f"[{gep:03d}] tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} | lr={lr:.2e} "
              f"| {time.time()-t0:.0f}s")

        if va_acc > best["val_acc"]:
            best.update(val_acc=va_acc, epoch=gep)
            torch.save({"model_state": model.state_dict(),
                        "class_names": class_names,
                        "img_size": img_size,
                        "model_name": model_name,
                        "val_acc": va_acc},
                       out_dir / "best_model.pth")
            print(f"      -> new best saved (val_acc={va_acc:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                print(f"      -> early stopping (no improvement for {patience} epochs)")
                break
    return start_epoch + epochs


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="multiple leaf dataset")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--model_name", default="caformer_s18.sail_in22k_ft_in1k")
    ap.add_argument("--img_size", type=int, default=384)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--freeze_epochs", type=int, default=3)
    ap.add_argument("--finetune_epochs", type=int, default=40)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--finetune_lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_dedup", action="store_true", help="disable duplicate removal")
    ap.add_argument("--no_amp", action="store_true", help="disable mixed precision")
    ap.add_argument("--export_onnx", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (not args.no_amp) and device == "cuda"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    if device == "cpu":
        print("WARNING: no GPU found. --img_size 224 and a lighter model are recommended.")

    # --- Data ---
    samples, class_names, n_dup = build_samples(Path(args.data_dir), dedup=not args.no_dedup)
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    print(f"\nNumber of classes: {len(class_names)} | images used: {len(paths)} "
          f"| duplicates dropped: {n_dup}")

    tr_p, tmp_p, tr_l, tmp_l = train_test_split(
        paths, labels, test_size=0.30, stratify=labels, random_state=args.seed)
    va_p, te_p, va_l, te_l = train_test_split(
        tmp_p, tmp_l, test_size=0.50, stratify=tmp_l, random_state=args.seed)
    print(f"Split -> train={len(tr_p)}  val={len(va_p)}  test={len(te_p)}")

    train_tf, eval_tf = build_transforms(args.img_size)
    train_ds = LeafDataset(tr_p, tr_l, train_tf)
    val_ds = LeafDataset(va_p, va_l, eval_tf)
    test_ds = LeafDataset(te_p, te_l, eval_tf)

    dl_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers,
                     pin_memory=(device == "cuda"),
                     persistent_workers=args.num_workers > 0)
    train_loader = DataLoader(train_ds, sampler=make_weighted_sampler(tr_l), **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **dl_kwargs)

    # --- Model ---
    model = timm.create_model(args.model_name, pretrained=True,
                              num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = GradScaler("cuda", enabled=use_amp)
    history, best = [], {"val_acc": -1.0, "epoch": -1}

    # --- Phase 1: classifier head only ---
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids
    opt1 = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                             lr=args.head_lr, weight_decay=args.weight_decay)
    print("\n=== Phase 1: head training (backbone frozen) ===")
    next_ep = train_phase(model, train_loader, val_loader, criterion, opt1, None,
                          device, scaler, use_amp, args.freeze_epochs, 0, best,
                          out_dir, class_names, args.img_size, args.model_name, history)

    # --- Phase 2: fine-tune the whole network ---
    for p in model.parameters():
        p.requires_grad = True
    opt2 = torch.optim.AdamW(model.parameters(), lr=args.finetune_lr,
                             weight_decay=args.weight_decay)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.finetune_epochs)
    print("\n=== Phase 2: full-network fine-tuning (cosine LR + early stopping) ===")
    train_phase(model, train_loader, val_loader, criterion, opt2, sched2,
                device, scaler, use_amp, args.finetune_epochs, next_ep, best,
                out_dir, class_names, args.img_size, args.model_name, history,
                patience=args.patience)

    # --- Test evaluation (best model) ---
    print(f"\nBest model: epoch {best['epoch']} (val_acc={best['val_acc']:.4f})")
    ckpt = torch.load(out_dir / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    _, test_acc, preds, tgts = run_epoch(model, test_loader, criterion, device, use_amp=use_amp)
    macro_f1 = f1_score(tgts, preds, average="macro")
    print(f"\nTEST accuracy: {test_acc:.4f} | Macro-F1: {macro_f1:.4f}")

    report = classification_report(tgts, preds, target_names=class_names, digits=4)
    print("\n" + report)
    (out_dir / "classification_report.txt").write_text(
        f"Test accuracy: {test_acc:.4f}\nMacro-F1: {macro_f1:.4f}\n\n{report}",
        encoding="utf-8")
    with open(out_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # --- Confusion matrix ---
    cm = confusion_matrix(tgts, preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (acc={test_acc:.3f})")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)

    # --- Training curves ---
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(ep, [h["train_loss"] for h in history], label="train")
    a1.plot(ep, [h["val_loss"] for h in history], label="val")
    a1.set_title("Loss"); a1.legend(); a1.set_xlabel("epoch")
    a2.plot(ep, [h["train_acc"] for h in history], label="train")
    a2.plot(ep, [h["val_acc"] for h in history], label="val")
    a2.set_title("Accuracy"); a2.legend(); a2.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out_dir / "history.png", dpi=150)
    print(f"\nOutputs saved to '{out_dir}'.")

    # --- ONNX ---
    if args.export_onnx:
        model.eval()
        dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
        onnx_path = out_dir / "model.onnx"
        torch.onnx.export(model, dummy, str(onnx_path),
                          input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                          opset_version=17)
        print(f"ONNX saved: {onnx_path}")


if __name__ == "__main__":
    main()
