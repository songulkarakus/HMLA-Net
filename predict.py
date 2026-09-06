"""
Prediction with a trained model — a single image or a folder.
Optional TTA (test-time augmentation).

Examples:
    python predict.py --image "path/leaf.jpg" --tta
    python predict.py --image "some_folder" --topk 3
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import timm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _find_class_names(ckpt_path, labels_path=None):
    """In a leafdx run the class names are not in the checkpoint but in results/class_names.json."""
    cands = [Path(labels_path)] if labels_path else []
    for parent in list(Path(ckpt_path).resolve().parents)[:5]:
        cands += [parent / "class_names.json", parent / "labels.json"]
    for c in cands:
        if c.exists():
            with open(c, encoding="utf-8") as f:
                return json.load(f)
    raise SystemExit("Class names not found. Pass the path of class_names.json with --labels.")


def load_model(ckpt_path, device, labels_path=None):
    """Supports both checkpoint formats:
       - leafdx run (results/.../best.pth): contains 'cfg' -> including the proposed HMLA-Net
       - the simple train.py path (outputs/best_model.pth): contains 'model_name'/'class_names'
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "cfg" in ckpt:
        from leafdx.config import RunConfig
        from leafdx.models import build_model
        cfg = RunConfig(**ckpt["cfg"])
        class_names = _find_class_names(ckpt_path, labels_path)
        model = build_model(cfg, len(class_names))
        img_size = cfg.img_size
    else:
        class_names = ckpt["class_names"]
        model = timm.create_model(ckpt["model_name"], pretrained=False,
                                  num_classes=len(class_names))
        img_size = ckpt["img_size"]
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    return model, class_names, img_size


def build_tf(img_size):
    return T.Compose([
        T.Resize(int(img_size * 1.15)),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


@torch.no_grad()
def predict_one(model, tf, path, device, tta):
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    batch = [x]
    if tta:
        batch += [torch.flip(x, dims=[3]), torch.flip(x, dims=[2])]  # h-flip, v-flip
    probs = torch.stack([F.softmax(model(b), dim=1) for b in batch]).mean(0)
    return probs.squeeze(0).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="image file or folder")
    ap.add_argument("--ckpt", default="outputs/best_model.pth",
                    help="outputs/best_model.pth (train.py) or results/.../best.pth (leafdx)")
    ap.add_argument("--labels", default=None,
                    help="path of class_names.json (for leafdx checkpoints; usually found automatically)")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--tta", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names, img_size = load_model(args.ckpt, device, args.labels)
    tf = build_tf(img_size)

    target = Path(args.image)
    files = [target] if target.is_file() else \
        sorted([f for f in target.rglob("*") if f.suffix.lower() in IMG_EXTS])

    for f in files:
        probs = predict_one(model, tf, f, device, args.tta)
        top = torch.topk(probs, min(args.topk, len(class_names)))
        print(f"\n{f.name}")
        for p, i in zip(top.values, top.indices):
            print(f"  {class_names[i]:<28} {p.item()*100:5.1f}%")


if __name__ == "__main__":
    main()
