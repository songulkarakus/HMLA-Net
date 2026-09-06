"""
Grad-CAM explainability (XAI) — which regions does the proposed model look at?
Produces a heat-map overlay for one example of every class.

Example:
    python gradcam.py --seed 42 --out results/proposed/gradcam
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from leafdx.config import RunConfig, PROPOSED_NAME
from leafdx.models import build_model
from leafdx.data import build_index, build_transforms, IMAGENET_MEAN, IMAGENET_STD


def safe_name(s):
    return s.replace("/", "_").replace(".", "-")


def grad_cam(model, x, target_layer, class_idx=None):
    store = {}

    def fwd(_m, _i, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    h = target_layer.register_forward_hook(fwd)
    logits = model(x)
    if class_idx is None:
        class_idx = int(logits.argmax(1))
    model.zero_grad(set_to_none=True)
    logits[0, class_idx].backward()
    h.remove()

    act = store["act"].detach()[0]          # [C,H,W]
    grad = store["grad"].detach()[0]        # [C,H,W]
    weights = grad.mean(dim=(1, 2))         # [C]
    cam = F.relu((weights[:, None, None] * act).sum(0))
    cam = cam / (cam.max() + 1e-8)
    return cam.cpu().numpy(), class_idx


def denorm(x):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (x.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--data_dir", default="multiple leaf dataset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/proposed/gradcam")
    ap.add_argument("--per_class", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = Path(args.results_dir)
    with open(res / "class_names.json", encoding="utf-8") as f:
        class_names = json.load(f)

    best = res / "proposed" / safe_name(PROPOSED_NAME) / f"seed{args.seed}" / "best.pth"
    if not best.exists():
        raise SystemExit(f"{best} does not exist. Train the proposed model first.")
    ck = torch.load(best, map_location=device, weights_only=False)
    cfg = RunConfig(**ck["cfg"])
    model = build_model(cfg, len(class_names)).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    target_layer = model.get_cam_layer()
    tf = build_transforms(cfg.img_size, train=False)

    samples, names, _ = build_index(args.data_dir, dedup=True)
    by_class = {}
    for path, lab in samples:
        by_class.setdefault(lab, []).append(path)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    picks = [(lab, by_class[lab][i]) for lab in sorted(by_class)
             for i in range(min(args.per_class, len(by_class[lab])))]
    n = len(picks)
    ncol = 6
    nrow = int(np.ceil(n / ncol))

    def overlay_of(x, heat_2d):
        """Overlays an [H,W] heat-map on the input image."""
        up = np.array(Image.fromarray((heat_2d * 255).astype(np.uint8))
                      .resize((cfg.img_size, cfg.img_size), Image.BILINEAR)) / 255.0
        return 0.5 * denorm(x[0]) + 0.5 * cm.jet(up)[..., :3]

    def new_grid():
        f, a = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 2.4))
        return f, np.atleast_1d(a).ravel()

    fig, axes = new_grid()
    lap_frames = []            # LAP's own attention map (if available)

    for ax, (lab, path) in zip(axes, picks):
        img = Image.open(path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        cam, pred = grad_cam(model, x, target_layer, class_idx=None)
        ax.imshow(overlay_of(x, cam))
        ok = "✓" if pred == lab else "✗"
        ax.set_title(f"{class_names[lab]}\n→{class_names[pred]} {ok}", fontsize=6)
        ax.axis("off")

        # the LAP attention map was computed in the same forward pass — a free second XAI figure
        attn = model.attention_map() if hasattr(model, "attention_map") else None
        if attn is not None:
            a = attn[0, 0].float().cpu().numpy()
            a = a / (a.max() + 1e-8)
            lap_frames.append((overlay_of(x, a), lab, pred, ok))

    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Grad-CAM — attention maps of the proposed model", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "gradcam_grid.png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_dir / 'gradcam_grid.png'}")

    if lap_frames:
        fig, axes = new_grid()
        for ax, (ov, lab, pred, ok) in zip(axes, lap_frames):
            ax.imshow(ov)
            ax.set_title(f"{class_names[lab]}\n→{class_names[pred]} {ok}", fontsize=6)
            ax.axis("off")
        for ax in axes[len(lap_frames):]:
            ax.axis("off")
        fig.suptitle("LAP — the model's own lesion attention map (gradient-free)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_dir / "lap_attention_grid.png", dpi=150)
        plt.close(fig)
        print(f"Saved: {out_dir / 'lap_attention_grid.png'}")


if __name__ == "__main__":
    main()
