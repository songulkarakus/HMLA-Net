"""
Exploratory data analysis (EDA) + data-set statistics.

Figures produced (results/dataset/):
  class_distribution.png     - class distribution (coloured by plant, imbalance)
  samples_grid.png           - one sample image per class
  image_dimensions.png       - width/height + aspect-ratio distribution
  color_analysis.png         - mean R/G/B per class (colour = disease signal)
  brightness_by_plant.png    - brightness distribution per plant (box plot)
  mean_image_per_class.png   - MEAN image per class (typical leaf)
  rgb_histogram.png          - overall RGB intensity histograms
  eda_stats.csv              - aggregate statistics per class
  dataset_summary.json       - summary

Example:  python dataset_report.py --out results/dataset --sample_per_class 50
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from leafdx.data import build_index


def plant_of(class_name):
    return class_name.split()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="multiple leaf dataset")
    ap.add_argument("--out", default="results/dataset")
    ap.add_argument("--sample_per_class", type=int, default=50,
                    help="images sampled per class for the colour/size statistics")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    samples, class_names, n_dup = build_index(args.data_dir, dedup=True)
    raw, _, _ = build_index(args.data_dir, dedup=False)
    counts = Counter(lab for _, lab in samples)
    by_class = defaultdict(list)
    for path, lab in samples:
        by_class[lab].append(path)
    plants = sorted({plant_of(c) for c in class_names})
    palette = dict(zip(plants, plt.cm.tab10.colors))

    # ---------------------------------------------------------------- #
    # 1) Image sampling + colour/size/brightness statistics
    # ---------------------------------------------------------------- #
    rows = []
    mean_imgs, S = {}, 64
    for lab, paths in by_class.items():
        sel = paths if len(paths) <= args.sample_per_class else \
            [paths[i] for i in rng.permutation(len(paths))[:args.sample_per_class]]
        acc, n = np.zeros((S, S, 3), np.float64), 0
        for p in sel:
            try:
                im = Image.open(p).convert("RGB")
                w, h = im.size
                arr = np.asarray(im.resize((S, S)), np.float32) / 255.0
                acc += arr
                n += 1
                rows.append({"class": class_names[lab], "plant": plant_of(class_names[lab]),
                             "width": w, "height": h, "aspect": w / max(h, 1),
                             "R": float(arr[..., 0].mean()), "G": float(arr[..., 1].mean()),
                             "B": float(arr[..., 2].mean()), "brightness": float(arr.mean())})
            except Exception:
                pass
        if n > 0:
            mean_imgs[lab] = acc / n
    stat = pd.DataFrame(rows)

    # aggregate statistics CSV per class
    agg = (stat.groupby(["plant", "class"])
           .agg(n_sampled=("R", "size"), R=("R", "mean"), G=("G", "mean"),
                B=("B", "mean"), brightness=("brightness", "mean"),
                width=("width", "mean"), height=("height", "mean"),
                aspect=("aspect", "mean")).reset_index())
    agg.insert(3, "n_total", agg["class"].map(
        {class_names[i]: counts.get(i, 0) for i in range(len(class_names))}))
    agg.to_csv(out / "eda_stats.csv", index=False)

    # ---------------------------------------------------------------- #
    # 2) Summary JSON
    # ---------------------------------------------------------------- #
    per_plant = defaultdict(int)
    for i, c in enumerate(class_names):
        per_plant[plant_of(c)] += counts.get(i, 0)
    dim_counts = Counter((int(r["width"]), int(r["height"])) for _, r in stat.iterrows())
    summary = {
        "n_classes": len(class_names), "n_plants": len(per_plant),
        "n_images_dedup": len(samples), "n_images_raw": len(raw),
        "n_duplicates_removed": n_dup, "images_per_plant": dict(per_plant),
        "min_class": min(counts.values()), "max_class": max(counts.values()),
        "imbalance_ratio": round(max(counts.values()) / max(min(counts.values()), 1), 2),
        "common_dimensions": {f"{w}x{h}": c for (w, h), c in dim_counts.most_common(5)},
    }
    with open(out / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # class-distribution table
    dist = pd.DataFrame([{"class": class_names[i], "plant": plant_of(class_names[i]),
                          "n_images": counts.get(i, 0)} for i in range(len(class_names))]
                        ).sort_values(["plant", "class"]).reset_index(drop=True)
    dist.to_csv(out / "class_distribution.csv", index=False)

    def save(fig, name):
        fig.tight_layout(); fig.savefig(out / name, dpi=150); plt.close(fig)

    # ---------------------------------------------------------------- #
    # 3) EDA figures
    # ---------------------------------------------------------------- #
    # (a) class distribution
    try:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(range(len(dist)), dist["n_images"],
               color=[palette[p] for p in dist["plant"]])
        ax.set_xticks(range(len(dist)))
        ax.set_xticklabels(dist["class"], rotation=90, fontsize=7)
        ax.set_ylabel("Number of images")
        ax.set_title(f"Class distribution ({len(samples)} images, {len(class_names)} classes, "
                     f"imbalance {summary['imbalance_ratio']}x)")
        ax.legend([plt.Rectangle((0, 0), 1, 1, color=palette[p]) for p in plants],
                  plants, title="Plant")
        save(fig, "class_distribution.png")
    except Exception as e:
        print(f"(warning: class_distribution skipped: {e})")

    # (b) sample-image grid (one per class)
    try:
        n = len(class_names); ncol = 7; nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
        axes = np.atleast_1d(axes).ravel()
        for ax, i in zip(axes, range(n)):
            try:
                ax.imshow(Image.open(by_class[i][0]).convert("RGB"))
            except Exception:
                pass
            ax.set_title(class_names[i], fontsize=6); ax.axis("off")
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle("One sample image per class", fontsize=12)
        save(fig, "samples_grid.png")
    except Exception as e:
        print(f"(warning: samples_grid skipped: {e})")

    # (c) image-size distribution
    try:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
        a1.scatter(stat["width"], stat["height"], s=6, alpha=0.3,
                   c=[palette[p] for p in stat["plant"]])
        a1.set_xlabel("Width (px)"); a1.set_ylabel("Height (px)")
        a1.set_title("Image dimensions")
        a2.hist(stat["aspect"], bins=40, color="#4c72b0")
        a2.set_xlabel("Aspect ratio (w/h)"); a2.set_ylabel("Number of images")
        a2.set_title("Aspect-ratio distribution")
        save(fig, "image_dimensions.png")
    except Exception as e:
        print(f"(warning: image_dimensions skipped: {e})")

    # (d) mean colour per class (R/G/B)
    try:
        cm = agg.sort_values(["plant", "class"])
        x = np.arange(len(cm)); w = 0.27
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(x - w, cm["R"], w, label="R", color="#d62728")
        ax.bar(x, cm["G"], w, label="G", color="#2ca02c")
        ax.bar(x + w, cm["B"], w, label="B", color="#1f77b4")
        ax.set_xticks(x); ax.set_xticklabels(cm["class"], rotation=90, fontsize=7)
        ax.set_ylabel("Mean channel value [0-1]")
        ax.set_title("Mean colour per class (disease colour signal)")
        ax.legend()
        save(fig, "color_analysis.png")
    except Exception as e:
        print(f"(warning: color_analysis skipped: {e})")

    # (e) brightness per plant (box plot)
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = [stat[stat["plant"] == p]["brightness"].values for p in plants]
        # matplotlib >= 3.9 renamed 'labels' -> 'tick_labels'
        try:
            bp = ax.boxplot(data, tick_labels=plants, patch_artist=True)
        except TypeError:
            bp = ax.boxplot(data, labels=plants, patch_artist=True)
        for patch, p in zip(bp["boxes"], plants):
            patch.set_facecolor(palette[p])
        ax.set_ylabel("Mean brightness [0-1]")
        ax.set_title("Image brightness per plant")
        save(fig, "brightness_by_plant.png")
    except Exception as e:
        print(f"(warning: brightness_by_plant skipped: {e})")

    # (f) MEAN image per class (typical leaf)
    try:
        n = len(class_names); ncol = 7; nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
        axes = np.atleast_1d(axes).ravel()
        for ax, i in zip(axes, range(n)):
            if i in mean_imgs:
                ax.imshow(np.clip(mean_imgs[i], 0, 1))
            ax.set_title(class_names[i], fontsize=6); ax.axis("off")
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle("Mean image per class (typical pattern)", fontsize=12)
        save(fig, "mean_image_per_class.png")
    except Exception as e:
        print(f"(warning: mean_image_per_class skipped: {e})")

    # (g) overall RGB histogram
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        for ch, col in zip(["R", "G", "B"], ["#d62728", "#2ca02c", "#1f77b4"]):
            ax.hist(stat[ch], bins=40, alpha=0.5, label=ch, color=col)
        ax.set_xlabel("Mean channel value [0-1]"); ax.set_ylabel("Number of images")
        ax.set_title("RGB intensity distribution (whole sample)")
        ax.legend()
        save(fig, "rgb_histogram.png")
    except Exception as e:
        print(f"(warning: rgb_histogram skipped: {e})")

    print(f"\nEDA report written to '{out}' "
          f"({len(list(out.glob('*.png')))} figures).")


if __name__ == "__main__":
    main()
