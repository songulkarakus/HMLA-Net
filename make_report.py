"""
Paper-ready report generator. Run after analyze.py.
Collects all tables + figures + statistics in a single Markdown file: results/REPORT.md

Example:  python make_report.py --out_dir results
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def load_csv(p):
    return pd.read_csv(p) if Path(p).exists() else None


def load_json(p):
    if Path(p).exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def df_to_md(df, cols=None, headers=None, floatfmt="{:.4f}"):
    """DataFrame -> Markdown table (without the tabulate dependency)."""
    if df is None or len(df) == 0:
        return "_(no data)_\n"
    cols = cols or list(df.columns)
    cols = [c for c in cols if c in df.columns]
    headers = headers or cols

    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)

    out = "| " + " | ".join(headers) + " |\n"
    out += "| " + " | ".join("---" for _ in headers) + " |\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(fmt(r[c]) for c in cols) + " |\n"
    return out


def meanstd_col(df, base, fmt="{:.4f}±{:.4f}"):
    """accuracy_mean/std -> a single 'acc' column."""
    m, s = f"{base}_mean", f"{base}_std"
    if m in df.columns:
        df = df.copy()
        df[base] = [fmt.format(a, b if pd.notna(b) else 0.0)
                    for a, b in zip(df[m], df.get(s, [0] * len(df)))]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()
    out = Path(args.out_dir)
    L = []  # lines

    def w(s=""):
        L.append(s)

    w("# Leaf Disease Classification — Results Report")
    w()
    w("> Generated automatically (`make_report.py`). Tables come from `results/*.csv`, "
      "figures from `results/*.png`.")
    w()

    # ---------- Data set ----------
    ds = load_json(out / "dataset" / "dataset_summary.json")
    if ds:
        w("## 1. Data set")
        w()
        w(f"- **Classes / plants:** {ds['n_classes']} classes, {ds['n_plants']} plants")
        w(f"- **Images:** {ds['n_images_dedup']} (duplicate removal: "
          f"{ds['n_duplicates_removed']} dropped; raw {ds['n_images_raw']})")
        w(f"- **Imbalance ratio:** {ds['imbalance_ratio']}x "
          f"(min {ds['min_class']}, max {ds['max_class']} / class)")
        w(f"- **Per plant:** " +
          ", ".join(f"{k}={v}" for k, v in ds["images_per_plant"].items()))
        w()
        # EDA figures (those that exist)
        eda = [("class_distribution.png", "Class distribution"),
               ("samples_grid.png", "One sample image per class"),
               ("mean_image_per_class.png", "Mean image per class"),
               ("color_analysis.png", "Mean colour per class (disease signal)"),
               ("image_dimensions.png", "Image size / aspect-ratio distribution"),
               ("brightness_by_plant.png", "Brightness per plant"),
               ("rgb_histogram.png", "RGB intensity distribution")]
        for fn, cap in eda:
            if (out / "dataset" / fn).exists():
                w(f"**{cap}**")
                w()
                w(f"![{cap}](dataset/{fn})")
                w()

    # ---------- Model comparison ----------
    summ = load_csv(out / "summary_internal_test.csv")
    if summ is not None:
        w("## 2. Model comparison (internal test)")
        w()
        best = summ.iloc[0]
        w(f"**Best model: `{best['model']}`** — accuracy "
          f"{best.get('accuracy_mean', float('nan')):.4f} ± "
          f"{best.get('accuracy_std', 0):.4f}, "
          f"macro-F1 {best.get('f1_macro_mean', float('nan')):.4f}.")
        w()
        d = meanstd_col(summ, "accuracy")
        d = meanstd_col(d, "f1_macro")
        cols = ["model", "accuracy", "f1_macro", "auc_macro_ovr_mean",
                "sensitivity_macro_mean", "specificity_macro_mean", "g_mean_macro_mean"]
        heads = ["Model", "Accuracy", "F1-macro", "AUC", "Sensitivity",
                 "Specificity", "G-mean"]
        pair = [(c, h) for c, h in zip(cols, heads) if c in d.columns]
        w(df_to_md(d, [c for c, _ in pair], [h for _, h in pair]))
        w("![Model ranking](ranking_test.png)")
        w()

    # ---------- External validation ----------
    ext = load_csv(out / "summary_external.csv")
    if ext is not None:
        info = load_json(out / "external_info.json")
        w("## 3. External (cross-dataset) validation")
        w()
        if info:
            w(f"Independent set: {info['n_images']} images, "
              f"{len(info['covered_classes'])} classes covered "
              f"({', '.join(info['covered_classes'])}).")
            w()
        d = meanstd_col(ext, "accuracy")
        d = meanstd_col(d, "f1_macro")
        w(df_to_md(d, ["model", "accuracy", "f1_macro"],
                   ["Model", "Accuracy", "F1-macro"]))
        w()

    # ---------- Complexity / efficiency ----------
    cx = load_csv(out / "complexity.csv")
    if cx is not None:
        w("## 4. Model complexity and inference speed")
        w()
        w(df_to_md(cx, ["model", "params_M", "size_MB", "gflops", "latency_ms",
                        "fps", "train_sec_mean", "accuracy_mean"],
                   ["Model", "Params (M)", "Size (MB)", "GFLOPs", "Latency (ms)",
                    "FPS", "Training (s)", "Accuracy"], floatfmt="{:.2f}"))
        w("![Efficiency](efficiency_test.png)")
        w()

    # ---------- Statistical analysis ----------
    stats = load_json(out / "stats_report.json")
    mc = load_json(out / "mcnemar.json")
    w("## 5. Statistical analysis")
    w()
    if stats and stats.get("friedman_nemenyi"):
        fn = stats["friedman_nemenyi"]
        w(f"- **Friedman test:** χ²={fn['friedman_stat']:.3f}, "
          f"p={fn['friedman_p']:.3e} ({fn['n_blocks']} blocks, {fn['k_models']} models). "
          f"Nemenyi critical difference (CD)={fn['nemenyi_cd']:.3f}.")
        w("- See the critical difference diagram for the average-rank differences.")
        w()
        w("![Critical difference diagram](cd_diagram.png)")
        w()
    pv = load_csv(out / "proposed_vs_baselines.csv")
    if pv is not None and "significant_holm" in pv.columns:
        n_sig = int(pv["significant_holm"].sum())
        w(f"- **Proposed vs baselines (Wilcoxon, Holm-Bonferroni corrected):** "
          f"the difference is significant (p<0.05) for {n_sig} of {len(pv)} baselines.")
    if mc and mc.get("per_seed"):
        ps = mc["per_seed"][0]
        w(f"- **McNemar (proposed vs best baseline `{mc['best_baseline']}`):** "
          f"p={ps['p_value']:.3e}. 95% CI of the proposed accuracy: "
          f"[{ps['proposed_acc_ci95'][1]:.4f}, {ps['proposed_acc_ci95'][2]:.4f}].")
    w()

    # ---------- Ablation ----------
    ab = load_csv(out / "ablation_summary.csv")
    if ab is not None:
        w("## 6. Ablation study")
        w()
        w("Each row removes one component from the proposed model; "
          "a negative `delta_vs_full` means the component contributes.")
        w()
        cols = ["variant", "accuracy_mean", "accuracy_std", "delta_vs_full",
                "wilcoxon_p_holm", "cohens_d"]
        heads = ["Variant", "Accuracy", "Std", "Δ (vs full)", "p (Holm)", "Cohen d"]
        pair = [(c, h) for c, h in zip(cols, heads) if c in ab.columns]
        w(df_to_md(ab, [c for c, _ in pair], [h for _, h in pair]))
        w()

    # ---------- Explainability ----------
    gc = list(out.rglob("gradcam_grid.png"))
    lap = list(out.rglob("lap_attention_grid.png"))
    tsne = list(out.glob("proposed/*/seed*/tsne_test.png"))
    if gc or lap or tsne:
        w("## 7. Explainability (XAI)")
        w()
        if gc:
            w("**Grad-CAM** (fusion layer, stride-16 resolution):")
            w()
            w(f"![Grad-CAM]({gc[0].relative_to(out).as_posix()})")
            w()
        if lap:
            w("**LAP attention map** — the lesion weights the model learned itself during "
              "pooling (needs no gradients, evidence independent of Grad-CAM):")
            w()
            w(f"![LAP attention]({lap[0].relative_to(out).as_posix()})")
            w()
        if tsne:
            w(f"![t-SNE feature space]({tsne[0].relative_to(out).as_posix()})")
            w()

    w("## 8. Reproducibility")
    w()
    w("- Fixed seeds (per-run global RNG), stratified splits, content-hash duplicate removal.")
    w("- Within a block all models use the same train/val/test split "
      "(paired tests are valid).")
    w("- Full settings in `config.json` of every run, raw predictions in `predictions_*.npz`.")
    w()

    report_path = out / "REPORT.md"
    report_path.write_text("\n".join(L), encoding="utf-8")
    print(f"Report written: {report_path}")
    print(f"  ({len(L)} lines, {sum(1 for x in L if x.startswith('##'))} sections)")


if __name__ == "__main__":
    main()
