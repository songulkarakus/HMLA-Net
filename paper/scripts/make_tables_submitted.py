"""Regenerates the bodies of Tables 3, 7 and 8 of the submitted version (kept unchanged in the revision)
from the archived replication-A outputs, with the generator of the submitted version, and checks that
each regenerated body is identical to the preserved block in scripts/tables_submitted/.

Inputs: replication_A/summary/summary_internal_test.csv (Table 3), submitted_version/_ablation_both.csv
(Table 7), replication_A/runs/proposed/*/per_class_*.csv (Table 8) and submitted_version/numbers.json
(caption numbers). Outputs: output/tables_submitted_regenerated/tab_{internal,ablation,perclass}.tex.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

MAN = Path(__file__).resolve().parents[1]
A = MAN / "replication_A"
SUB = MAN / "submitted_version"
OUT = MAN / "output" / "tables_submitted_regenerated"
OUT.mkdir(parents=True, exist_ok=True)
REF = MAN / "scripts" / "tables_submitted"
N = json.loads((SUB / "numbers.json").read_text())
PROPOSED_RUNS = A / "runs" / "proposed" / "Proposed_HMLA_CAFormerS18"
SEEDS = (42, 1234, 2024)

PROPOSED = "Proposed_HMLA_CAFormerS18"

BACKBONE = "caformer_s18.sail_in22k_ft_in1k"

NICE = {
    PROPOSED: r"\textbf{HMLA-Net (ours)}",
    "caformer_s18.sail_in22k_ft_in1k": r"CAFormer-S18\,$^{\dagger}$",
    "caformer_s18-sail_in22k_ft_in1k": r"CAFormer-S18\,$^{\dagger}$",
    "swin_small_patch4_window7_224": "Swin-S", "swin_tiny_patch4_window7_224": "Swin-T",
    "vit_base_patch16_224": "ViT-B/16", "vit_small_patch16_224": "ViT-S/16",
    "deit3_small_patch16_224": "DeiT III-S", "legacy_seresnext50_32x4d": "SE-ResNeXt-50",
    "resnext50_32x4d": "ResNeXt-50", "legacy_xception": "Xception",
    "mobilenetv3_large_100": "MobileNetV3-L", "mobilenetv2_100": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0", "efficientnet_b3": "EfficientNet-B3",
    "mnasnet_100": "MnasNet-A1", "regnety_016": "RegNetY-1.6GF",
    "ghostnet_100": "GhostNet", "convnext_tiny": "ConvNeXt-T",
    "convnext_small": "ConvNeXt-S", "densenet121": "DenseNet-121",
    "densenet201": "DenseNet-201", "resnet50": "ResNet-50", "resnet101": "ResNet-101",
    "inception_v3": "Inception-v3", "vgg16": "VGG-16", "vgg19": "VGG-19",
}

FAMILY = {
    "resnet50": "CNN", "resnet101": "CNN", "densenet121": "CNN", "densenet201": "CNN",
    "vgg16": "CNN", "vgg19": "CNN", "inception_v3": "CNN", "legacy_xception": "CNN",
    "resnext50_32x4d": "CNN", "legacy_seresnext50_32x4d": "CNN",
    "mobilenetv3_large_100": "eff.", "mobilenetv2_100": "eff.",
    "efficientnet_b0": "eff.", "efficientnet_b3": "eff.", "mnasnet_100": "eff.",
    "regnety_016": "eff.", "ghostnet_100": "eff.",
    "convnext_tiny": "CNN$^{*}$", "convnext_small": "CNN$^{*}$",
    "vit_small_patch16_224": "ViT", "vit_base_patch16_224": "ViT",
    "swin_tiny_patch4_window7_224": "ViT", "swin_small_patch4_window7_224": "ViT",
    "deit3_small_patch16_224": "ViT",
    "caformer_s18.sail_in22k_ft_in1k": "hybrid",
    "caformer_s18-sail_in22k_ft_in1k": "hybrid",
    PROPOSED: "hybrid",
}


def per_class_tables():
    per = []
    for s in SEEDS:
        d = pd.read_csv(PROPOSED_RUNS / f"seed{s}" / "per_class_external_mld24.csv")
        d = d[d.support > 0].copy()
        d["seed"] = s
        per.append(d)
    per = pd.concat(per)
    g = per.groupby("class").agg(prec=("precision", "mean"), prec_sd=("precision", "std"),
                                 rec=("recall_sensitivity", "mean"), rec_sd=("recall_sensitivity", "std"),
                                 f1=("f1-score", "mean"), f1_sd=("f1-score", "std"),
                                 support=("support", "mean")).reset_index()
    g = g.sort_values("f1", ascending=False)
    g.to_csv(OUT / "_per_class_external.csv", index=False)
    pi = []
    for s in SEEDS:
        d = pd.read_csv(PROPOSED_RUNS / f"seed{s}" / "per_class_test.csv")
        d["seed"] = s
        pi.append(d)
    pi = pd.concat(pi)
    gi = pi.groupby("class").agg(prec=("precision", "mean"), rec=("recall_sensitivity", "mean"),
                                 f1=("f1-score", "mean"), f1_sd=("f1-score", "std"),
                                 support=("support", "mean")).reset_index()
    gi.to_csv(OUT / "_per_class_internal.csv", index=False)


RESULTS = {}


def emit(name, body):
    out = OUT / f"{name}.tex"
    text = body.rstrip() + "\n"
    out.write_text(text)
    lines = (REF / f"{name}.tex").read_text().splitlines()
    ref_body = "\n".join(lines[1:-1]).rstrip() + "\n"
    RESULTS[name] = (text == ref_body)
    print(f"  {name:14s} -> {out.relative_to(MAN)}  "
          + ("== tables_submitted (identical)" if RESULTS[name] else "!! DIFFERS from tables_submitted"))


def ms(m, s, d=4, bold=False):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "--"
    txt = f"{m:.{d}f}"
    if s is not None and not (isinstance(s, float) and np.isnan(s)):
        txt += rf"\,\tiny$\pm${s:.{d}f}"
    return rf"\textbf{{{txt}}}" if bold else txt


def wrap(body, cols, caption, label, notes, small=r"\footnotesize", wide=False):
    W = r"\fulllength" if wide else r"\textwidth"
    L = [r"\begin{table}[H]",
         rf"\caption{{{caption}\label{{{label}}}}}"]
    if wide:
        L.append(r"\begin{adjustwidth}{-\extralength}{0cm}")
    L += [small, r"\setlength{\tabcolsep}{3pt}",
          rf"\begin{{tabularx}}{{{W}}}{{{cols}}}",
          r"\toprule", body, r"\bottomrule", r"\end{tabularx}"]
    if wide:
        L.append(r"\end{adjustwidth}")
    L += [rf"\noindent{{\footnotesize{{{notes}}}}}", r"\end{table}", ""]
    return "\n".join(L)


def t_internal():
    it = pd.read_csv(A / "summary" / "summary_internal_test.csv")
    it = it.sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)
    best = {c: it[c].max() for c in ["accuracy_mean", "f1_macro_mean",
                                     "balanced_accuracy_mean", "mcc_mean",
                                     "auc_macro_ovr_mean"]}
    best["ece_mean"] = it["ece_mean"].min()
    rows = []
    for i, t in it.iterrows():
        b = t.model == PROPOSED
        rows.append(" & ".join([
            f"{i + 1}", NICE[t.model], FAMILY[t.model],
            ms(t.accuracy_mean, t.accuracy_std, 4, b),
            ms(t.f1_macro_mean, t.f1_macro_std, 4, b),
            ms(t.balanced_accuracy_mean, None, 4, b),
            ms(t.mcc_mean, None, 4, b),
            ms(t.auc_macro_ovr_mean, None, 4, b),
            ms(t.ece_mean, None, 3, b),
            (rf"\textbf{{{t.params_M:.1f}}}" if b else f"{t.params_M:.1f}"),
            (rf"\textbf{{{t.gflops:.2f}}}" if b else f"{t.gflops:.2f}"),
        ]) + r" \\")
    head = (r"\textbf{\#} & \textbf{Model} & \textbf{Family} & \textbf{Accuracy} & "
            r"\textbf{Macro-F1} & \textbf{Bal.\ acc.} & \textbf{MCC} & \textbf{AUC} & "
            r"\textbf{ECE} & \textbf{Par.\ (M)} & \textbf{GFLOPs} \\" + "\n" + r"\midrule")
    body = head + "\n" + "\n".join(rows)
    notes = (r"Mean $\pm$ standard deviation over three seeds (42, 1234, 2024); "
             r"within a seed every model receives the identical 70/15/15 partition. "
             r"Models are ordered by macro-F1. "
             r"eff.\ = efficiency-oriented CNN; CNN$^{*}$ = modernised CNN (ConvNeXt); "
             r"hybrid = interleaved convolution and attention. "
             r"$^{\dagger}$ the bare backbone of the proposed model, included so that the "
             r"contribution of the added components can be separated from the contribution "
             r"of the backbone. "
             r"ECE = expected calibration error (15 bins, lower is better); AUC is macro "
             r"one-vs-rest. Complexity is measured at $224\times224$ with a batch of one.")
    emit("tab_internal", wrap(
        body, "@{}r L l c c c c c c r r@{}",
        "Held-out internal test performance of the 26 architectures on the four-crop, "
        "21-class corpus. The whole benchmark spans "
        f"{N['internal']['f1_spread_pp']:.2f}~pp of macro-F1 and the top eight models "
        f"span {N['internal']['gap_to_8th_pp']:.2f}~pp; Section~\\ref{{sec:res-stat}} "
        "shows that no pair of models in this table can be separated statistically.",
        "tab:internal", notes, wide=True))


def t_ablation():
    ab = pd.read_csv(SUB / "_ablation_both.csv")
    pretty = {"full": r"\emph{full model}",
              "wo_msf": "MSF (multi-scale fusion)", "wo_lap": "LAP (lesion-aware pooling)",
              "wo_lap_gate": "LAP gating and multi-head attention",
              "wo_hier": "auxiliary plant head",
              "wo_mixstyle": "MixStyle", "wo_beta_calib": r"$\beta$ calibration",
              "wo_mixup": "MixUp / CutMix", "wo_mixup_cooldown": "MixUp cool-down",
              "wo_ema": "EMA weights", "wo_tta": "flip test-time augmentation",
              "wo_llrd": "layer-wise learning-rate decay",
              "wo_sampler": "class-balanced sampler",
              "wo_random_erasing": "random erasing",
              "wo_label_smoothing": "label smoothing",
              "wo_two_phase": "two-phase schedule",
              "wo_pretrained": "ImageNet pre-training"}
    arch = {"wo_msf", "wo_lap", "wo_lap_gate", "wo_hier", "wo_mixstyle"}
    ab = ab.sort_values("delta_ext_pp", ascending=False)
    full = ab[ab.variant == "full"].iloc[0]
    rows = [" & ".join([pretty["full"], ms(full.f1_macro_mean_int, full.f1_macro_std_int),
                        "--", ms(full.f1_macro_mean_ext, full.f1_macro_std_ext),
                        "--", f"{full.accuracy_mean:.4f}", "--"]) + r" \\", r"\midrule"]
    for t in ab.itertuples():
        if t.variant == "full":
            continue
        nm = pretty[t.variant]
        if t.variant in arch:
            nm = rf"\textbf{{{nm}}}"
        rows.append(" & ".join([
            nm, ms(t.f1_macro_mean_int, t.f1_macro_std_int),
            f"{t.delta_int_pp:+.2f}",
            ms(t.f1_macro_mean_ext, t.f1_macro_std_ext),
            f"{t.delta_ext_pp:+.2f}",
            f"{t.accuracy_mean:.4f}", f"{t.delta_ext_acc_pp:+.2f}",
        ]) + r" \\")
    head = (r"\multirow{2}{*}{\textbf{Component removed}} & "
            r"\multicolumn{2}{c}{\textbf{Internal test}} & "
            r"\multicolumn{4}{c}{\textbf{Zero-shot MLD24}} \\"
            "\n" + r"\cmidrule(lr){2-3}\cmidrule(lr){4-7}" + "\n"
            r" & \textbf{Macro-F1} & \textbf{$\Delta$ (pp)} & \textbf{Macro-F1} & "
            r"\textbf{$\Delta$ (pp)} & \textbf{Acc.} & \textbf{$\Delta$ (pp)} \\" + "\n" + r"\midrule")
    notes = (r"Every variant is trained from scratch with the same three seeds and "
             r"the same partitions as the full model. $\Delta$ is the metric of the full "
             r"model minus the metric of the variant, so a \emph{positive} $\Delta$ means "
             r"the removed component contributes. Architectural components are set in "
             r"bold; the remainder belong to the training recipe. "
             r"No $\Delta$ on the internal partition survives a Holm--Bonferroni-corrected "
             r"Wilcoxon test, and with three paired blocks the smallest attainable "
             r"two-sided $p$ is 0.25, so the internal column is reported as a descriptive "
             r"measurement rather than a test. "
             r"\emph{MixUp cool-down} completed two of the three seeds on the "
             r"internal partition and all three on MLD24.")
    emit("tab_ablation", wrap(
        head + "\n" + "\n".join(rows), "@{}L c c c c c c@{}",
        "Component ablation measured on both partitions. Removing a component changes "
        f"internal macro-F1 by at most {abs(ab[ab.variant != 'wo_pretrained'].delta_int_pp).max():.2f}~pp "
        "once pre-training is excluded, which is less than five times the run-to-run "
        f"noise of the pipeline ({N['noise_floor']['int_diff_pp']:.2f}~pp), whereas the same "
        "removals cost up to "
        f"{ab[ab.variant != 'wo_pretrained'].delta_ext_pp.max():.2f}~pp under zero-shot "
        "transfer. The two columns of $\\Delta$ are only weakly related (Spearman "
        f"$\\rho = {N['ablation']['spearman_int_ext_delta']:.2f}$, "
        f"$p = {N['ablation']['spearman_int_ext_delta_p']:.2f}$).",
        "tab:ablation", notes))


def t_perclass():
    ext = pd.read_csv(OUT / "_per_class_external.csv")
    inn = pd.read_csv(OUT / "_per_class_internal.csv").set_index("class")
    ext = ext.sort_values("f1", ascending=False)
    rows = []
    for t in ext.itertuples():
        c = t._1
        i = inn.loc[c]
        nm = c.replace("mango ", "")
        if c == "mango sooty mould":
            nm = rf"\textbf{{{nm}}}"
        rows.append(" & ".join([
            nm, f"{i.support:.0f}", f"{i.prec:.3f}", f"{i.rec:.3f}", f"{i.f1:.3f}",
            "800", f"{t.prec:.3f}", f"{t.rec:.3f}",
            rf"{t.f1:.3f}\,\tiny$\pm${t.f1_sd:.3f}",
        ]) + r" \\")
    head = (r"\multirow{2}{*}{\textbf{Mango class}} & \multicolumn{4}{c}{\textbf{Internal test}} & "
            r"\multicolumn{4}{c}{\textbf{Zero-shot MLD24}} \\" + "\n" +
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}" + "\n" +
            r" & \textbf{$n$} & \textbf{Prec.} & \textbf{Rec.} & \textbf{F1} & \textbf{$n$} & "
             r"\textbf{Prec.} & \textbf{Rec.} & \textbf{F1} \\" + "\n" +
            r"\midrule")
    notes = (r"Mean over three seeds; internal support is the mean test-partition "
             r"count. The six classes are those MLD24 shares with our label space. "
             r"\emph{sooty mould} is classified perfectly in-domain and is essentially "
             r"unrecognised out-of-domain (recall 0.011 / 0.000 / 0.005 in the three seeds) "
             r"while the other five average 0.838. Section~\ref{sec:res-perclass} shows "
             r"that a collapse of one class to exactly zero recall is shared by 19 of the "
             r"26 architectures, which points to a difference in what the two corpora "
             r"label rather than to a property of any one model.")
    emit("tab_perclass", wrap(
        head + "\n" + "\n".join(rows), "@{}L r c c c r c c c@{}",
        "Per-class behaviour of HMLA-Net on the six mango classes that both corpora share.",
        "tab:perclass", notes))


if __name__ == "__main__":
    per_class_tables()
    for fn in (t_internal, t_ablation, t_perclass):
        fn()
    if not all(RESULTS.values()):
        raise SystemExit("regenerated table bodies differ from the preserved blocks")
    print("  Tables 3, 7 and 8 of the submitted version regenerated and identical to the preserved blocks.")
