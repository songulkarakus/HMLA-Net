"""Regenerates the LaTeX bodies of the generated tables of the manuscript from output/numbers.json and
output/derived/ -> output/tables/<table>.tex. Tables 3, 7 and 8 reuse the table blocks of the submitted
version (scripts/tables_submitted/), with their captions and notes regenerated.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MAN = Path(__file__).resolve().parents[1]
ROOT = MAN
A = MAN / "replication_A"
DER = MAN / "output" / "derived"
TABLES = MAN / "output" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)
N = json.loads((MAN / "output" / "numbers.json").read_text())

PROPOSED = "Proposed_HMLA_CAFormerS18"
BACKBONE = "caformer_s18.sail_in22k_ft_in1k"
NICE = {
    PROPOSED: r"\textbf{HMLA-Net (ours)}", BACKBONE: r"CAFormer-S18\,$^{\dagger}$",
    "swin_small_patch4_window7_224": "Swin-S", "swin_tiny_patch4_window7_224": "Swin-T",
    "vit_base_patch16_224": "ViT-B/16", "vit_small_patch16_224": "ViT-S/16",
    "deit3_small_patch16_224": "DeiT III-S", "legacy_seresnext50_32x4d": "SE-ResNeXt-50",
    "resnext50_32x4d": "ResNeXt-50", "legacy_xception": "Xception",
    "mobilenetv3_large_100": "MobileNetV3-L", "mobilenetv2_100": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0", "efficientnet_b3": "EfficientNet-B3",
    "mnasnet_100": "MnasNet-A1", "regnety_016": "RegNetY-1.6GF", "ghostnet_100": "GhostNet",
    "convnext_tiny": "ConvNeXt-T", "convnext_small": "ConvNeXt-S", "densenet121": "DenseNet-121",
    "densenet201": "DenseNet-201", "resnet50": "ResNet-50", "resnet101": "ResNet-101",
    "inception_v3": "Inception-v3", "vgg16": "VGG-16", "vgg19": "VGG-19",
}
FAMILY = {
    "resnet50": "CNN", "resnet101": "CNN", "densenet121": "CNN", "densenet201": "CNN", "vgg16": "CNN",
    "vgg19": "CNN", "inception_v3": "CNN", "legacy_xception": "CNN", "resnext50_32x4d": "CNN",
    "legacy_seresnext50_32x4d": "CNN", "mobilenetv3_large_100": "eff.", "mobilenetv2_100": "eff.",
    "efficientnet_b0": "eff.", "efficientnet_b3": "eff.", "mnasnet_100": "eff.", "regnety_016": "eff.",
    "ghostnet_100": "eff.", "convnext_tiny": "CNN$^{*}$", "convnext_small": "CNN$^{*}$",
    "vit_small_patch16_224": "ViT", "vit_base_patch16_224": "ViT", "swin_tiny_patch4_window7_224": "ViT",
    "swin_small_patch4_window7_224": "ViT", "deit3_small_patch16_224": "ViT", BACKBONE: "hybrid", PROPOSED: "hybrid",
}


def nice(m):
    if m.endswith(" [matched]"):
        return NICE[m[:-10]] + r"\,(M)"
    return NICE[m]


def fam(m):
    return FAMILY[m.replace(" [matched]", "")]


def emit(name, body):
    out = TABLES / f"{name}.tex"
    out.write_text(body.rstrip() + "\n")
    print(f"  {name:16s} -> {out.relative_to(MAN)}")


def ms(m, s, d=4, bold=False):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "--"
    txt = f"{m:.{d}f}"
    if s is not None and not (isinstance(s, float) and np.isnan(s)):
        txt += rf"\,\tiny$\pm${s:.{d}f}"
    return rf"\textbf{{{txt}}}" if bold else txt


def wrap(body, cols, caption, label, notes, small=r"\footnotesize", wide=False):
    W = r"\fulllength" if wide else r"\textwidth"
    L = [r"\begin{table}[H]", rf"\caption{{{caption}\label{{{label}}}}}"]
    if wide:
        L.append(r"\begin{adjustwidth}{-\extralength}{0cm}")
    L += [small, r"\setlength{\tabcolsep}{3pt}", rf"\begin{{tabularx}}{{{W}}}{{{cols}}}", r"\toprule", body,
          r"\bottomrule", r"\end{tabularx}"]
    if wide:
        L.append(r"\end{adjustwidth}")
    L += [rf"\noindent{{\footnotesize{{{notes}}}}}", r"\end{table}", ""]
    return "\n".join(L)


def t_dataset():
    cd = pd.read_csv(A / "dataset_eda" / "class_distribution.csv")
    lines, tot = [], 0
    for plant, g in cd.groupby("plant"):
        names = ", ".join(sorted(c.split(" ", 1)[1] for c in g["class"]))
        lines.append(" & ".join([plant.capitalize(), str(len(g)), f"{g.n_images.sum():,}".replace(",", "{,}"),
                                 f"{g.n_images.min()}--{g.n_images.max()}", rf"\scriptsize {names}"]) + r" \\")
        tot += g.n_images.sum()
    lines.append(r"\midrule")
    lines.append(" & ".join([r"\textbf{Total}", r"\textbf{21}", rf"\textbf{{{tot:,}}}".replace(",", "{,}"),
                             r"\textbf{209--500}", ""]) + r" \\")
    head = (r"\textbf{Crop} & \textbf{Classes} & \textbf{Images} & \textbf{Range} & \textbf{Disease classes} \\" + "\n" + r"\midrule")
    notes = (r"Counts are after content-hash (MD5) de-duplication, which removed 76 exact duplicates from the 7{,}255 raw "
             r"files. The imbalance ratio is 2.39:1. Every class is a folder of the form \texttt{<crop> leaf/<crop> <condition>}; "
             r"the crop prefix is what the auxiliary crop head of Section~\ref{sec:arch} predicts. Class names are quoted "
             r"as the corpus defines them (it writes \emph{jackfruit sooty mold}); the text uses \emph{mould}.")
    emit("tab_dataset", wrap(head + "\n" + "\n".join(lines), "@{}l r r r L@{}", "Composition of the four-crop corpus.",
                             "tab:dataset", notes, small=r"\small"))


def _orig_block(name):
    src = (MAN / "scripts" / "tables_submitted" / f"{name}.tex").read_text()
    lines = src.splitlines()
    return "\n".join(lines[1:-1]) + "\n"


def t_internal():
    emit("tab_internal", _orig_block("tab_internal"))


def t_external():
    ex = pd.read_csv(DER / "_external_A.csv")
    rows = []
    for i, t_ in ex.iterrows():
        b = t_.model == PROPOSED
        rows.append(" & ".join([
            f"{i + 1}", f"{int(t_.rank_int)}", nice(t_.model),
            ms(t_.accuracy_mean, t_.accuracy_std, 4, b), ms(t_.f1_macro_mean, t_.f1_macro_std, 4, b),
            ms(t_.P3_f1, None, 4, b),
            (rf"\textbf{{{100 * t_.leakage:.1f}}}" if b else f"{100 * t_.leakage:.1f}"),
            ms(t_.auc_macro_ovr_mean, None, 4, b), ms(t_.ece_mean, None, 3, b),
            (rf"\textbf{{{t_.gmean:.3f}}}" if b else f"{t_.gmean:.3f}"),
            (rf"\textbf{{{t_.drop_acc_pp:.1f}}}" if b else f"{t_.drop_acc_pp:.1f}")]) + r" \\")
    head = (r"\textbf{\#} & \textbf{\#$_{\mathrm{int}}$} & \textbf{Model} & \textbf{Accuracy} & \textbf{Macro-F1} & "
            r"\textbf{F1$_{\mathrm{restr.}}$} & \textbf{Leak (\%)} & \textbf{AUC} & \textbf{ECE} & \textbf{G-mean} & "
            r"\textbf{$\Delta$acc.\ (pp)} \\" + "\n" + r"\midrule")
    A_ = N["A"]
    notes = (r"Zero-shot: the checkpoints of Table~\ref{tab:internal} are applied to MLD24 without any training, "
             r"fine-tuning or target-domain statistic. Because MLD24 covers 6 of the 21 labels, every macro average here is "
             r"taken over those six classes and is therefore \emph{not} of the same cardinality as the 21-class macro "
             r"averages of Table~\ref{tab:internal} (Section~\ref{sec:metrics}). Models are ordered by accuracy; "
             r"\#$_{\mathrm{int}}$ is the rank of the same model in Table~\ref{tab:internal}. F1$_{\mathrm{restr.}}$ is "
             r"macro-F1 when the logits are restricted to the six shared classes (protocol P3, a separate experiment); Leak "
             r"is the share of MLD24 images assigned to one of the 15 labels absent from MLD24. $\Delta$acc.\ is the "
             r"accuracy lost between the partitions. Because the six classes are equally sized, accuracy, balanced accuracy "
             r"and macro-recall coincide. G-mean is the geometric mean of the six per-class recalls: one class at zero recall "
             r"drives it below 0.01 whatever the remaining recalls. The replicate-to-replicate difference of the same configuration on this "
             rf"partition is {N['replication']['external_f1']['mean_abs_pp']:.1f}~pp of macro-F1 on average "
             r"(Section~\ref{sec:res-replication}), so the leading group is not internally ordered.")
    emit("tab_external", wrap(
        "\n".join([head] + rows), "@{}r r L c c c c c c c r@{}",
        f"Zero-shot transfer to MLD24 (4{{,}}800 images, six mango classes, an independent corpus collected in Bangladesh). "
        f"The same 26 checkpoints span {A_['external']['acc_spread_pp']:.0f}~pp of accuracy and "
        f"{A_['external']['f1_spread_pp']:.0f}~pp of macro-F1, and the internal ranking predicts the external one only weakly "
        f"(Spearman $\\rho = {A_['transfer']['spearman_acc']:.2f}$, $p = {A_['transfer']['spearman_acc_p']:.3f}$).",
        "tab:external", notes, wide=True))


def t_recipe():
    RM = N["recipe_matched"]
    rows = []
    order = [BACKBONE, "swin_small_patch4_window7_224", "legacy_seresnext50_32x4d", "efficientnet_b3", "vit_small_patch16_224"]
    byk = {(r_["split"], r_["model"]): r_ for r_ in RM["rows"]}
    for m in order:
        i, e = byk[("internal_test", m)], byk[("external", m)]
        rows.append(" & ".join([
            NICE[m], ms(i["standard"], i["standard_sd"]), ms(i["matched"], i["matched_sd"]), f"{i['recipe_effect_pp']:+.2f}",
            ms(e["standard"], e["standard_sd"]), ms(e["matched"], e["matched_sd"]), f"{e['recipe_effect_pp']:+.2f}",
            f"{e['architecture_effect_pp']:+.2f}"]) + r" \\")
    rows.append(r"\midrule")
    rows.append(" & ".join([r"\textbf{HMLA-Net (ours)}", "--", f"{N['internal']['proposed_f1']:.4f}", "--", "--",
                            f"{N['external']['proposed_f1']:.4f}", "--", "--"]) + r" \\")
    head = (r"\multirow{2}{*}{\textbf{Backbone}} & \multicolumn{3}{c}{\textbf{Internal test macro-F1}} & "
            r"\multicolumn{4}{c}{\textbf{Zero-shot MLD24 macro-F1}} \\" + "\n" + r"\cmidrule(lr){2-4}\cmidrule(lr){5-8}" + "\n" +
            r" & \textbf{generic} & \textbf{HMLA recipe} & \textbf{$\Delta$ (pp)} & \textbf{generic} & "
            r"\textbf{HMLA recipe} & \textbf{$\Delta$ (pp)} & \textbf{HMLA-Net $-$ backbone (pp)} \\" + "\n" + r"\midrule")
    T = N["replication"]["hmla_triplicate"]["external_f1"]
    C = N["replication"]["caformer_matched_duplicate"]["external_f1"]
    notes = (r"Replication B, mean (\tiny$\pm$\footnotesize s.d.) over the same three seeds and partitions. \emph{generic} is the common recipe of "
             r"Section~\ref{sec:baselines}; \emph{HMLA recipe} adds MixUp/CutMix, EMA weights, layer-wise learning-rate "
             r"decay and $V_4$ flip test-time augmentation, i.e.\ the complete training and inference recipe of the "
             r"proposed model with its architectural components (MSF, LAP, crop head, MixStyle) switched off. $\Delta$ "
             r"is HMLA recipe minus generic (positive: the recipe helps). The last column is HMLA-Net minus the backbone "
             r"under the identical recipe, the controlled comparison of Section~\ref{sec:res-recipe}; against CAFormer-S18 it is "
             rf"{RM['ext_arch_effect_vs_caformer_pp']:+.2f}~pp, and with all replication-B runs pooled "
             rf"(HMLA-Net {T['mean9']:.4f} over nine runs, CAFormer-S18 with the HMLA recipe {C['mean6']:.4f} over six) it "
             rf"is {RM['pooled_arch_effect_pp']:+.2f}~pp, inside the replicate-to-replicate variation of "
             rf"{N['replication']['external_f1']['mean_abs_pp']:.1f}~pp. None of the {RM['n_tests']} paired comparisons "
             r"survives Holm correction (three blocks; Section~\ref{sec:stats}).")
    emit("tab_recipe", wrap(
        "\n".join([head] + rows), "@{}L c c c c c c c@{}",
        "Recipe-matched comparison. Five backbones were retrained with the complete HMLA-Net recipe so that the "
        "training and inference recipe is identical between them and the proposed model.",
        "tab:recipe", notes, wide=True))


def t_replication():
    R = N["replication"]
    rows = []
    for key, lab in [("internal_f1", "internal test, macro-F1"), ("internal_acc", "internal test, accuracy"),
                     ("external_f1", "zero-shot MLD24, macro-F1"), ("external_acc", "zero-shot MLD24, accuracy")]:
        s = R[key]
        rows.append(" & ".join([lab, f"{s['n_pairs']}", f"{s['mean_abs_pp']:.2f}", f"{s['sd_pp']:.2f}",
                                f"{s['max_abs_pp']:.2f}", f"{s['mean_abs_of_means_pp']:.2f}",
                                f"{s['max_abs_of_means_pp']:.2f}", f"{s['spread_A_pp']:.2f}", f"{s['spread_B_pp']:.2f}",
                                f"{s['spearman_AB']:.2f}", f"{s['max_rank_move']}"]) + r" \\")
    head = (r"\textbf{Partition, metric} & \textbf{$n$} & \textbf{mean $|\Delta|$} & \textbf{s.d.($\Delta$)} & "
            r"\textbf{max $|\Delta|$} & \textbf{mean $|\bar{\Delta}|$} & \textbf{max $|\bar{\Delta}|$} & "
            r"\textbf{spread A} & \textbf{spread B} & \textbf{$\rho_{AB}$} & "
            r"\textbf{max rank move} \\" + "\n" + r"\midrule")
    T = R["hmla_triplicate"]
    notes = (r"Two complete, independent executions of the 26 generic-recipe configurations (25 backbones and the "
             r"proposed model, three seeds each): replication A produced the original submission, replication B this "
             r"revision. Same code, recorded hyper-parameters, partitions and hardware; non-deterministic cuDNN "
             r"kernels and data-loader ordering are the expected sources of the difference. $\Delta$ is A minus B for the "
             r"same configuration, seed and partition, in percentage points, and s.d.($\Delta$) is the standard deviation "
             r"of the signed $\Delta$; $\bar{\Delta}$ is A minus B for the three-seed means of a configuration, the "
             r"quantity on which the tables of this paper compare models; \emph{spread} is the range of the 26 "
             r"three-seed means; "
             r"$\rho_{AB}$ is the Spearman correlation between the two orderings; \emph{max rank move} is the largest "
             r"change of rank of any configuration. HMLA-Net itself was trained three times in replication B "
             r"(benchmark entry, ablation reference and a dedicated repeat); its nine runs give internal macro-F1 "
             rf"{T['internal_f1']['mean9']:.4f}\,$\pm$\,{T['internal_f1']['sd9']:.4f} and external macro-F1 "
             rf"{T['external_f1']['mean9']:.4f}\,$\pm$\,{T['external_f1']['sd9']:.4f}, with the three-seed means of the "
             rf"replicates at {', '.join(f'{v:.4f}' for v in T['external_f1']['replicate_means'])} externally.")
    emit("tab_replication", wrap(
        "\n".join([head] + rows), "@{}L r c c c c c c c c c@{}",
        "Replicate-to-replicate variation measured by re-executing the whole benchmark.", "tab:replication", notes))


def t_ablation():
    body = _orig_block("tab_ablation")
    body = re.sub(r"\\caption\{Component ablation measured on both partitions\..*?\\label\{tab:ablation\}\}",
                  lambda m: (r"\caption{Component ablation of replication A measured on both partitions. Removing a component "
                             r"changes internal macro-F1 by at most 0.29~pp once pre-training is excluded, whereas the same "
                             r"removals change zero-shot macro-F1 by up to 5.06~pp; the two columns of $\Delta$ are only weakly "
                             r"related (Spearman $\rho = 0.44$, $p = 0.08$), and the external ordering is not reproduced by the "
                             r"second execution (Section~\ref{sec:res-ablation}).\label{tab:ablation}}"), body, count=1, flags=re.S)
    assert "noise of the pipeline" not in body
    body = body.replace("MixUp cool-down & 0.9846", r"MixUp cool-down\,$^{\ddagger}$ & 0.9846", 1)
    body = body.replace(r"\emph{MixUp cool-down} completed two of the three seeds on the internal partition and all three on MLD24.",
                        r"$^{\ddagger}$ \emph{MixUp cool-down} completed two of the three seeds on the internal partition and all "
                        r"three on MLD24; its internal value is therefore not fully paired and is excluded from the paired "
                        r"statistics. In the second execution of the benchmark (Section~\ref{sec:res-replication}) all 16 removals "
                        r"completed all three seeds; that execution's ablation is provided as supplementary material and its "
                        r"external ordering of the components correlates with this one at Spearman $\rho = "
                        + f"{N['ablation']['spearman_AB_ext_delta']:.2f}" + r"$ under either reference definition of Section~\ref{sec:res-ablation}.")
    emit("tab_ablation", body)


def t_tta():
    T = N["tta"]
    names = {"none": "trivial group $\\{e\\}$ (no TTA)", "klein": "Klein four-group $V_4$ (flips)",
             "dihedral8": "dihedral group $D_4$ (flips and rotations)"}
    rows = []
    for r_ in T["rows"]:
        inv = T["invariance"][r_["group"]] if r_["group"] in T["invariance"] else None
        rows.append(" & ".join([
            names[r_["group"]], f"{r_['n_views']}", f"{r_['internal_f1_macro']:.4f}", f"{r_['external_accuracy']:.4f}",
            f"{r_['external_f1_macro']:.4f}", f"{r_['latency_ms']:.1f}", f"{r_['fps']:.1f}",
            ("--" if inv is None else f"{inv['orbit_dev']:.1e}"), ("--" if inv is None else f"{inv['raw_dev']:.3f}")]) + r" \\")
    head = (r"\textbf{Group $G$} & \textbf{$|G|$} & \textbf{Int.\ F1} & \textbf{Ext.\ acc.} & \textbf{Ext.\ F1} & "
            r"\textbf{ms} & \textbf{img/s} & \textbf{orbit dev.} & \textbf{raw dev.} \\" + "\n" + r"\midrule")
    notes = (r"One checkpoint of HMLA-Net (replication B, seed 1234) evaluated under three inference groups; latency at "
             r"a batch size of one on the RTX A4000. \emph{orbit dev.} is the largest absolute change of the orbit-averaged "
             r"posterior when the input is transformed by an element of $G$ (floating-point noise only, i.e.\ exact "
             r"invariance); \emph{raw dev.} is the same quantity for the single-view network, which is only "
             r"approximately equivariant after training-time flip augmentation. The differences between the three "
             rf"rows ({T['ext_f1_gain_klein_pp']:+.2f} and {T['ext_f1_gain_d4_pp']:+.2f}~pp of external macro-F1 for "
             r"$V_4$ and $D_4$ against no TTA) are single-run measurements and lie within the replicate-to-replicate "
             r"variation of Section~\ref{sec:res-replication}; the latency ratios are measured values for this system and batch size.")
    emit("tab_tta", wrap(
        "\n".join([head] + rows), "@{}L c c c c c c c c@{}",
        "Orbit averaging at inference: what the symmetry group costs and buys.", "tab:tta", notes))


def t_perclass():
    emit("tab_perclass", _orig_block("tab_perclass"))


def literature_row_numbers():
    vals = {"internal_accuracy_pct": f"{100 * N['A']['internal']['proposed_acc']:.2f}",
            "zero_shot_accuracy_pct": f"{100 * N['A']['external']['proposed_acc']:.2f}"}
    (TABLES / "tab_literature_this_work.json").write_text(json.dumps(vals, indent=2) + "\n")
    print("  literature table, 'This work' row ->", vals)


if __name__ == "__main__":
    for fn in (t_dataset, t_internal, t_external, t_recipe, t_replication, t_ablation, t_tta, t_perclass):
        fn()
    literature_row_numbers()
