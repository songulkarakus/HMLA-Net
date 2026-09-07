"""Regenerates the data-driven figures of the manuscript (600 dpi, Times 10/9 pt, width = \textwidth)
from output/numbers.json and output/derived/ -> output/figures/.
"""
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from PIL import Image

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

MAN = Path(__file__).resolve().parents[1]
ROOT = MAN
B = MAN / "replication_B"
A = MAN / "replication_A"
ARCH = MAN / "replication_A" / "runs"
OLDFIG = MAN / "submitted_version" / "figures"
FIG = MAN / "output" / "figures"
DER = MAN / "output" / "derived"
FIG.mkdir(parents=True, exist_ok=True)
N = json.loads((MAN / "output" / "numbers.json").read_text())

DPI = 600
FULL = 5.4567
BODY, SMALL = 10, 9
PROPOSED = "Proposed_HMLA_CAFormerS18"
BACKBONE = "caformer_s18.sail_in22k_ft_in1k"
ACC, BB, GREY, GREY_D, MATCH = "#B02418", "#1F5C99", "#9AA0A6", "#4D5156", "#C98A2B"
CROP = {"cashew": "#4E79A7", "jackfruit": "#F28E2B", "mango": "#59A14F", "tomato": "#B07AA1"}

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": BODY, "axes.titlesize": BODY, "axes.labelsize": BODY,
    "xtick.labelsize": BODY, "ytick.labelsize": BODY, "legend.fontsize": SMALL,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.5, "grid.alpha": 1.0,
    "axes.axisbelow": True, "legend.frameon": False, "legend.handlelength": 1.4,
    "figure.dpi": DPI, "savefig.dpi": DPI, "savefig.bbox": None, "savefig.pad_inches": 0.0,
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.012, "figure.constrained_layout.w_pad": 0.012,
    "figure.constrained_layout.hspace": 0.02, "figure.constrained_layout.wspace": 0.02,
})

NAMES = {
    PROPOSED: "HMLA-Net (ours)", BACKBONE: "CAFormer-S18",
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


def short(m):
    m = str(m).replace("benchmark/", "").replace("proposed/", "")
    if m.endswith(" [matched]"):
        return NAMES.get(m[:-10], m[:-10]) + " (M)"
    return NAMES.get(m, m)


def colour(m):
    if m == PROPOSED:
        return ACC
    if m == BACKBONE:
        return BB
    if str(m).endswith("[matched]"):
        return MATCH
    return GREY


def save(fig, name):
    p = FIG / name
    want = fig.get_size_inches()[0]
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    im = Image.open(p)
    got = im.size[0] / DPI
    flag = "" if abs(got - want) < 0.01 else f"  << GENISLIK SAPMASI ({want:.2f})"
    print(f"  {name:26s} {im.size[0]}x{im.size[1]} px  {got:.2f}x{im.size[1] / DPI:.2f} in{flag}")


def panel(ax, tag, dx=-0.085, dy=1.03):
    ax.text(dx, dy, tag, transform=ax.transAxes, fontweight="bold", va="bottom", ha="left")


def fig_dataset():
    cd = pd.read_csv(A / "dataset_eda" / "class_distribution.csv")
    cd = cd.sort_values(["plant", "n_images"], ascending=[True, True])
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(FULL, 4.8), gridspec_kw=dict(width_ratios=[2.15, 1]))
    fig.get_layout_engine().set(wspace=0.10)
    y = np.arange(len(cd))
    a0.barh(y, cd.n_images, color=[CROP[p] for p in cd.plant], height=0.72, edgecolor="white", linewidth=0.4)
    a0.set_yticks(y); a0.set_yticklabels([c.split(" ", 1)[1] for c in cd["class"]])
    for i, v in enumerate(cd.n_images):
        a0.text(v + 6, i, str(v), va="center", fontsize=SMALL, color=GREY_D)
    a0.set_xlabel("Images after de-duplication"); a0.set_xlim(0, 575); a0.grid(axis="y", visible=False)
    a0.legend(handles=[Line2D([], [], marker="s", ls="", ms=6, color=CROP[c], label=c.capitalize()) for c in CROP],
              loc="lower right", ncol=1, borderaxespad=0.4)
    panel(a0, "(a)", dx=-0.36)
    per = N["dataset"]["images_per_plant"]; ks = list(per)
    a1.bar(range(4), [per[k] for k in ks], color=[CROP[k] for k in ks], width=0.66, edgecolor="white", linewidth=0.4)
    ncls = {"cashew": 2, "jackfruit": 6, "mango": 6, "tomato": 7}
    for i, k in enumerate(ks):
        a1.text(i, per[k] + 45, f"{per[k]}\n{ncls[k]} cl.", ha="center", fontsize=SMALL, color=GREY_D, linespacing=1.15)
    a1.set_xticks(range(4)); a1.set_xticklabels([k.capitalize() for k in ks], rotation=30, ha="right")
    a1.set_ylabel("Images"); a1.set_ylim(0, 3250); a1.grid(axis="x", visible=False)
    panel(a1, "(b)", dx=-0.30)
    save(fig, "fig1_dataset.png")


def fig_internal_ranking():
    it = pd.read_csv(DER / "_internal_A.csv").sort_values("f1_macro_mean").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(FULL, 6.4))
    y = np.arange(len(it))
    cols = [colour(m) for m in it.model]
    bars = ax.barh(y, it.f1_macro_mean, xerr=it.f1_macro_std, height=0.68, color=cols, edgecolor="white",
                   linewidth=0.4, error_kw=dict(ecolor=GREY_D, elinewidth=0.7, capsize=1.8, capthick=0.7))
    for b, m in zip(bars, it.model):
        if str(m).endswith("[matched]"):
            b.set_hatch("////"); b.set_edgecolor("white")
    ax.set_yticks(y); ax.set_yticklabels([short(m) for m in it.model])
    for lab, m in zip(ax.get_yticklabels(), it.model):
        if m == PROPOSED:
            lab.set_color(ACC); lab.set_fontweight("bold")
        elif m == BACKBONE:
            lab.set_color(BB)
        elif str(m).endswith("[matched]"):
            lab.set_color(MATCH)
    best = it.f1_macro_mean.max()
    mean_abs = N["replication"]["internal_f1"]["mean_abs_pp"] / 100
    max_abs = N["replication"]["internal_f1"]["max_abs_pp"] / 100
    ax.add_patch(Rectangle((best - max_abs, -0.7), max_abs, len(it) - 0.1, color=ACC, alpha=0.07, lw=0, zorder=0))
    ax.add_patch(Rectangle((best - mean_abs, -0.7), mean_abs, len(it) - 0.1, color=ACC, alpha=0.12, lw=0, zorder=0))
    ax.axvline(best - max_abs, color=ACC, lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.axvline(best - mean_abs, color=ACC, lw=0.8, ls=(0, (1, 2)), zorder=1)
    n_mean = int((it.f1_macro_mean >= best - mean_abs).sum())
    n_max = int((it.f1_macro_mean >= best - max_abs).sum())
    ax.set_xlim(0.960, 0.9895)
    ax.set_xlabel("Macro-F1 on the internal test partition (mean $\\pm$ s.d., 3 seeds)")
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.8, len(it) + 1.6)
    ax.text(best - max_abs - 0.0003, len(it) + 0.35, f"max replicate\ndifference ({max_abs * 100:.2f} pp)",
            fontsize=SMALL, color=ACC, va="bottom", ha="right", linespacing=1.15)
    ax.text(best - mean_abs + 0.0003, len(it) + 0.35, f"mean ({mean_abs * 100:.2f} pp)",
            fontsize=SMALL, color=ACC, va="bottom", ha="left", linespacing=1.15)
    print(f"      [fig3] mean band {mean_abs*100:.2f} pp -> {n_mean} models; max band {max_abs*100:.2f} pp -> {n_max} models")
    save(fig, "fig3_internal_ranking.png")


def fig_cd():
    st = json.loads((A / "summary" / "stats_report.json").read_text().replace("NaN", "null"))
    fa = st["friedman_nemenyi"]
    ranks = pd.Series({NAMES.get(k, k) and k: v for k, v in fa["avg_ranks"].items()}).sort_values()
    cd = fa["nemenyi_cd"]
    fig, ax = plt.subplots(figsize=(FULL, 4.8))
    lo, hi = 1, 27
    axis_y = 0.0
    ax.plot([lo, hi], [axis_y, axis_y], color="black", lw=0.9)
    for t in range(lo, hi + 1, 2):
        ax.plot([t, t], [axis_y, axis_y + 0.11], color="black", lw=0.7)
        ax.text(t, axis_y + 0.17, str(t), ha="center", va="bottom", fontsize=SMALL)
    ax.text((lo + hi) / 2, axis_y + 0.46, "average rank over 3 seeds (lower is better)", ha="center", fontsize=BODY)
    names, vals = list(ranks.index), list(ranks.values)
    half = int(np.ceil(len(names) / 2))
    for i, (nm, rk) in enumerate(zip(names, vals)):
        left = i < half
        row = i if left else len(names) - 1 - i
        yy = -0.30 - 0.135 * row
        xend = lo - 0.5 if left else hi + 0.5
        c = colour(nm) if colour(nm) != GREY else GREY_D
        lw = 1.1 if nm in (PROPOSED, BACKBONE) else 0.65
        ax.plot([rk, rk], [axis_y, yy], color=c, lw=lw)
        ax.plot([rk, xend], [yy, yy], color=c, lw=lw)
        ax.text(xend + (-0.3 if left else 0.3), yy, f"{short(nm)} ({rk:.1f})", ha="right" if left else "left",
                va="center", fontsize=SMALL, color=c, fontweight="bold" if nm == PROPOSED else "normal")
    y_cd = axis_y + 0.68
    x0 = min(vals)
    ax.plot([x0, x0 + cd], [y_cd, y_cd], color=ACC, lw=2.2, solid_capstyle="butt")
    for xx in (x0, x0 + cd):
        ax.plot([xx, xx], [y_cd - 0.055, y_cd + 0.055], color=ACC, lw=1.1)
    ax.text(x0 + cd / 2, y_cd + 0.10, f"CD = {cd:.1f} at $\\alpha$ = 0.05 with 3 blocks: wider than the entire\n"
            f"observed rank range ({max(vals) - min(vals):.1f}); the test cannot separate any pair by design",
            ha="center", va="bottom", fontsize=SMALL, color=ACC, linespacing=1.25)
    ax.set_xlim(lo - 6.6, hi + 7.6)
    ax.set_ylim(-0.30 - 0.135 * (half - 1) - 0.16, y_cd + 0.50)
    ax.axis("off")
    save(fig, "fig4_cd_diagram.png")


def fig_transfer():
    ex = pd.read_csv(DER / "_external_A.csv")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(FULL, 5.8), gridspec_kw=dict(width_ratios=[1.06, 1]))
    fig.get_layout_engine().set(wspace=0.08)
    for _, t in ex.iterrows():
        c = colour(t.model)
        z = 5 if t.model in (PROPOSED, BACKBONE) else 2
        a0.scatter(t.int_f1, t.accuracy_mean, s=48 if z == 5 else 28, color=c, edgecolor="white", linewidth=0.6, zorder=z)
    lab = {PROPOSED: (0.9700, 0.790, "left"), BACKBONE: (0.9700, 0.585, "left"),
           "vgg16": (0.9800, 0.290, "center"), "vgg19": (0.9700, 0.235, "left"),
           "legacy_seresnext50_32x4d": (0.9865, 0.72, "center"), "efficientnet_b3": (0.9810, 0.706, "right")}
    for m, (tx, ty, ha) in lab.items():
        t = ex[ex.model == m].iloc[0]
        c = colour(m) if colour(m) != GREY else GREY_D
        a0.annotate(short(m), (t.int_f1, t.accuracy_mean), xytext=(tx, ty), fontsize=SMALL, color=c, ha=ha, va="center",
                    fontweight="bold" if m == PROPOSED else "normal",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.5, shrinkA=1, shrinkB=3.5))
    a0.set_xlabel("Macro-F1, internal test"); a0.set_ylabel("Accuracy, zero-shot MLD24")
    a0.set_ylim(0.15, 0.83); a0.set_xlim(0.9662, 0.9870); a0.set_xticks([0.970, 0.975, 0.980, 0.985])
    tr = N["A"]["transfer"]
    a0.text(0.97, 0.03, f"Spearman $\\rho$ = {tr['spearman_acc']:.2f} ($p$ = {tr['spearman_acc_p']:.3f})\n"
            f"spread {N['A']['internal']['f1_spread_pp']:.1f} pp $\\rightarrow$ {N['A']['external']['acc_spread_pp']:.0f} pp",
            transform=a0.transAxes, va="bottom", ha="right", fontsize=SMALL, color=GREY_D, linespacing=1.3)
    panel(a0, "(a)", dx=-0.20)
    ex2 = ex.sort_values("drop_acc_pp")
    y = np.arange(len(ex2))
    a1.barh(y, ex2.drop_acc_pp, height=0.70, color=[colour(m) for m in ex2.model], edgecolor="white", linewidth=0.4)
    a1.set_yticks(y); a1.set_yticklabels([short(m) for m in ex2.model], fontsize=SMALL)
    for lab_, m in zip(a1.get_yticklabels(), ex2.model):
        if m == PROPOSED:
            lab_.set_color(ACC); lab_.set_fontweight("bold")
        elif m == BACKBONE:
            lab_.set_color(BB)
    med = tr["median_drop_acc_pp"]
    a1.axvline(med, color=GREY_D, lw=0.8, ls=(0, (4, 2)))
    a1.text(med + 1.5, 1.2, f"median\n{med:.1f} pp", fontsize=SMALL, color=GREY_D, va="center", linespacing=1.2)
    a1.set_xlabel("Accuracy lost, internal test\n$\\rightarrow$ zero-shot MLD24 (pp)")
    a1.set_xlim(0, 82); a1.set_xticks([0, 25, 50, 75]); a1.set_ylim(-0.8, len(ex2) - 0.2); a1.grid(axis="y", visible=False)
    panel(a1, "(b)", dx=-0.46)
    save(fig, "fig5_transfer.png")


def fig_replication():
    ri = pd.read_csv(DER / "_replication_internal_pairs.csv")
    re_ = pd.read_csv(DER / "_replication_external_pairs.csv")
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 6.4), gridspec_kw=dict(height_ratios=[1.35, 1]))
    (a0, a1), (a2, a3) = axes
    R = N["replication"]
    for ax, m, key, lab_pts in [
            (a0, ri, "internal_f1", {"benchmark/vgg19": (0.9772, 0.9880, "right"), "proposed/" + PROPOSED: (0.9862, 0.9808, "left"),
                                     "benchmark/convnext_small": (0.9850, 0.9765, "left"), "benchmark/resnet101": (0.9688, 0.9752, "left")}),
            (a1, re_, "external_f1", {"benchmark/vgg19": (0.31, 0.52, "left"), "benchmark/vgg16": (0.265, 0.31, "left"),
                                      "proposed/" + PROPOSED: (0.36, 0.680, "left"), "benchmark/vit_base_patch16_224": (0.66, 0.50, "center"),
                                      "benchmark/swin_small_patch4_window7_224": (0.36, 0.765, "left")})]:
        g = m.groupby("cfg")[["f1_macro_A", "f1_macro_B"]].mean()
        lo, hi = (0.966, 0.991) if key.startswith("internal") else (0.25, 0.79)
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.7, ls=(0, (4, 2)), zorder=1)
        for cfg, t in g.iterrows():
            mm = cfg.split("/")[1]
            c = colour(mm)
            ax.scatter(t.f1_macro_A, t.f1_macro_B, s=40 if mm in (PROPOSED, BACKBONE) else 24, color=c,
                       edgecolor="white", linewidth=0.5, zorder=5 if mm in (PROPOSED, BACKBONE) else 3)
        for cfg, (tx, ty, ha) in lab_pts.items():
            t = g.loc[cfg]; mm = cfg.split("/")[1]
            c = colour(mm) if colour(mm) != GREY else GREY_D
            ax.annotate(short(mm), (t.f1_macro_A, t.f1_macro_B), xytext=(tx, ty), fontsize=SMALL, color=c, ha=ha,
                        va="center", fontweight="bold" if mm == PROPOSED else "normal",
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.5, shrinkA=1, shrinkB=3))
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Macro-F1, replication A"); ax.set_ylabel("Macro-F1, replication B")
        s = R[key]
        ax.text(0.97, 0.03, f"Spearman $\\rho$ = {s['spearman_AB']:.2f}\nmax rank move: {s['max_rank_move']} places",
                transform=ax.transAxes, va="bottom", ha="right", fontsize=SMALL, color=GREY_D, linespacing=1.3)
        if key.startswith("internal"):
            ax.set_xticks([0.970, 0.975, 0.980, 0.985, 0.990]); ax.set_yticks([0.970, 0.975, 0.980, 0.985, 0.990])
    panel(a0, "(a) internal test", dx=-0.22); panel(a1, "(b) zero-shot MLD24", dx=-0.22)
    for ax, m, key, rng in [(a2, ri, "internal_f1", (-1.2, 1.2)), (a3, re_, "external_f1", (-22, 22))]:
        d = m.dF1_pp
        bins = np.linspace(rng[0], rng[1], 25)
        ax.hist(d, bins=bins, color=GREY, edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="black", lw=0.7)
        s = R[key]
        ax.text(0.97, 0.95, f"78 paired runs\nmean |$\\Delta$| = {s['mean_abs_pp']:.2f} pp\ns.d. = {s['sd_pp']:.2f} pp\n"
                f"max |$\\Delta$| = {s['max_abs_pp']:.2f} pp", transform=ax.transAxes, va="top", ha="right",
                fontsize=SMALL, color=GREY_D, linespacing=1.3)
        ax.set_xlabel("Macro-F1, A $-$ B (pp), paired runs")
        ax.set_ylabel("runs"); ax.grid(axis="x", visible=False)
    panel(a2, "(c)", dx=-0.22); panel(a3, "(d)", dx=-0.22)
    save(fig, "fig6_replication.png")


ABL_SHORT = {"wo_llrd": "layer-wise LR decay", "wo_mixup": "MixUp / CutMix", "wo_mixstyle": "MixStyle",
             "wo_label_smoothing": "label smoothing", "wo_tta": "flip TTA", "wo_mixup_cooldown": "MixUp cool-down",
             "wo_lap": "LAP (lesion-aware pooling)", "wo_ema": "EMA weights", "wo_beta_calib": "$\\beta$ calibration",
             "wo_random_erasing": "random erasing", "wo_two_phase": "two-phase schedule",
             "wo_msf": "MSF (multi-scale fusion)", "wo_hier": "auxiliary plant head",
             "wo_lap_gate": "LAP gating + multi-head", "wo_sampler": "balanced sampler"}
ARCH_V = {"wo_mixstyle", "wo_lap", "wo_msf", "wo_hier", "wo_lap_gate"}


def fig_ablation():
    rows = sorted([x for x in N["A"]["ablation"]["rows"] if x["variant"] != "wo_pretrained"], key=lambda x: x["d_ext_pp"])
    fig, ax = plt.subplots(figsize=(FULL, 5.9))
    y = np.arange(len(rows)); h = 0.36
    ax.barh(y + h / 2, [x["d_ext_pp"] for x in rows], height=h, color=ACC, label="zero-shot MLD24")
    ax.barh(y - h / 2, [x["d_int_pp"] for x in rows], height=h, color=GREY, label="internal test")
    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels([ABL_SHORT[x["variant"]] for x in rows])
    for lab, x in zip(ax.get_yticklabels(), rows):
        if x["variant"] in ARCH_V:
            lab.set_fontweight("bold")
    ax.set_xlabel("Macro-F1 lost when the component is removed (pp)")
    ax.set_xlim(-1.3, 5.6); ax.set_ylim(-0.7, len(rows) - 0.3); ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", borderaxespad=0.6)
    ax.text(0.98, 0.27, "bold = architectural component", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=SMALL, color=GREY_D)
    save(fig, "fig7_ablation.png")


def fig_perclass():
    src = "A"
    ext = pd.read_csv(DER / f"_per_class_external_{src}.csv").sort_values("f1", ascending=True)
    inn = pd.read_csv(DER / f"_per_class_internal_{src}.csv").set_index("class")
    cm = pd.read_csv(DER / f"_confusion_external_{src}.csv", index_col=0)
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(FULL, 7.6), gridspec_kw=dict(height_ratios=[1, 1.35]))
    fig.get_layout_engine().set(hspace=0.12)
    y = np.arange(len(ext))
    a0.barh(y + 0.19, [inn.loc[c, "f1"] for c in ext["class"]], height=0.36, color=GREY, edgecolor="white", linewidth=0.4)
    a0.barh(y - 0.19, ext.f1, xerr=ext.f1_sd, height=0.36, color=ACC, edgecolor="white", linewidth=0.4,
            error_kw=dict(ecolor=GREY_D, elinewidth=0.7, capsize=1.6, capthick=0.7))
    a0.set_yticks(y); a0.set_yticklabels([c.replace("mango ", "") for c in ext["class"]])
    a0.set_xlim(0, 1.42); a0.set_xticks([0, 0.5, 1.0]); a0.set_xlabel("Per-class F1"); a0.grid(axis="y", visible=False)
    top = len(ext) - 1
    a0.text(1.03, top + 0.19, "internal test", fontsize=SMALL, color=GREY_D, va="center")
    a0.text(1.03, top - 0.19, "zero-shot MLD24", fontsize=SMALL, color=ACC, va="center")
    panel(a0, "(a)", dx=-0.24)
    trues = list(cm.index); cols = list(cm.columns)
    cols = [c for c in cols if c in trues] + [c for c in cols if c not in trues]
    M = cm[cols].values
    im = a1.imshow(M, cmap="Reds", vmin=0, vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] >= 0.01:
                a1.text(j, i, f"{M[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=SMALL,
                        color="white" if M[i, j] > 0.55 else GREY_D)
    a1.set_xticks(range(len(cols)))
    a1.set_xticklabels([c.replace("mango ", "M. ").replace("jackfruit ", "J. ").replace("cashew ", "C. ")
                        .replace("tomato ", "T. ") for c in cols], rotation=40, ha="right", fontsize=SMALL)
    a1.set_yticks(range(len(trues))); a1.set_yticklabels([t.replace("mango ", "") for t in trues], fontsize=SMALL)
    for k, lab in enumerate(a1.get_yticklabels()):
        if trues[k] == "mango sooty mould":
            lab.set_color(ACC); lab.set_fontweight("bold")
    a1.set_xlabel("predicted"); a1.set_ylabel("true (MLD24)"); a1.grid(False)
    n6 = sum(c in trues for c in cols)
    a1.axvline(n6 - 0.5, color=GREY_D, lw=0.8)
    a1.text(n6 - 0.42, -0.72, "labels absent from MLD24", fontsize=SMALL, color=GREY_D, ha="left", va="center")
    cb = fig.colorbar(im, ax=a1, fraction=0.036, pad=0.025)
    cb.set_ticks([0, 0.5, 1.0]); cb.ax.tick_params(labelsize=SMALL); cb.outline.set_linewidth(0.5)
    panel(a1, "(b)", dx=-0.24, dy=1.05)
    save(fig, "fig8_perclass.png")


def fig_efficiency():
    ex = pd.read_csv(DER / "_external_A.csv")
    P = N["A"]["params"]["external_acc"]
    fig, ax = plt.subplots(figsize=(FULL, 4.6))
    x = np.log10(ex.params_M); yv = ex.accuracy_mean
    coef = np.polyfit(x, yv, 2)
    xs = np.linspace(np.log10(2), np.log10(160), 200)
    ax.plot(10 ** xs, np.polyval(coef, xs), color=GREY_D, lw=1.0, ls=(0, (4, 2)), zorder=1)
    for lo, hi, c in [(1.8, 10, "#EEEEEE"), (10, 50, "#F7F7F7"), (50, 170, "#EEEEEE")]:
        ax.axvspan(lo, hi, color=c, lw=0, zorder=0)
    sizes = 12 + 9.5 * ex.gflops
    for (_, t), s in zip(ex.iterrows(), sizes):
        c = colour(t.model)
        ax.scatter(t.params_M, t.accuracy_mean, s=s, color=c, alpha=0.92, edgecolor="white", linewidth=0.6,
                   zorder=5 if t.model in (PROPOSED, BACKBONE) else 2)
    lab = {PROPOSED: (26, 0.800, "left"), "efficientnet_b3": (3.2, 0.760, "left"),
           "legacy_seresnext50_32x4d": (60, 0.745, "left"), BACKBONE: (12, 0.56, "right"),
           "mobilenetv2_100": (2.1, 0.415, "left"), "vit_base_patch16_224": (85, 0.56, "center"),
           "vgg16": (70, 0.27, "right"), "vgg19": (70, 0.20, "right")}
    for m, (tx, ty, ha) in lab.items():
        t = ex[ex.model == m].iloc[0]
        c = colour(m) if colour(m) != GREY else GREY_D
        ax.annotate(short(m), (t.params_M, t.accuracy_mean), xytext=(tx, ty), fontsize=SMALL, ha=ha, va="center",
                    color=c, fontweight="bold" if m == PROPOSED else "normal",
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.5, shrinkA=1.5, shrinkB=4))
    ax.set_xscale("log"); ax.set_xlim(1.8, 170); ax.set_ylim(0.15, 0.84)
    ax.set_xticks([2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("Parameters (M, log scale)"); ax.set_ylabel("Accuracy, zero-shot MLD24")
    for lo, hi, txt in [(1.8, 10, "<10 M"), (10, 50, "10--50 M"), (50, 170, ">50 M")]:
        ax.text(np.sqrt(lo * hi), 0.835, txt, ha="center", va="top", fontsize=SMALL, color=GREY_D)
    ax.text(0.30, 0.03, f"quadratic in $\\log_{{10}}$ parameters: $R^2$ = {P['quadratic_R2']:.2f}\n"
            f"(linear $R^2$ = {P['linear_R2']:.2f}); vertex at {P['vertex_M']:.1f} M\n"
            f"Spearman $\\rho$ = {P['spearman_rho']:.2f}, $p$ = {P['spearman_p']:.2f}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=SMALL, color=GREY_D, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    leg = [plt.scatter([], [], s=12 + 9.5 * g_, color=GREY, edgecolor="white", linewidth=0.6, label=f"{g_:g}") for g_ in (1, 5, 15)]
    ax.legend(handles=leg, title="GFLOPs", labelspacing=1.0, borderpad=0.4, handletextpad=1.0, title_fontsize=SMALL,
              loc="lower left", bbox_to_anchor=(0.0, 0.05))
    save(fig, "fig11_efficiency.png")


def _tiles(img, thr=0.5, row_thr=0.25):
    a = np.asarray(img.convert("L"))
    mask = a < 235
    rows = mask.mean(axis=1) > row_thr
    bands = []
    start = None
    for i, v in enumerate(list(rows) + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start > 40:
                bands.append((start, i))
            start = None
    out = []
    for (y0, y1) in bands:
        colsm = mask[y0:y1].mean(axis=0) > thr
        s = None; cb = []
        for j, v in enumerate(list(colsm) + [False]):
            if v and s is None:
                s = j
            elif not v and s is not None:
                if j - s > 40:
                    cb.append((s, j))
                s = None
        out.append([(x0, y0, x1, y1) for (x0, x1) in cb])
    return out


RUNS_B = B / "runs"


def _grid_boxes(path):
    src = Image.open(path).convert("RGB")
    tiles = _tiles(src)
    boxes = [b for row in tiles for b in row]
    return src, boxes


def fig_gradcam():
    gb = RUNS_B / "proposed" / "gradcam" / "gradcam_grid.png"
    src, boxes = _grid_boxes(gb if gb.exists() else OLDFIG / "fig9_gradcam.png")
    order = [c["class"] for c in N["dataset"]["classes"]]
    assert len(boxes) == 21, len(boxes)
    ncol, nrow = 6, 4
    fig, axes = plt.subplots(nrow, ncol, figsize=(FULL, FULL / ncol * nrow * 1.22))
    fig.get_layout_engine().set(wspace=0.012, hspace=0.06, w_pad=0.004, h_pad=0.004)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, box, cname in zip(axes.ravel(), boxes, order):
        ax.imshow(np.asarray(src.crop(box)))
        crop, dis = cname.split(" ", 1)
        ax.set_title(dis, fontsize=SMALL, pad=2.0, color=CROP[crop])
        ax.axis("on"); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_color(CROP[crop]); sp.set_linewidth(1.0)
    save(fig, "fig9_gradcam.png")


def fig_tsne():
    from sklearn.manifold import TSNE
    eb = RUNS_B / "proposed" / PROPOSED / "seed1234" / "embeddings_test.npz"
    z = np.load(eb if eb.exists() else ARCH / "proposed" / PROPOSED / "seed1234" / "embeddings_test.npz")
    emb, lab = z["emb"], z["labels"]
    classes = json.loads((B / "summary" / "class_names.json").read_text())
    pts = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(FULL, 5.2))
    for k, c in enumerate(classes):
        m = lab == k
        crop = c.split(" ", 1)[0]
        ax.scatter(pts[m, 0], pts[m, 1], s=7, color=CROP[crop], alpha=0.8, linewidth=0, zorder=2)
        cx, cy = np.median(pts[m, 0]), np.median(pts[m, 1])
        ax.text(cx, cy + 2.2, c.split(" ", 1)[1], fontsize=SMALL, ha="center", va="bottom", color=CROP[crop],
                zorder=5, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=5, color=CROP[c], label=c.capitalize()) for c in CROP],
              loc="lower left", ncol=4, borderaxespad=0.3, columnspacing=1.0)
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([])
    save(fig, "fig10_tsne.png")


def fig_lap():
    gc = RUNS_B / "proposed" / "gradcam" / "gradcam_grid.png"
    la = RUNS_B / "proposed" / "gradcam" / "lap_attention_grid.png"
    if not (gc.exists() and la.exists()):
        print("  fig13_lap skipped (no replication-B grid)"); return
    order = [c["class"] for c in N["dataset"]["classes"]]
    pick = ["cashew fungal infection", "jackfruit phyllosticta", "jackfruit leaf spot", "mango anthracnose",
            "mango sooty mould", "tomato bacterial spot", "tomato spider mites"]
    sg, bg = _grid_boxes(gc); sl, bl = _grid_boxes(la)
    assert len(bg) == 21 and len(bl) == 21, (len(bg), len(bl))
    fig, axes = plt.subplots(2, len(pick), figsize=(FULL, FULL / len(pick) * 2 * 1.26))
    fig.get_layout_engine().set(wspace=0.012, hspace=0.05, w_pad=0.004, h_pad=0.004)
    for j, cname in enumerate(pick):
        k = order.index(cname)
        crop, dis = cname.split(" ", 1)
        for i, (src, boxes) in enumerate([(sg, bg), (sl, bl)]):
            ax = axes[i, j]
            ax.imshow(np.asarray(src.crop(boxes[k])))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(CROP[crop]); sp.set_linewidth(1.0)
            if i == 0:
                ax.set_title(dis, fontsize=SMALL, pad=2.0, color=CROP[crop])
            if j == 0:
                ax.set_ylabel("Grad-CAM" if i == 0 else "LAP attention", fontsize=SMALL)
    save(fig, "fig13_lap.png")


def fig_forensics():
    src = Image.open(B / "class_forensics" / "panels" / "mango_sooty_mould.png").convert("RGB")
    tiles = _tiles(src)
    assert len(tiles) == 2 and all(len(rw) == 6 for rw in tiles), [len(rw) for rw in tiles]
    fig, axes = plt.subplots(2, 6, figsize=(FULL, FULL / 6 * 2 * 1.12))
    fig.get_layout_engine().set(wspace=0.012, hspace=0.05, w_pad=0.004, h_pad=0.004)
    for i, (row, name, col) in enumerate(zip(tiles, ["internal corpus", "MLD24"], [GREY_D, ACC])):
        for j, box in enumerate(row):
            ax = axes[i, j]
            x0, y0, x1, y1 = box
            sd = min(x1 - x0, y1 - y0)
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            ax.imshow(np.asarray(src.crop((cx - sd // 2, cy - sd // 2, cx + sd // 2, cy + sd // 2))))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(col); sp.set_linewidth(1.0)
            if j == 0:
                ax.set_ylabel(name, fontsize=BODY, color=col)
    save(fig, "fig12_forensics.png")


if __name__ == "__main__":
    print("Figures (600 dpi, Times, 10/9 pt):")
    fig_dataset()
    fig_internal_ranking()
    fig_cd()
    fig_transfer()
    fig_replication()
    fig_ablation()
    fig_perclass()
    fig_efficiency()
    fig_gradcam()
    fig_tsne()
    fig_lap()
    fig_forensics()
    print("Done ->", FIG)
