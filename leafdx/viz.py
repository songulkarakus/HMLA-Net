"""Visualisation: confusion matrix, curves, ROC/PR, calibration, t-SNE, CD diagram."""
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize


def plot_confusion(cm, class_names, path, normalize=True, title=""):
    cm = np.asarray(cm, float)
    if normalize:
        cm = cm / np.clip(cm.sum(1, keepdims=True), 1e-9, None)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.5),
                                    max(7, len(class_names) * 0.5)))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or "Confusion matrix")
    if len(class_names) <= 25:
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                v = cm[i, j]
                if v > 0.01:
                    ax.text(j, i, f"{v:.2f}" if normalize else int(v),
                            ha="center", va="center", fontsize=5,
                            color="white" if v > 0.5 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_history(history: List[Dict], path):
    ep = [h["epoch"] for h in history]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(ep, [h["train_loss"] for h in history], label="train")
    a1.plot(ep, [h["val_loss"] for h in history], label="val")
    a1.set_title("Loss"); a1.set_xlabel("epoch"); a1.legend()
    a2.plot(ep, [h["train_acc"] for h in history], label="train")
    a2.plot(ep, [h["val_acc"] for h in history], label="val")
    a2.set_title("Accuracy"); a2.set_xlabel("epoch"); a2.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_roc(y_true, y_prob, class_names, path):
    classes = list(range(len(class_names)))
    yb = label_binarize(y_true, classes=classes)
    present = [i for i in classes if yb[:, i].sum() > 0]
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in present:
        fpr, tpr, _ = roc_curve(yb[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, lw=1, label=f"{class_names[i]} ({auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC (one-vs-rest)")
    ax.legend(fontsize=5, ncol=2, loc="lower right")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_pr(y_true, y_prob, class_names, path):
    classes = list(range(len(class_names)))
    yb = label_binarize(y_true, classes=classes)
    present = [i for i in classes if yb[:, i].sum() > 0]
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in present:
        pr, rc, _ = precision_recall_curve(yb[:, i], y_prob[:, i])
        ax.plot(rc, pr, lw=1, label=class_names[i])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (one-vs-rest)")
    ax.legend(fontsize=5, ncol=2, loc="lower left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_reliability(y_true, y_prob, path, n_bins=15):
    """Calibration (reliability) diagram + ECE."""
    conf = y_prob.max(1)
    pred = y_prob.argmax(1)
    correct = (pred == np.asarray(y_true)).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            xs.append(conf[m].mean()); ys.append(correct[m].mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect")
    ax.plot(xs, ys, "o-", label="model")
    ax.set_xlabel("Mean confidence"); ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram"); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_tsne(features: np.ndarray, labels, class_names, path, seed=0):
    """2-D t-SNE embedding of the learned features."""
    from sklearn.manifold import TSNE
    n = len(features)
    perp = max(5, min(30, n // 4))
    emb = TSNE(n_components=2, init="pca", perplexity=perp,
               random_state=seed).fit_transform(features)
    fig, ax = plt.subplots(figsize=(9, 8))
    labels = np.asarray(labels)
    for i, c in enumerate(class_names):
        m = labels == i
        if m.sum() > 0:
            ax.scatter(emb[m, 0], emb[m, 1], s=8, label=c)
    ax.set_title("t-SNE (learned features)")
    ax.legend(fontsize=5, ncol=2, markerscale=1.5, loc="best")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_cd_diagram(avg_ranks: Dict[str, float], cd: float, path, title=""):
    """Critical difference (Nemenyi) diagram."""
    items = sorted(avg_ranks.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    ranks = [v for _, v in items]
    k = len(names)
    lo, hi = min(ranks), max(ranks)
    lo, hi = math.floor(lo), math.ceil(hi)

    fig, ax = plt.subplots(figsize=(10, max(3, k * 0.35)))
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, k + 2)
    ax.axis("off")
    # top axis (rank line)
    y0 = k + 1
    ax.plot([lo, hi], [y0, y0], "k-", lw=1)
    for t in range(lo, hi + 1):
        ax.plot([t, t], [y0, y0 + 0.15], "k-", lw=1)
        ax.text(t, y0 + 0.3, str(t), ha="center", fontsize=8)
    # CD bar
    ax.plot([lo, lo + cd], [y0 + 0.8, y0 + 0.8], "k-", lw=2)
    ax.text(lo + cd / 2, y0 + 0.95, f"CD = {cd:.2f}", ha="center", fontsize=8)
    # one marker per model
    for i, (nm, rk) in enumerate(zip(names, ranks)):
        y = k - i
        ax.plot([rk, rk], [y0, y], "k-", lw=0.7)
        ax.plot([rk, lo], [y, y], "k-", lw=0.7)
        ax.text(lo - 0.05, y, f"{nm} ({rk:.2f})", ha="right", va="center", fontsize=7)
    ax.set_title(title or "Critical difference diagram (Nemenyi, α=0.05)")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
