"""Metrics: accuracy, F1, AUC, PR-AUC, kappa, MCC, sensitivity/specificity, G-mean, ECE, top-k."""
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_recall_fscore_support,
    cohen_kappa_score, matthews_corrcoef, roc_auc_score, log_loss,
    average_precision_score, classification_report, confusion_matrix,
)
from sklearn.preprocessing import label_binarize

# Fixed key list so that the master CSV always has the same columns
METRIC_KEYS = [
    "accuracy", "balanced_accuracy", "top5_accuracy",
    "precision_macro", "recall_macro", "f1_macro",
    "precision_weighted", "recall_weighted", "f1_weighted",
    "sensitivity_macro", "specificity_macro", "g_mean_macro",
    "cohen_kappa", "mcc",
    "auc_macro_ovr", "auc_weighted_ovr", "average_precision_macro",
    "log_loss", "ece", "n_zero_recall_classes",
]


def expected_calibration_error(y_true, y_prob, n_bins: int = 15) -> float:
    """ECE — mean gap between confidence and accuracy."""
    y_true = np.asarray(y_true)
    conf = y_prob.max(axis=1)
    pred = y_prob.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() > 0:
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def _specificity_per_class(cm, present):
    """Specificity (TN / (TN+FP)) for the given classes."""
    total = cm.sum()
    spec = []
    for i in present:
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        spec.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
    return np.array(spec, float)


def compute_metrics(y_true, y_pred, y_prob, class_names: List[str]) -> Dict:
    """All scalar metrics. The key set is ALWAYS the same (not computable -> NaN)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n_classes = len(class_names)
    labels_all = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels_all)

    # CRITICAL FOR EXTERNAL VALIDATION: macro averages are taken ONLY over the classes
    # that actually occur in y_true. Otherwise the sklearn default uses
    # union(y_true, y_pred); on the external set (only 6 of 21 classes covered) every
    # extra class the model predicts by mistake enters the denominator as 0 ->
    # f1_macro drops artificially and two models are NOT COMPARABLE (the denominator
    # depends on the model). On the internal test all 21 classes occur, so this
    # change does NOT affect the internal results.
    present = np.unique(y_true)
    cls = list(present)

    out = {k: float("nan") for k in METRIC_KEYS}

    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=cls, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=cls, average="weighted", zero_division=0)
    # per-class recall (sensitivity) and specificity — same class set
    _, r_c, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=cls, average=None, zero_division=0)
    spec_c = _specificity_per_class(cm, present)

    out.update(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        precision_macro=p_macro, recall_macro=r_macro, f1_macro=f_macro,
        precision_weighted=p_w, recall_weighted=r_w, f1_weighted=f_w,
        sensitivity_macro=float(r_c.mean()) if len(r_c) else float("nan"),
        specificity_macro=float(spec_c.mean()) if len(spec_c) else float("nan"),
        # G-mean: zero sensitivity on a SINGLE class must not collapse the metric to
        # exactly 0.0. On the external set (6 of 21 classes covered, out-of-domain) this
        # was a cliff: a model with 0 recall on one class scored 0.0 while a model with
        # a single lucky prediction scored ~0.7 — the models became incomparable.
        # Log-mean with an epsilon instead.
        g_mean_macro=float(np.exp(np.mean(np.log(np.clip(r_c, 1e-12, 1.0)))))
        if len(r_c) else float("nan"),
        n_zero_recall_classes=int((r_c == 0).sum()) if len(r_c) else 0,
        cohen_kappa=cohen_kappa_score(y_true, y_pred),
        mcc=matthews_corrcoef(y_true, y_pred),
    )

    if y_prob is not None:
        yb = label_binarize(y_true, classes=labels_all)
        # top-5 accuracy (if there are more than 5 classes)
        if n_classes > 5:
            top5 = np.argsort(-y_prob, axis=1)[:, :5]
            out["top5_accuracy"] = float(np.mean([t in row for t, row in zip(y_true, top5)]))
        try:
            if len(present) == n_classes:
                out["auc_macro_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                                     average="macro", labels=labels_all)
                out["auc_weighted_ovr"] = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                                        average="weighted", labels=labels_all)
            else:
                # ON THE EXTERNAL SET: the probabilities used to be re-normalised over the
                # covered classes. Since AUC is purely rank-based and every row was
                # divided by a DIFFERENT constant, this changed the metric materially;
                # moreover the divisor depended on how much mass the model placed on the
                # 15 uncovered classes, i.e. it varied by model. (average_precision was
                # already computed on the un-normalised score -> the two metrics lived on
                # two different score definitions.)
                # The right way: compute the one-vs-rest AUC of every covered class on
                # the RAW column and take the macro / weighted average.
                aucs, support = [], []
                for j in present:
                    aucs.append(roc_auc_score(yb[:, j], y_prob[:, j]))
                    support.append(int(yb[:, j].sum()))
                out["auc_macro_ovr"] = float(np.mean(aucs))
                out["auc_weighted_ovr"] = float(np.average(aucs, weights=support))
        except Exception as e:
            print(f"    (warning: AUC could not be computed: {type(e).__name__}: {e})")
        try:
            out["average_precision_macro"] = average_precision_score(
                yb[:, present], y_prob[:, present], average="macro")
        except Exception as e:
            print(f"    (warning: average_precision could not be computed: {type(e).__name__}: {e})")
        try:
            out["log_loss"] = log_loss(y_true, y_prob, labels=labels_all)
        except Exception as e:
            print(f"    (warning: log_loss could not be computed: {type(e).__name__}: {e})")
        out["ece"] = expected_calibration_error(y_true, y_prob)

    return out


def per_class_dataframe(y_true, y_pred, class_names: List[str]):
    """Per-class precision / recall (= sensitivity) / specificity / F1 / support."""
    import pandas as pd
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    rep = classification_report(y_true, y_pred, labels=list(range(n)),
                                target_names=class_names, output_dict=True,
                                zero_division=0)
    spec = _specificity_per_class(cm, list(range(n)))
    rows = []
    for i, c in enumerate(class_names):
        rows.append({"class": c,
                     "precision": rep[c]["precision"],
                     "recall_sensitivity": rep[c]["recall"],
                     "specificity": float(spec[i]),
                     "f1-score": rep[c]["f1-score"],
                     "support": rep[c]["support"]})
    return pd.DataFrame(rows)


def confusion(y_true, y_pred, n_classes: int):
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
