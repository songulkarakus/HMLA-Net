"""leafdx — Leaf disease classification research pipeline (HMLA-Net benchmark).

Modules:
  config   : central settings, model registry (25 baselines + the proposed model)
  data     : indexing, hash-based de-duplication, stratified splits, plant groups,
             external validation
  models   : the proposed HMLA-Net (CAFormer-S18 + MSF + LAP + hierarchical head
             + MixStyle) and the baseline factory
  engine   : training/evaluation loops (AMP, MixUp, EMA, two-phase schedule, RESUME)
  metrics  : accuracy, F1, AUC, kappa, MCC, ECE, per-class report
  stats    : Friedman+Nemenyi, Wilcoxon, McNemar, bootstrap CI, Cohen's d
  viz      : confusion matrix, curves, ROC/PR, calibration, t-SNE, CD diagram
"""
__version__ = "1.0.0"
