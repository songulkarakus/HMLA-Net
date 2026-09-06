"""Central configuration and model registry."""
from dataclasses import dataclass, asdict
from typing import List, Optional

# --------------------------------------------------------------------------- #
# 25 BASELINES (timm names). CNN + Transformer + hybrid families for a fair comparison.
# All are ImageNet-pretrained and run at 224x224.
# --------------------------------------------------------------------------- #
BASELINE_MODELS: List[str] = [
    # --- Classic CNNs ---
    "resnet50",
    "resnet101",
    "densenet121",
    "densenet201",
    "vgg16",
    "vgg19",
    "inception_v3",
    "legacy_xception",
    "resnext50_32x4d",
    "legacy_seresnext50_32x4d",
    # --- Efficient / mobile CNNs ---
    "mobilenetv3_large_100",
    "mobilenetv2_100",
    "efficientnet_b0",
    "efficientnet_b3",
    "mnasnet_100",
    "regnety_016",
    "ghostnet_100",
    # --- Modern CNNs ---
    "convnext_tiny",
    "convnext_small",
    # --- Vision Transformer families ---
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "swin_tiny_patch4_window7_224",
    "swin_small_patch4_window7_224",
    "deit3_small_patch16_224",
    # --- The BARE backbone of the proposed model (fairness control) ---
    # This entry is critical: it answers the question "does the gain come from the
    # components or from a stronger backbone?" directly. Bare backbone vs
    # backbone + HMLA components = our contribution.
    "caformer_s18.sail_in22k_ft_in1k",
]  # 25 models (any that cannot be downloaded is skipped automatically; target >= 20)

# --------------------------------------------------------------------------- #
# PROPOSED MODEL — HMLA-Net
#   Hierarchical Multi-scale Lesion-Attention Network
#   Backbone: CAFormer-S18 (MetaFormer; first two stages convolution, last two attention)
#   Components: MSF (multi-scale fusion) + LAP (lesion-aware attentive pooling)
#               + HCL (hierarchical plant-disease head) + MixStyle (domain randomisation)
# --------------------------------------------------------------------------- #
PROPOSED_BACKBONE = "caformer_s18.sail_in22k_ft_in1k"
PROPOSED_NAME = "Proposed_HMLA_CAFormerS18"


@dataclass
class RunConfig:
    """All settings of a single training run."""
    # identity
    model_name: str = PROPOSED_BACKBONE
    is_proposed: bool = False
    experiment: str = "benchmark"           # benchmark | ablation | proposed
    seed: int = 42

    # data
    data_dir: str = "multiple leaf dataset"
    img_size: int = 224
    val_frac: float = 0.15
    test_frac: float = 0.15
    dedup: bool = True

    # training
    batch_size: int = 32
    num_workers: int = 4
    freeze_epochs: int = 2                  # phase 1: head only
    finetune_epochs: int = 30               # phase 2: whole network
    head_lr: float = 1e-3
    finetune_lr: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    dropout: float = 0.3                    # proposed-model head (tunable with Optuna)
    patience: int = 8                       # early stopping
    warmup_epochs: int = 2                  # warm-up before the cosine schedule

    # --- model selection (checkpoint + early-stopping criterion) ---
    # The data set is imbalanced (209 <-> 500) and the paper's primary metric is
    # macro-F1; selecting the best epoch by accuracy while reporting macro-F1 was
    # inconsistent. The SAME criterion is applied to ALL models.
    select_metric: str = "f1_macro"         # "f1_macro" | "val_acc" | "balanced_accuracy"

    # --- ARCHITECTURAL components of the proposed model (the ablation toggles them) ---
    use_msf: bool = False                   # multi-scale fusion (stride 8/16/32 -> 16)
    use_lap: bool = False                   # lesion-aware attentive pooling (MIL)
    use_hier: bool = False                  # hierarchical plant-disease head
    use_mixstyle: bool = False              # MixStyle domain randomisation
    feat_dim: int = 256                     # number of MSF fusion channels
    lambda_plant: float = 0.3               # weight of the auxiliary plant loss
    hier_beta: float = 1.0                  # weight of the plant log-probability at inference
                                            # (re-selected on val if calibrate_beta=True)
    calibrate_beta: bool = False            # search hier_beta on VAL after training
    mixstyle_p: float = 0.5                 # MixStyle application probability
    mixstyle_alpha: float = 0.1             # MixStyle Beta(alpha, alpha)
    lap_heads: int = 4                      # LAP multi-head attention (1 = single head)
    lap_gated: bool = True                  # LAP gated attention (Ilse et al. 2018, ABMIL)
    plant_of: Optional[List[int]] = None    # class -> plant mapping (filled by the runner)

    # --- training-recipe components (baselines receive a subset of them) ---
    use_mixup: bool = False                 # MixUp + CutMix
    mixup_prob: float = 0.8                 # per-batch application probability
    mixup_off_epochs: int = 8               # MixUp OFF for the LAST n epochs ("cool-down")
    use_ema: bool = False                   # exponential moving average of the weights
    ema_decay: float = 0.0                  # 0 -> derived automatically from the step count
    use_tta: bool = False                   # test-time augmentation
    use_llrd: bool = False                  # layer-wise LR decay (transformer fine-tuning)
    llrd_gamma: float = 0.75                # LR multiplier between stages (along backbone depth)
    use_sampler: bool = True                # WeightedRandomSampler (class imbalance)
    use_random_erasing: bool = True
    use_pretrained: bool = True
    two_phase: bool = True                  # False -> single-phase fine-tune (freeze_epochs=0)

    # system
    use_amp: bool = True
    deterministic: bool = False             # True -> deterministic cuDNN (exact repeat)

    def to_dict(self):
        return asdict(self)


def _apply_overrides(cfg: RunConfig, overrides: dict) -> RunConfig:
    """Applies overrides. AN UNKNOWN KEY IS AN ERROR.

    Previously this was a silent setattr: a mistyped key in best_params.json created a
    dead attribute, build_model fell back to the getattr default and the tuned
    hyper-parameter was silently ignored.
    """
    unknown = [k for k in overrides if k not in RunConfig.__dataclass_fields__]
    if unknown:
        raise ValueError(
            f"Unknown RunConfig key(s): {unknown}. "
            f"Valid keys: {sorted(RunConfig.__dataclass_fields__)}")
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def make_proposed_config(**overrides) -> RunConfig:
    """Proposed model (HMLA-Net) — all components enabled."""
    cfg = RunConfig(
        model_name=PROPOSED_BACKBONE,
        is_proposed=True,
        experiment="proposed",
        use_msf=True,
        use_lap=True,
        use_hier=True,
        use_mixstyle=True,
        use_mixup=True,
        use_ema=True,
        use_tta=True,
        use_llrd=True,
        calibrate_beta=True,
        use_sampler=True,
        use_random_erasing=True,
        use_pretrained=True,
        two_phase=True,
    )
    return _apply_overrides(cfg, overrides)


def make_baseline_config(model_name: str, **overrides) -> RunConfig:
    """Baseline: the shared training recipe, proposed-model-specific components OFF."""
    cfg = RunConfig(
        model_name=model_name,
        is_proposed=False,
        experiment="benchmark",
        use_msf=False,
        use_lap=False,
        use_hier=False,
        use_mixstyle=False,
        use_mixup=False,
        use_ema=False,
        use_tta=False,
        use_sampler=True,
        use_random_erasing=True,
        use_pretrained=True,
        two_phase=True,
    )
    return _apply_overrides(cfg, overrides)


# --------------------------------------------------------------------------- #
# ABLATION: remove one component at a time from the proposed model and measure it.
# The first 4 rows are our ARCHITECTURAL contribution, the rest is the training recipe.
# --------------------------------------------------------------------------- #
def ablation_variants():
    return {
        "full":               dict(),                              # full proposed model
        # --- architectural components ---
        "wo_msf":             dict(use_msf=False),                 # single scale (stride-32 only)
        "wo_lap":             dict(use_lap=False),                 # GAP instead of LAP
        "wo_hier":            dict(use_hier=False),                # no plant head
        "wo_mixstyle":        dict(use_mixstyle=False),            # no domain randomisation
        # --- architectural sub-components ---
        "wo_lap_gate":        dict(lap_gated=False, lap_heads=1),  # plain single-head attention
        # --- training recipe ---
        "wo_mixup":           dict(use_mixup=False),
        "wo_ema":             dict(use_ema=False),
        "wo_tta":             dict(use_tta=False),
        "wo_llrd":            dict(use_llrd=False),                # flat LR (no layer-wise decay)
        "wo_beta_calib":      dict(calibrate_beta=False),          # hier_beta fixed at 1.0
        "wo_mixup_cooldown":  dict(mixup_off_epochs=0),            # MixUp on until the end
        "wo_sampler":         dict(use_sampler=False),
        "wo_random_erasing":  dict(use_random_erasing=False),
        "wo_label_smoothing": dict(label_smoothing=0.0),
        "wo_two_phase":       dict(two_phase=False, freeze_epochs=0),
        "wo_pretrained":      dict(use_pretrained=False),
    }
