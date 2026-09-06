"""
Models: the proposed HMLA-Net and the baseline factory.

HMLA-Net — Hierarchical Multi-scale Lesion-Attention Network
============================================================
    input [B,3,H,W]
      └─ CAFormer-S18 backbone (features_only, out_indices=(1,2,3))
           stages_0 ─► MixStyle          [B, 64, H/4,  W/4 ]
           stages_1 ─► MixStyle          [B,128, H/8,  W/8 ]  ─┐
           stages_2                      [B,320, H/16, W/16]  ─┤ 3 scales
           stages_3                      [B,512, H/32, W/32]  ─┘
      └─ MSF   multi-scale fusion -> stride-16 grid     [B,256,H/16,W/16]  (Grad-CAM target)
      └─ LAP   lesion-aware attentive pooling (MIL)     [B,512]            (t-SNE embedding)
      └─ LayerNorm -> Dropout
      └─ fc_disease [B,21]   +   fc_plant [B,4] (auxiliary)

At inference the two heads are fused; the output dimension STAYS 21:
    logit[c] = logit_disease[c] + beta * log_softmax(logit_plant)[plant_of[c]]
"""
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .config import RunConfig, PROPOSED_BACKBONE


# --------------------------------------------------------------------------- #
# MixStyle — Zhou et al., ICLR 2021
# Mixes the feature statistics (mu, sigma) within a batch: the model stops relying on
# style cues such as the "studio background" and depends on lesion content instead ->
# robustness to domain shift (clean in-domain data -> field-condition external data).
# --------------------------------------------------------------------------- #
class MixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p, self.alpha, self.eps = p, alpha, eps

    def extra_repr(self):
        return f"p={self.p}, alpha={self.alpha}"

    def forward(self, x):
        if not self.training or x.size(0) < 2 or torch.rand(()).item() > self.p:
            return x
        mu = x.mean(dim=(2, 3), keepdim=True)
        sig = (x.var(dim=(2, 3), keepdim=True) + self.eps).sqrt()
        x_norm = (x - mu) / sig
        lam = torch.distributions.Beta(self.alpha, self.alpha) \
                   .sample((x.size(0), 1, 1, 1)).to(x.device, x.dtype)
        perm = torch.randperm(x.size(0), device=x.device)
        mu_mix = mu * lam + mu[perm] * (1 - lam)
        sig_mix = sig * lam + sig[perm] * (1 - lam)
        return x_norm * sig_mix + mu_mix


class LayerNorm2d(nn.LayerNorm):
    """Channel-axis LayerNorm on [B,C,H,W] tensors."""
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


# --------------------------------------------------------------------------- #
# MSF — Multi-Scale Fusion
# Lesion scale varies enormously across classes (phyllosticta = millimetre-sized spot,
# die back = the whole leaf). A single stride-32 map cannot carry this range.
# We project the 3 stages to a common channel width and fuse them on the stride-16 grid.
# Side benefit: Grad-CAM at 14x14 instead of 7x7 -> much sharper XAI figures.
# --------------------------------------------------------------------------- #
class MSF(nn.Module):
    """Multi-scale fusion.

    Two design decisions were measured and corrected:

    1) THE FINE SCALE IS DOWNSAMPLED WITH A LEARNED STRIDE. The previous version reduced
       the stride-8 (28x28) map to 14x14 with `adaptive_avg_pool2d`. Average pooling
       averaged a millimetre-sized lesion spot with its 3 healthy neighbours and thereby
       destroyed the reason MSF exists. Replaced by a learned depthwise stride-2
       convolution: the network decides which fine detail to carry.
    2) THE SCALES ARE SUMMED WITH WEIGHTS. A plain sum trusts the three scales equally,
       yet the dominant scale changes with the class (spot lesion vs whole leaf).
       Learnable scalar weights normalised with a softmax (BiFPN style) leave this to
       the data.
    """

    def __init__(self, in_chs: Sequence[int], dim: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, dim, 1, bias=False), LayerNorm2d(dim))
            for c in in_chs)
        # The learned reduction is built ONLY for scales finer than the reference
        # (coarse scales are bilinearly upsampled). An unused module = dead weights and
        # an "unused parameter" error under DDP; hence only the required ones are created.
        self.ref_idx = len(in_chs) // 2
        self.down = nn.ModuleList(
            nn.Conv2d(dim, dim, 3, stride=2, padding=1, groups=dim, bias=False)
            if i < self.ref_idx else nn.Identity()
            for i in range(len(in_chs)))
        self.scale_w = nn.Parameter(torch.zeros(len(in_chs)))   # softmax -> starts equal
        self.smooth = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),   # depthwise
            nn.Conv2d(dim, dim, 1, bias=False),                          # pointwise
            LayerNorm2d(dim), nn.GELU())

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # reference grid = the middle scale (stride-16). feature_info.reduction() is
        # reported incorrectly for caformer in timm 1.0.8; hence the scale is taken
        # from the tensor shape.
        ref = feats[self.ref_idx].shape[-2:]
        w = self.scale_w.softmax(0)
        fused = None
        for i, (f, lat) in enumerate(zip(feats, self.lateral)):
            y = lat(f)
            while y.shape[-1] > ref[-1] and i < self.ref_idx:   # fine -> learned stride-2
                y = self.down[i](y)
            if y.shape[-2:] != ref:                 # coarse -> bilinear upsample
                y = F.interpolate(y, size=ref, mode="bilinear", align_corners=False)
            y = y * w[i]
            fused = y if fused is None else fused + y
        return self.smooth(fused)


# --------------------------------------------------------------------------- #
# LAP — Lesion-Aware Attentive Pooling (multiple-instance learning)
# ~90% of a leaf is healthy tissue; GAP dilutes the signal of a small lesion.
# We treat the leaf as a "bag of patches" and pool with learned weights (+ a max branch).
# --------------------------------------------------------------------------- #
class LAP(nn.Module):
    """Lesion-aware attentive pooling (MIL).

    Two upgrades (both measured by the `wo_lap_gate` ablation):

    - GATED ATTENTION (Ilse et al., ICML 2018 — the gated variant of ABMIL):
      `a = w^T (tanh(Vh) * sigmoid(Uh))`. A plain `tanh` scorer saturates in one
      direction; the sigmoid gate models separately how much each patch may speak,
      which improves discrimination on fine-grained lesions.
    - MULTI-HEAD pooling: one attention map means one region of interest. A leaf may
      carry several lesions of different appearance (necrosis + chlorotic halo). K heads
      each select a different patch; the result is projected back to `dim`, so the
      output dimension STAYS 2*dim (the t-SNE/Grad-CAM code works unchanged).
    """

    def __init__(self, dim: int = 256, hidden: int = 128,
                 heads: int = 4, gated: bool = True):
        super().__init__()
        self.heads = max(1, int(heads))
        self.gated = bool(gated)
        self.v = nn.Conv2d(dim, hidden, 1)
        self.u = nn.Conv2d(dim, hidden, 1) if self.gated else None
        self.w = nn.Conv2d(hidden, self.heads, 1)
        # project the concatenated heads back to dim -> num_features is unchanged
        self.proj = nn.Identity() if self.heads == 1 else \
            nn.Linear(self.heads * dim, dim, bias=False)
        self.log_tau = nn.Parameter(torch.zeros(1))     # learned sharpness
        self.last_attn = None                            # XAI: gradient-free attention map

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        b, c, h, w = f.shape
        e = torch.tanh(self.v(f))
        if self.gated:
            e = e * torch.sigmoid(self.u(f))
        logits = self.w(e).flatten(2) / self.log_tau.exp().clamp(0.2, 5.0)
        w_att = logits.softmax(-1)                       # [B,K,HW]
        # XAI: average over the heads -> a single map (the gradcam.py contract is kept)
        self.last_attn = w_att.detach().mean(1, keepdim=True).view(b, 1, h, w)
        flat = f.flatten(2)                              # [B,C,HW]
        z_att = torch.einsum("bcn,bkn->bkc", flat, w_att).flatten(1)  # [B,K*C]
        z_att = self.proj(z_att)                         # [B,C]
        z_max = flat.amax(-1)                            # [B,C]
        return torch.cat([z_att, z_max], dim=1)          # [B,2C]


# --------------------------------------------------------------------------- #
# Proposed model
# --------------------------------------------------------------------------- #
class ProposedModel(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int,
                 plant_of: Optional[Sequence[int]] = None,
                 pretrained: bool = True, feat_dim: int = 256, dropout: float = 0.3,
                 use_msf: bool = True, use_lap: bool = True, use_hier: bool = True,
                 use_mixstyle: bool = True, mixstyle_p: float = 0.5,
                 mixstyle_alpha: float = 0.1, hier_beta: float = 1.0,
                 lap_heads: int = 4, lap_gated: bool = True,
                 out_indices: Tuple[int, ...] = (1, 2, 3)):
        super().__init__()
        self.backbone, chs = self._make_backbone(backbone_name, pretrained, out_indices)
        self.use_msf = use_msf
        self.mixstyle_stages: List[str] = []
        if use_mixstyle:
            self.mixstyle_stages = _inject_mixstyle(self.backbone, mixstyle_p, mixstyle_alpha)

        # --- backbone -> spatial feature map ---
        if use_msf:
            self.msf = MSF(chs, feat_dim)
        else:                                            # ablation: single scale
            self.msf = nn.Sequential(
                nn.Conv2d(chs[-1], feat_dim, 1, bias=False), LayerNorm2d(feat_dim), nn.GELU())

        # --- pooling ---
        self.use_lap = use_lap
        if use_lap:
            self.pool = LAP(feat_dim, heads=lap_heads, gated=lap_gated)
            self.num_features = feat_dim * 2
        else:                                            # ablation: classic GAP
            self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
            self.num_features = feat_dim

        # --- heads ---
        self.norm = nn.LayerNorm(self.num_features)
        self.drop = nn.Dropout(dropout)
        self.fc_disease = nn.Linear(self.num_features, num_classes)

        # --- hierarchical plant head (auxiliary) ---
        valid = plant_of is not None and len(plant_of) == num_classes \
            and len(set(plant_of)) >= 2
        self.use_hier = bool(use_hier and valid)
        # hier_beta is NOT LEARNABLE: since the fusion is applied only at inference, an
        # nn.Parameter would never receive a gradient (it would be a dead weight).
        # Instead it is searched on the VALIDATION set after training
        # (engine.calibrate_hier_beta). Because the grid contains 0.0 the fusion can
        # never hurt: in the worst case beta=0 is selected and the model falls back to
        # the pure disease head.
        self.register_buffer("hier_beta_t", torch.tensor(float(hier_beta)))
        if self.use_hier:
            self.num_plants = int(max(plant_of)) + 1
            self.fc_plant = nn.Linear(self.num_features, self.num_plants)
            self.register_buffer("plant_of", torch.as_tensor(list(plant_of)).long())
        else:
            self.num_plants = 0
            self.fc_plant = None
            self.register_buffer("plant_of", torch.zeros(num_classes).long())

    # ------------------------------------------------------------------ #
    @property
    def hier_beta(self) -> float:
        return float(self.hier_beta_t.item())

    @torch.no_grad()
    def set_hier_beta(self, beta: float):
        """Stores the beta selected on the validation set (it also goes into the checkpoint)."""
        self.hier_beta_t.fill_(float(beta))

    @staticmethod
    def _make_backbone(name, pretrained, out_indices):
        """features_only backbone; falls back to the last 3 stages if out_indices is unsupported."""
        try:
            bb = timm.create_model(name, pretrained=pretrained,
                                   features_only=True, out_indices=out_indices)
        except (RuntimeError, TypeError, IndexError, KeyError):
            bb = timm.create_model(name, pretrained=pretrained, features_only=True)
            n = len(bb.feature_info.channels())
            bb = timm.create_model(name, pretrained=pretrained, features_only=True,
                                   out_indices=tuple(range(max(0, n - 3), n)))
        return bb, bb.feature_info.channels()

    def forward_features(self, x):
        feats = self.backbone(x)
        return self.msf(feats) if self.use_msf else self.msf(feats[-1])

    def forward_logits(self, x):
        """Raw head outputs WITHOUT the fusion: (logit_disease, logit_plant, z).

        The beta calibration uses this: the whole beta grid can be re-evaluated from a
        single forward pass (no need to run the network once per beta).
        """
        z = self.drop(self.norm(self.pool(self.forward_features(x))))
        logit_d = self.fc_disease(z)
        logit_p = self.fc_plant(z) if self.use_hier else None
        return logit_d, logit_p, z

    def fuse_logits(self, logit_d, logit_p, beta=None):
        """logit[c] = logit_d[c] + beta * log P(plant_of[c]).  beta=None -> the calibrated value."""
        if not self.use_hier or logit_p is None:
            return logit_d
        b = self.hier_beta_t if beta is None else beta
        return logit_d + b * logit_p.log_softmax(-1)[:, self.plant_of]

    def forward(self, x, return_aux: bool = False):
        logit_d, logit_p, z = self.forward_logits(x)
        out = logit_d
        if self.use_hier and not self.training:
            # hierarchical fusion at inference; the output dimension STAYS num_classes
            out = self.fuse_logits(logit_d, logit_p)
        return (out, logit_p, z) if return_aux else out

    def get_cam_layer(self):
        """Grad-CAM target: the fusion output (stride-16, 14x14 at 224)."""
        return self.msf

    @torch.no_grad()
    def attention_map(self):
        """LAP's own attention map — gradient-free, a second XAI figure for free."""
        return getattr(self.pool, "last_attn", None)


def _inject_mixstyle(backbone: nn.Module, p: float, alpha: float,
                     n_stages: int = 2) -> List[str]:
    """Adds MixStyle to the output of the first n stages (style lives in shallow layers).

    WRAPS with nn.Sequential instead of a hook: after the EMA deepcopy the copy owns
    its own MixStyle instance (no closure leak). The module order is preserved.
    """
    skip = ("stem", "patch_embed", "conv1", "bn1", "act1", "maxpool")
    names = [n for n, _ in backbone.named_children()
             if not any(k in n for k in skip)]
    used = []
    for n in names[:n_stages]:
        setattr(backbone, n, nn.Sequential(getattr(backbone, n), MixStyle(p, alpha)))
        used.append(n)
    return used


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_model(cfg: RunConfig, num_classes: int) -> nn.Module:
    if cfg.is_proposed:
        return ProposedModel(
            cfg.model_name, num_classes,
            plant_of=getattr(cfg, "plant_of", None),
            pretrained=cfg.use_pretrained,
            feat_dim=getattr(cfg, "feat_dim", 256),
            dropout=getattr(cfg, "dropout", 0.3),
            use_msf=getattr(cfg, "use_msf", True),
            use_lap=getattr(cfg, "use_lap", True),
            use_hier=getattr(cfg, "use_hier", True),
            use_mixstyle=getattr(cfg, "use_mixstyle", True),
            mixstyle_p=getattr(cfg, "mixstyle_p", 0.5),
            mixstyle_alpha=getattr(cfg, "mixstyle_alpha", 0.1),
            hier_beta=getattr(cfg, "hier_beta", 1.0),
            lap_heads=getattr(cfg, "lap_heads", 4),
            lap_gated=getattr(cfg, "lap_gated", True))
    # baseline: a standard timm classifier
    return timm.create_model(cfg.model_name, pretrained=cfg.use_pretrained,
                             num_classes=num_classes)


def get_head_params(model: nn.Module):
    """All non-backbone parameters, for phase 1 (frozen backbone)."""
    if isinstance(model, ProposedModel):
        params = []
        for mod in (model.msf, model.pool, model.norm, model.fc_disease):
            params += list(mod.parameters())
        if model.fc_plant is not None:
            params += list(model.fc_plant.parameters())
        return params
    return list(model.get_classifier().parameters())


# --------------------------------------------------------------------------- #
# Model complexity: parameter count, FLOPs, model size
# --------------------------------------------------------------------------- #
def complexity(model: nn.Module, img_size: int) -> dict:
    n_params = sum(p.numel() for p in model.parameters())
    size_mb = n_params * 4 / (1024 ** 2)  # float32 assumption
    flops = None
    # the dummy input must live on the SAME device as the model; otherwise FLOPs silently stay NaN
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    try:
        from fvcore.nn import FlopCountAnalysis
        was_training = model.training
        model.eval()
        dummy = torch.randn(1, 3, img_size, img_size, device=device)
        flops = float(FlopCountAnalysis(model, dummy).total()) / 1e9  # GFLOPs
        if was_training:
            model.train()
    except Exception:
        try:
            from thop import profile
            dummy = torch.randn(1, 3, img_size, img_size, device=device)
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
            flops = float(macs) * 2 / 1e9
        except Exception:
            flops = float("nan")
    return {"params_M": n_params / 1e6, "size_MB": size_mb, "gflops": flops}
