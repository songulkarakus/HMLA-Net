"""Training/evaluation engine: AMP, MixUp, EMA, two-phase schedule, RESUME."""
import copy
import math
import os
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from .models import build_model, get_head_params, ProposedModel
from .utils import seed_everything


# --------------------------------------------------------------------------- #
# EMA — exponential moving average of the weights (stable, better generalisation)
# --------------------------------------------------------------------------- #
class EMA:
    """Exponential moving average of the weights — SCALED BY THE STEP COUNT.

    THE BUG IN THE PREVIOUS VERSION: the decay was a fixed 0.9998. That value suits
    millions of ImageNet steps; on this data set (~5,000 images, batch 32 -> ~158
    steps/epoch, 32 epochs = ~5,056 steps) the effective window (1/(1-d) = 5,000 steps)
    was as long as THE WHOLE TRAINING. Result: 36% of the shadow weights were still the
    RANDOMLY INITIALISED head of epoch 0 (msf/pool/norm/fc_* are trained from scratch)
    and `best.pth` was saved from that shadow. Because EMA is enabled only for the
    proposed model, the bug penalised the proposed model alone.

    The fix has two parts:
      1) the decay is derived from the step count (default horizon = 3 epochs), and
      2) a warm-up correction (`(1+t)/(10+t)`) glues the shadow to the model during the
         first steps, so the random initialisation never carries lasting weight.
    """

    def __init__(self, model: nn.Module, decay: float = 0.0,
                 steps_per_epoch: int = 0, horizon_epochs: float = 3.0):
        if not decay:                       # 0/None -> derive from the training length
            window = max(1.0, horizon_epochs * max(1, steps_per_epoch))
            decay = min(0.9999, max(0.99, 1.0 - 1.0 / window))
        self.decay = float(decay)
        self.step = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def effective_decay(self) -> float:
        return min(self.decay, (1.0 + self.step) / (10.0 + self.step))

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step += 1
        d = self.effective_decay()
        for s, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                s.copy_(m)

    def state_dict(self):
        return {"shadow": self.shadow.state_dict(), "step": self.step,
                "decay": self.decay}

    def load_state_dict(self, sd):
        # stay compatible with old (plain state_dict) checkpoints
        if "shadow" in sd:
            self.shadow.load_state_dict(sd["shadow"])
            self.step = int(sd.get("step", 0))
            self.decay = float(sd.get("decay", self.decay))
        else:
            self.shadow.load_state_dict(sd)


# --------------------------------------------------------------------------- #
# RNG states (exact reproducibility on resume)
# --------------------------------------------------------------------------- #
def _rng_state(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None,
    }


def _set_rng_state(st, device):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if device == "cuda" and st.get("cuda") is not None:
        torch.cuda.set_rng_state_all(st["cuda"])


# --------------------------------------------------------------------------- #
# Phase machine
# --------------------------------------------------------------------------- #
def _set_phase(model, phase):
    if phase == 1:
        head = set(id(p) for p in get_head_params(model))
        for p in model.parameters():
            p.requires_grad = id(p) in head
    else:
        for p in model.parameters():
            p.requires_grad = True


def _no_decay(name: str, param: torch.Tensor) -> bool:
    """Norm weights, biases and 1-D learned scalars receive NO weight decay.

    Applying decay to them in AdamW is a standard mistake in transformer fine-tuning:
    it pulls the LayerNorm scale towards 0 and corrupts the representation. (Learned
    scalars such as LAP.log_tau and MSF.scale_w belong to this group as well.)
    """
    return param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower()


def _backbone_depth_groups(model):
    """Splits the backbone into name prefixes ordered from shallow to deep (for LLRD).

    Returned list: [shallowest, ..., deepest]. Non-backbone (trained-from-scratch)
    modules are assigned the deepest-layer LR by the caller.
    """
    bb = getattr(model, "backbone", None)
    if bb is None:
        return []
    names = [n for n, _ in bb.named_children()]
    # timm features_only: stem, stages_0..N  -> already ordered from shallow to deep
    stem = [n for n in names if any(k in n for k in
                                    ("stem", "patch_embed", "conv1", "bn1", "act1", "maxpool"))]
    stages = [n for n in names if n not in stem]
    return [[s] for s in stem] + [[s] for s in stages]


def _build_optimizer(model, cfg, phase):
    wd = cfg.weight_decay
    if phase == 1:
        named = [(n, p) for n, p in model.named_parameters()
                 if id(p) in {id(q) for q in get_head_params(model)}]
        groups = [
            {"params": [p for n, p in named if not _no_decay(n, p)], "weight_decay": wd},
            {"params": [p for n, p in named if _no_decay(n, p)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW([g for g in groups if g["params"]],
                                 lr=cfg.head_lr)

    lr = cfg.finetune_lr
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not getattr(cfg, "use_llrd", False):
        groups = [
            {"params": [p for n, p in named if not _no_decay(n, p)], "weight_decay": wd},
            {"params": [p for n, p in named if _no_decay(n, p)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW([g for g in groups if g["params"]], lr=lr)

    # --- layer-wise LR decay (LLRD) ---
    # The shallow layers of a pretrained backbone carry generic texture/edge filters and
    # are damaged by a large LR; deep layers are task-specific. The gamma^(depth
    # difference) multiplier is standard in transformer fine-tuning and gives a clear
    # gain on small data sets.
    gamma = float(getattr(cfg, "llrd_gamma", 0.75))
    depth = _backbone_depth_groups(model)
    n_layers = len(depth)
    groups, claimed = [], set()
    for i, prefixes in enumerate(depth):
        scale = gamma ** (n_layers - 1 - i)          # deepest -> 1.0
        sel = [(n, p) for n, p in named
               if any(n.startswith(f"backbone.{pre}") for pre in prefixes)]
        claimed.update(id(p) for _, p in sel)
        for decay_on in (True, False):
            ps = [p for n, p in sel if (not _no_decay(n, p)) == decay_on]
            if ps:
                groups.append({"params": ps, "lr": lr * scale,
                               "weight_decay": wd if decay_on else 0.0})
    # non-backbone (trained-from-scratch) modules: full LR
    rest = [(n, p) for n, p in named if id(p) not in claimed]
    for decay_on in (True, False):
        ps = [p for n, p in rest if (not _no_decay(n, p)) == decay_on]
        if ps:
            groups.append({"params": ps, "lr": lr,
                           "weight_decay": wd if decay_on else 0.0})
    return torch.optim.AdamW(groups, lr=lr)


def _build_scheduler(optimizer, cfg):
    """Warm-up + cosine (phase 2 only)."""
    total = cfg.finetune_epochs
    warm = min(cfg.warmup_epochs, max(total - 1, 1))

    def fn(e):
        if e < warm:
            return (e + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (e - warm) / max(total - warm, 1)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


# --------------------------------------------------------------------------- #
# Evaluation (optional TTA). Runs on the given model.
# --------------------------------------------------------------------------- #
TTA_VIEWS = 4       # identity + horizontal + vertical + 180 degrees (measure_latency uses this)


@torch.no_grad()
def evaluate(model, loader, device, use_amp=True, tta=False, criterion=None):
    model.eval()
    all_prob, all_true = [], []
    total_loss, n = 0.0, 0
    for imgs, tgts in tqdm(loader, desc="eval", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        tgts = tgts.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp):
            out = model(imgs)
            if criterion is not None:
                total_loss += criterion(out, tgts).item() * imgs.size(0)
            prob = F.softmax(out.float(), dim=1)
            if tta:
                # A leaf image has no canonical orientation (training also uses
                # horizontal + vertical flips). The full dihedral flip group: the +180
                # degree view comes for free and spreads the average over 4 views
                # instead of 3.
                for aug in (torch.flip(imgs, [3]),
                            torch.flip(imgs, [2]),
                            torch.flip(imgs, [2, 3])):
                    prob = prob + F.softmax(model(aug).float(), dim=1)
                prob = prob / float(TTA_VIEWS)
        all_prob.append(prob.cpu())
        all_true.append(tgts.cpu())
        n += imgs.size(0)
    y_prob = torch.cat(all_prob).numpy()
    y_true = torch.cat(all_true).numpy()
    y_pred = y_prob.argmax(1)
    loss = total_loss / n if criterion is not None else float("nan")
    return y_true, y_pred, y_prob, loss


# --------------------------------------------------------------------------- #
# Model-selection criterion — checkpointing and early stopping decide with this.
# Selecting by accuracy while reporting macro-F1 on 21 imbalanced classes was inconsistent.
# --------------------------------------------------------------------------- #
def selection_score(y_true, y_pred, metric: str = "f1_macro") -> float:
    from sklearn.metrics import f1_score, balanced_accuracy_score
    if metric in ("val_acc", "accuracy"):
        return float((np.asarray(y_true) == np.asarray(y_pred)).mean())
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


# --------------------------------------------------------------------------- #
# Calibration of the hierarchical fusion coefficient (beta) on the VALIDATION set
#
# During training the loss is computed on `logit_disease` only, whereas inference used
# `logit_disease + beta * log P(plant)`; beta had never been tuned and was fixed at 1.0.
# That is a train/inference mismatch: a confident but wrong plant head suppresses ALL
# classes of that plant at once.
#
# Because the grid contains 0.0 the calibration can never hurt — in the worst case
# beta=0 is selected and the model falls back to the pure disease head.
# --------------------------------------------------------------------------- #
BETA_GRID = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0)


@torch.no_grad()
def calibrate_hier_beta(model, loader, device, use_amp=True,
                        metric: str = "f1_macro", grid=BETA_GRID):
    """Collects the raw logits in a single forward pass, then scans the grid on the CPU."""
    model.eval()
    ld_all, lp_all, y_all = [], [], []
    for imgs, tgts in tqdm(loader, desc="beta-cal", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        with autocast("cuda", enabled=use_amp):
            logit_d, logit_p, _ = model.forward_logits(imgs)
        ld_all.append(logit_d.float().cpu())
        lp_all.append(logit_p.float().cpu())
        y_all.append(tgts.cpu())
    ld = torch.cat(ld_all)
    lp = torch.cat(lp_all)
    y = torch.cat(y_all).numpy()
    plant_of = model.plant_of.cpu()
    log_p = lp.log_softmax(-1)[:, plant_of]

    scores = {}
    for b in grid:
        pred = (ld + float(b) * log_p).argmax(1).numpy()
        scores[float(b)] = selection_score(y, pred, metric)
    best_beta = max(scores, key=scores.get)
    return best_beta, scores


# --------------------------------------------------------------------------- #
# Main training loop (resume-capable)
# --------------------------------------------------------------------------- #
def train_run(cfg, num_classes, loaders, device, run_dir: Path, log=print):
    """Trains one model; writes the best weights to best.pth. Resumes where it left off."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path = run_dir / "last.pth"
    best_path = run_dir / "best.pth"
    train_loader, val_loader, _ = loaders

    # exact per-run reproducibility (head init, augmentation, sampler, mixup)
    seed_everything(cfg.seed, getattr(cfg, "deterministic", False))
    model = build_model(cfg, num_classes).to(device)
    scaler = GradScaler("cuda", enabled=cfg.use_amp and device == "cuda")

    # MixUp + loss function
    mixup_fn, soft_criterion = None, None
    if cfg.use_mixup:
        from timm.data import Mixup
        from timm.loss import SoftTargetCrossEntropy
        mixup_fn = Mixup(mixup_alpha=0.2, cutmix_alpha=1.0,
                         prob=float(getattr(cfg, "mixup_prob", 0.8)),
                         switch_prob=0.5, mode="batch",
                         label_smoothing=cfg.label_smoothing, num_classes=num_classes)
        soft_criterion = SoftTargetCrossEntropy()
    hard_criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    select_metric = getattr(cfg, "select_metric", "f1_macro")

    # --- hierarchical (plant) auxiliary head ---
    # plant_mat: [num_classes, num_plants] one-hot. When MixUp produces soft targets
    # the plant target is softened automatically via y @ plant_mat -> the two
    # components do not conflict.
    hier = isinstance(model, ProposedModel) and model.use_hier
    plant_mat = None
    if hier:
        plant_mat = F.one_hot(model.plant_of, model.num_plants).float().to(device)
        log(f"  hierarchical head enabled: {model.num_plants} plants, "
            f"lambda={cfg.lambda_plant}")

    ema = EMA(model, decay=getattr(cfg, "ema_decay", 0.0),
              steps_per_epoch=len(train_loader)) if cfg.use_ema else None
    if ema is not None:
        log(f"  EMA decay={ema.decay:.5f} (horizon ~3 epochs, {len(train_loader)} steps/epoch)")
    # `two_phase` used to be read NOWHERE; the `wo_two_phase` ablation only worked by
    # accident because it also passed freeze_epochs=0. Now it is a real switch.
    freeze_epochs = cfg.freeze_epochs if getattr(cfg, "two_phase", True) else 0
    total_epochs = freeze_epochs + cfg.finetune_epochs
    # the cool-down is at most half of the training: in short runs (e.g. --epochs 5) a
    # fixed 8-epoch cool-down would switch MixUp off entirely and equal `wo_mixup`.
    mixup_off = min(int(getattr(cfg, "mixup_off_epochs", 0)), total_epochs // 2)
    mixup_stop = total_epochs - mixup_off
    history, best = [], {"score": -1.0, "val_acc": -1.0, "epoch": -1, "source": "raw"}
    start_epoch, resumed_phase, ck_opt, ck_sched, no_improve = 0, None, None, None, 0

    # --- RESUME ---
    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        if ema and ck.get("ema_state"):
            ema.load_state_dict(ck["ema_state"])
        scaler.load_state_dict(ck["scaler_state"])
        history = ck["history"]
        best = ck["best"]
        best.setdefault("score", best.get("val_acc", -1.0))   # old-checkpoint compatibility
        best.setdefault("source", "raw")
        start_epoch = ck["next_epoch"]
        resumed_phase = ck["phase"]
        no_improve = ck.get("no_improve", 0)     # the early-stopping counter continues too
        ck_opt, ck_sched = ck.get("opt_state"), ck.get("sched_state")
        _set_rng_state(ck["rng"], device)
        log(f"  [resume] epoch {start_epoch}/{total_epochs} "
            f"(best {select_metric}={best['score']:.4f})")
        if start_epoch >= total_epochs:
            _maybe_calibrate_beta(cfg, num_classes, best_path, val_loader,
                                  device, select_metric, log)
            return history, best_path

    opt, sched, cur_phase = None, None, None    # no_improve is set above (resume-safe)
    for epoch in range(start_epoch, total_epochs):
        phase = 1 if epoch < freeze_epochs else 2
        if phase != cur_phase:
            _set_phase(model, phase)
            opt = _build_optimizer(model, cfg, phase)
            sched = _build_scheduler(opt, cfg) if phase == 2 else None
            if epoch == start_epoch and resumed_phase == phase and ck_opt is not None:
                opt.load_state_dict(ck_opt)
                if sched is not None and ck_sched is not None:
                    sched.load_state_dict(ck_sched)
            cur_phase = phase

        # ---- train ----
        t_ep = time.time()
        model.train()
        correct, seen, tr_loss = 0, 0, 0.0
        # MixUp "cool-down": mixing is switched off for the last n epochs. Final epochs
        # spent on mixed images prevent the model from settling on the real (unmixed)
        # test distribution; a measurable loss on small, fine-grained data sets.
        mixup_on = mixup_fn is not None and epoch < mixup_stop
        if mixup_fn is not None and epoch == mixup_stop and mixup_off > 0:
            log(f"  MixUp switched off (cool-down for the last {mixup_off} epochs)")
        for imgs, tgts in tqdm(train_loader, desc=f"ep{epoch} p{phase}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)
            orig = tgts
            if mixup_on and imgs.size(0) % 2 == 0:
                imgs, tgts = mixup_fn(imgs, tgts)
                crit = soft_criterion
            else:
                crit = hard_criterion
            with autocast("cuda", enabled=cfg.use_amp and device == "cuda"):
                if hier:
                    out, logit_p, _ = model(imgs, return_aux=True)
                    loss = crit(out, tgts)
                    p_tgt = (tgts @ plant_mat) if tgts.dim() == 2 else plant_mat[tgts]
                    loss = loss + cfg.lambda_plant * (
                        -(p_tgt * logit_p.log_softmax(-1)).sum(-1).mean())
                else:
                    out = model(imgs)
                    loss = crit(out, tgts)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if ema is not None:
                ema.update(model)
            tr_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == orig).sum().item()
            seen += imgs.size(0)
        if sched is not None:
            sched.step()

        # ---- val ----
        # With EMA enabled the RAW model is ALWAYS evaluated as well and the epoch's
        # checkpoint is saved from the BETTER of the two. The previous version blindly
        # used the EMA shadow; when the shadow lagged behind (see the EMA class) there
        # was no way to notice. Extra cost: one val pass per epoch (~3% time), in return
        # the "EMA corrupts the model" risk disappears entirely.
        cand = [("raw", model)] + ([("ema", ema.shadow)] if ema is not None else [])
        results = {}
        for src, mdl in cand:
            yt, yp, _, vl = evaluate(mdl, val_loader, device,
                                     cfg.use_amp and device == "cuda",
                                     tta=False, criterion=hard_criterion)
            results[src] = (selection_score(yt, yp, select_metric),
                            float((yt == yp).mean()), vl)
        src_best = max(results, key=lambda s: results[s][0])
        score, va_acc, va_loss = results[src_best]
        eval_model = dict(cand)[src_best]

        lr_now = max(g["lr"] for g in opt.param_groups)
        epoch_sec = time.time() - t_ep
        row = dict(epoch=epoch, phase=phase, train_loss=tr_loss / seen,
                   train_acc=correct / seen, val_loss=va_loss, val_acc=va_acc,
                   val_score=score, val_src=src_best, mixup=int(mixup_on),
                   lr=lr_now, epoch_sec=round(epoch_sec, 2))
        for src, (sc, ac, _) in results.items():          # keep both candidates
            row[f"val_score_{src}"] = sc
            row[f"val_acc_{src}"] = ac
        history.append(row)
        log(f"  [{epoch:03d}|p{phase}] tr_loss={tr_loss/seen:.4f} "
            f"tr_acc={correct/seen:.4f} val_loss={va_loss:.4f} "
            f"val_acc={va_acc:.4f} {select_metric}={score:.4f} "
            f"({src_best}) lr={lr_now:.2e}")

        # ---- best ----
        improved = score > best["score"]
        if improved:
            best.update(score=score, val_acc=va_acc, epoch=epoch, source=src_best)
            torch.save({"model_state": eval_model.state_dict(),
                        "cfg": cfg.to_dict(), "score": score, "val_acc": va_acc,
                        "select_metric": select_metric, "source": src_best,
                        "epoch": epoch}, best_path)
            no_improve = 0
        else:
            no_improve += 1

        # ---- checkpoint (for RESUME, every epoch) ----
        tmp = str(last_path) + ".tmp"
        torch.save({"model_state": model.state_dict(),
                    "ema_state": ema.state_dict() if ema else None,
                    "opt_state": opt.state_dict(),
                    "sched_state": sched.state_dict() if sched else None,
                    "scaler_state": scaler.state_dict(),
                    "history": history, "best": best, "no_improve": no_improve,
                    "next_epoch": epoch + 1, "phase": phase,
                    "rng": _rng_state(device), "cfg": cfg.to_dict()}, tmp)
        os.replace(tmp, last_path)

        # ---- early stopping (phase 2) ----
        if phase == 2 and cfg.patience and no_improve >= cfg.patience:
            log(f"  early stopping (epoch {epoch}, no improvement for {cfg.patience} epochs)")
            break

    _maybe_calibrate_beta(cfg, num_classes, best_path, val_loader,
                          device, select_metric, log)
    return history, best_path


def _maybe_calibrate_beta(cfg, num_classes, best_path, val_loader, device,
                          select_metric, log=print):
    """After training, selects hier_beta on VAL and writes it into best.pth.

    Never touches the test set; because the grid contains 0.0 the result can never be
    worse than the uncalibrated state. The selected beta goes into the `hier_beta_t`
    buffer inside the checkpoint, so predict.py/gradcam.py/run_external.py pick it up
    automatically.
    """
    if not (getattr(cfg, "calibrate_beta", False) and cfg.is_proposed
            and getattr(cfg, "use_hier", False) and Path(best_path).exists()):
        return
    ck = torch.load(best_path, map_location=device, weights_only=False)
    if ck.get("hier_beta") is not None:            # resume: already calibrated
        log(f"  hier_beta already calibrated: {ck['hier_beta']}")
        return
    model = build_model(cfg, num_classes).to(device)
    model.load_state_dict(ck["model_state"])
    if not (isinstance(model, ProposedModel) and model.use_hier):
        return
    beta, scores = calibrate_hier_beta(model, val_loader, device,
                                       cfg.use_amp and device == "cuda", select_metric)
    model.set_hier_beta(beta)
    ck["model_state"] = model.state_dict()
    ck["hier_beta"] = float(beta)
    ck["hier_beta_scores"] = {str(k): v for k, v in scores.items()}
    torch.save(ck, best_path)
    log(f"  hier_beta calibrated: beta={beta} "
        f"(val {select_metric}: beta=0 -> {scores[0.0]:.4f}, "
        f"beta={beta} -> {scores[beta]:.4f})")
