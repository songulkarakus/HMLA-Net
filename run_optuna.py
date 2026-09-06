"""
Hyper-parameter optimisation with Optuna — ALL models (24 baselines + the proposed model).

One 'study' per model: finetune_lr, weight_decay, label_smoothing (+ dropout and other
model-specific coefficients for the proposed model) are searched. TPE sampler +
MedianPruner (early stopping). Every study is stored in SQLite -> it RESUMES WHERE IT
STOPPED. The best parameters are written to results/optuna/best_params.json and fed to
the benchmark:

    python run_benchmark.py --tuned_params results/optuna/best_params.json

Examples:
    python run_optuna.py --n_trials 25 --hpo_epochs 8 --limit_per_class 200
    python run_optuna.py --quick --models resnet18        # fast smoke test
"""
import argparse
import json
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from leafdx.config import (BASELINE_MODELS, make_proposed_config,
                           make_baseline_config, PROPOSED_NAME)
from leafdx.data import build_index, stratified_split, make_loaders, plant_index
from leafdx.engine import (evaluate, selection_score, _set_phase, _build_optimizer)
from leafdx.models import build_model, ProposedModel
from leafdx.utils import seed_everything, fmt_duration
from run_benchmark import safe_name, subsample

optuna.logging.set_verbosity(optuna.logging.WARNING)


def hpo_train_eval(cfg, class_names, splits, device, trial, hpo_epochs):
    """Short training with the SAME SHAPE AS THE BENCHMARK; val macro-F1 every epoch (pruning).

    This used to be a flat single-phase loop: one AdamW over ALL parameters with
    `finetune_lr`. Harmless for the baselines (the "head" is a single Linear). For the
    proposed model the head = MSF + LAP + norm + fc_disease + fc_plant, i.e. millions of
    parameters initialised FROM SCRATCH; training them with the backbone LR pushed TPE
    towards "the smallest lr that does not damage the pretrained backbone" — the
    opposite of what the new modules need. The same two-phase recipe as
    engine.train_run is used now (frozen phase with head_lr, then warm-up + cosine).
    """
    seed_everything(cfg.seed)
    use_amp = cfg.use_amp and device == "cuda"
    model = build_model(cfg, len(class_names)).to(device)
    train_loader, val_loader, _ = make_loaders(splits, cfg, device)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = GradScaler("cuda", enabled=use_amp)
    select_metric = getattr(cfg, "select_metric", "f1_macro")

    hier = isinstance(model, ProposedModel) and model.use_hier
    plant_mat = (F.one_hot(model.plant_of, model.num_plants).float().to(device)
                 if hier else None)

    freeze = min(int(cfg.freeze_epochs), max(0, hpo_epochs - 1))
    best_score, opt, sched, cur_phase = 0.0, None, None, None
    for epoch in range(hpo_epochs):
        phase = 1 if epoch < freeze else 2
        if phase != cur_phase:
            _set_phase(model, phase)
            opt = _build_optimizer(model, cfg, phase)
            sched = (torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, hpo_epochs - freeze)) if phase == 2 else None)
            cur_phase = phase
        model.train()
        for imgs, tgts in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            tgts = tgts.to(device, non_blocking=True)
            with autocast("cuda", enabled=use_amp):
                if hier:
                    out, logit_p, _ = model(imgs, return_aux=True)
                    loss = crit(out, tgts) + cfg.lambda_plant * (
                        -(plant_mat[tgts] * logit_p.log_softmax(-1)).sum(-1).mean())
                else:
                    loss = crit(model(imgs), tgts)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        if sched is not None:
            sched.step()
        yt, yp, _, _ = evaluate(model, val_loader, device, use_amp)
        score = selection_score(yt, yp, select_metric)
        best_score = max(best_score, score)
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best_score


def make_objective(base_cfg, class_names, splits, device, hpo_epochs):
    def objective(trial):
        params = {
            "finetune_lr": trial.suggest_float("finetune_lr", 1e-5, 1e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True),
            "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
        }
        if base_cfg.is_proposed:
            # head_lr is THE MOST CRITICAL PARAMETER FOR THE PROPOSED MODEL: MSF/LAP/heads
            # are trained from scratch and phase 1 is governed by this LR. It was absent
            # from the old search space (unnoticed because it is nearly irrelevant for
            # the baselines). The other model-specific coefficients are searched too.
            params["head_lr"] = trial.suggest_float("head_lr", 1e-4, 1e-2, log=True)
            params["dropout"] = trial.suggest_float("dropout", 0.1, 0.5)
            params["lambda_plant"] = trial.suggest_float("lambda_plant", 0.0, 0.6)
            params["mixstyle_p"] = trial.suggest_float("mixstyle_p", 0.0, 0.75)
            params["mixup_prob"] = trial.suggest_float("mixup_prob", 0.3, 1.0)
            params["llrd_gamma"] = trial.suggest_float("llrd_gamma", 0.55, 1.0)
        cfg = replace(base_cfg, **params)
        return hpo_train_eval(cfg, class_names, splits, device, trial, hpo_epochs)
    return objective


def save_plots(study, out_dir, name):
    try:
        from optuna.visualization.matplotlib import (plot_optimization_history,
                                                      plot_param_importances)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ax = plot_optimization_history(study)
        ax.figure.tight_layout(); ax.figure.savefig(out_dir / f"{name}_history.png", dpi=130)
        plt.close(ax.figure)
        if len([t for t in study.trials if t.state.name == "COMPLETE"]) >= 2:
            ax = plot_param_importances(study)
            ax.figure.tight_layout()
            ax.figure.savefig(out_dir / f"{name}_importance.png", dpi=130)
            plt.close(ax.figure)
    except Exception as e:
        print(f"    (warning: optuna plot for {name} skipped: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="multiple leaf dataset")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--n_trials", type=int, default=25)
    ap.add_argument("--proposed_trials", type=int, default=None,
                    help="separate trial budget for the proposed model (default: n_trials*2, "
                         "because its search space is twice as large as the baselines')")
    ap.add_argument("--hpo_epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hpo_split_seed", type=int, default=7777,
                    help="seed of the HPO split — must NOT coincide with the benchmark seeds")
    ap.add_argument("--models", default="all")
    ap.add_argument("--no_proposed", action="store_true")
    ap.add_argument("--limit_per_class", type=int, default=None,
                    help="sub-sample to speed up the HPO (suggested: 150-300)")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    opt_dir = out_dir / "optuna"
    opt_dir.mkdir(parents=True, exist_ok=True)
    t_pipeline = time.time()
    started = datetime.now()
    print(f"Device: {device}")
    print(f"START: {started:%Y-%m-%d %H:%M:%S}")

    n_trials, hpo_epochs, limit = args.n_trials, args.hpo_epochs, args.limit_per_class
    if args.quick:
        args.img_size, n_trials, hpo_epochs = 96, 3, 1
        limit = limit or 30
        print("QUICK mode: img=96, 3 trials, 1 epoch")

    import timm
    requested = BASELINE_MODELS if args.models == "all" else \
        [m.strip() for m in args.models.split(",") if m.strip()]
    available = set(timm.list_models())
    # 'arch.pretrained_tag' names are reduced to the root name for the check
    model_list = [m for m in requested
                  if m in available or m.split(".")[0] in available]

    samples, class_names, _ = build_index(args.data_dir, dedup=True)
    work = subsample(samples, limit, args.seed) if limit else samples
    # THE HPO SPLIT MUST BE DISJOINT FROM THE BENCHMARK SEEDS. It used to be split with
    # args.seed=42; without --limit_per_class this was IDENTICAL to the split of
    # benchmark seed 42 (the hyper-parameters were selected on the val set later used
    # for early stopping), and with --limit_per_class a different split emerged and
    # images that fall into the benchmark's TEST set were used in the HPO (leakage).
    splits = stratified_split(work, args.hpo_split_seed, val_frac=0.15, test_frac=0.15)
    print(f"Classes: {len(class_names)} | HPO train={len(splits['train'])} "
          f"val={len(splits['val'])} | split seed={args.hpo_split_seed} "
          f"(disjoint from the benchmark seeds)")

    # The HPO tunes the model AS IT RUNS IN THE BENCHMARK. The proposed model's
    # use_mixup/use_ema/use_hier/use_mixstyle flags used to be switched off during the
    # HPO: the baselines already run with these flags off, so they were tuned exactly
    # under benchmark conditions, whereas the proposed model was tuned in a
    # configuration that would NEVER BE RUN (e.g. a label_smoothing selected with MixUp
    # off was stacked on top of MixUp's soft targets in the benchmark).
    # use_tta is excluded: it only affects inference, quadruples the evaluation cost
    # and does not change the selected hyper-parameters.
    shared = dict(data_dir=args.data_dir, img_size=args.img_size,
                  batch_size=args.batch_size, num_workers=args.num_workers,
                  seed=args.seed, use_tta=False)
    if args.no_pretrained:
        shared["use_pretrained"] = False

    specs = []
    if not args.no_proposed:
        pcfg = make_proposed_config(**shared)
        # FILL plant_of: runner.run_one did this, run_optuna did not. Because of the
        # `valid = plant_of is not None ...` check in models.py the hierarchical head
        # silently stayed off during the HPO.
        po, plant_names = plant_index(class_names)
        if po is not None:
            pcfg.plant_of = po
            print(f"Proposed-model plant groups: {plant_names}")
        else:
            print("WARNING: plant groups could not be derived -> hierarchical head off in the HPO.")
        specs.append((PROPOSED_NAME, pcfg))
    for m in model_list:
        specs.append((m, make_baseline_config(m, **shared)))

    best_params, summary_rows = {}, []
    for name, base in specs:
        safe = safe_name(name)
        storage = f"sqlite:///{(opt_dir / (safe + '.db')).as_posix()}"
        study = optuna.create_study(
            study_name=safe, storage=storage, load_if_exists=True,
            direction="maximize", sampler=TPESampler(seed=args.seed),
            pruner=MedianPruner(n_warmup_steps=max(1, hpo_epochs // 3)))
        done = len([t for t in study.trials
                    if t.state.name in ("COMPLETE", "PRUNED")])
        # the proposed model's search space is ~2x larger -> the same budget gives a worse optimum
        budget = (args.proposed_trials or n_trials * 2) if base.is_proposed else n_trials
        remaining = max(0, budget - done)
        print(f"\n=== {name} === ({done}/{budget} trials complete, {remaining} remaining)")
        t_model = time.time()
        if remaining > 0:
            try:
                study.optimize(
                    make_objective(base, class_names, splits, device, hpo_epochs),
                    n_trials=remaining, catch=(RuntimeError,))
            except KeyboardInterrupt:
                print("Interrupted. Re-run the same command to resume.")
                raise
            except Exception as e:
                print(f"[ERROR] {name}: {e}")
                traceback.print_exc()

        model_sec = time.time() - t_model
        n_complete = len([t for t in study.trials if t.state.name == "COMPLETE"])
        n_fail = len([t for t in study.trials if t.state.name == "FAIL"])
        if n_fail:
            # study.optimize(catch=(RuntimeError,)) also swallows CUDA OOM. Previously,
            # even if ALL trials of the proposed model crashed, no warning was printed,
            # the model never appeared in best_params.json and it ran in the benchmark
            # with DEFAULT hyper-parameters — while the 25 baselines were tuned.
            print(f"    WARNING: {n_fail} trial(s) ended with an ERROR (usually CUDA OOM). "
                  f"Try lowering --batch_size.")
        if n_complete == 0:
            print(f"    WARNING: NO completed trial for '{name}' -> it will not be written to "
                  f"best_params.json and will run with DEFAULT hyper-parameters in the benchmark.")
        if n_complete > 0:
            best_params[name] = {"params": study.best_params,
                                 "value": float(study.best_value),
                                 "n_trials": len(study.trials),
                                 "study_sec": round(model_sec, 1)}
            print(f"    best val-F1={study.best_value:.4f} | "
                  f"time {fmt_duration(model_sec)} | {study.best_params}")
            save_plots(study, opt_dir, safe)
            summary_rows.append({"model": name, "best_val_f1": study.best_value,
                                 "n_trials": len(study.trials),
                                 "study_sec": round(model_sec, 1), **study.best_params})
            # save after every model (partial results are usable too)
            with open(opt_dir / "best_params.json", "w", encoding="utf-8") as f:
                json.dump(best_params, f, indent=2)

    if summary_rows:
        import pandas as pd
        pd.DataFrame(summary_rows).to_csv(opt_dir / "optuna_summary.csv", index=False)
    ended = datetime.now()
    elapsed = time.time() - t_pipeline
    with open(opt_dir / "timing_optuna.json", "w", encoding="utf-8") as f:
        json.dump({"script": "run_optuna",
                   "started_at": started.isoformat(timespec="seconds"),
                   "ended_at": ended.isoformat(timespec="seconds"),
                   "total_sec": round(elapsed, 1),
                   "total_human": fmt_duration(elapsed)}, f, indent=2)
    print(f"\nHPO finished. Best parameters: {opt_dir / 'best_params.json'}")
    print(f"START: {started:%Y-%m-%d %H:%M:%S}  END: {ended:%Y-%m-%d %H:%M:%S}  "
          f"TOTAL: {fmt_duration(elapsed)}")
    print(f"Usage:  python run_benchmark.py --tuned_params "
          f"{opt_dir / 'best_params.json'}")


if __name__ == "__main__":
    main()
