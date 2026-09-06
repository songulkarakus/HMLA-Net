"""
Benchmark: 25 baselines + the PROPOSED model (HMLA-Net), multiple seeds, external validation.
All outputs are saved per (model, seed). If the run stops it resumes where it left off.

Examples:
    python run_benchmark.py --seeds 42,1234,2024 --batch_size 32
    python run_benchmark.py --quick               # fast smoke test (even on CPU)
    python run_benchmark.py --external_dir external/mango --external_map external/mango_map.json
"""
import argparse
import json
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import timm

from leafdx.config import (BASELINE_MODELS, make_proposed_config,
                           make_baseline_config, PROPOSED_NAME)
from leafdx.data import (build_index, stratified_split, stratified_kfold_splits,
                         build_external_index)
from leafdx.runner import run_one, _append_master
from leafdx.utils import capture_environment, fmt_duration


def safe_name(s: str) -> str:
    return s.replace("/", "_").replace(".", "-")


def subsample(samples, per_class, seed):
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for s in samples:
        by[s[1]].append(s)
    out = []
    for lab, items in by.items():
        idx = rng.permutation(len(items))[:per_class]
        out += [items[i] for i in idx]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="multiple leaf dataset")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=None, help="finetune_epochs override")
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stopping patience (default 8; 0 = off)")
    ap.add_argument("--select_metric", default=None,
                    help="checkpoint/early-stopping criterion: f1_macro (default) | "
                         "val_acc | balanced_accuracy. Applied to ALL models alike.")
    ap.add_argument("--seeds", default="42,1234,2024")
    ap.add_argument("--models", default="all", help="'all' or comma-separated timm names")
    ap.add_argument("--max_models", type=int, default=None)
    ap.add_argument("--no_proposed", action="store_true")
    ap.add_argument("--external_dir", default=None)
    ap.add_argument("--external_map", default=None)
    ap.add_argument("--external_max_per_class", type=int, default=None,
                    help="max images per class in the external validation (balance/speed)")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test")
    ap.add_argument("--limit_per_class", type=int, default=None)
    ap.add_argument("--no_pretrained", action="store_true",
                    help="do not download ImageNet weights (offline/smoke test)")
    ap.add_argument("--kfold", type=int, default=None,
                    help="K-fold cross-validation (replaces --seeds when given)")
    ap.add_argument("--deterministic", action="store_true",
                    help="deterministic cuDNN mode (exact reproducibility, slightly slower)")
    ap.add_argument("--tuned_params", default=None,
                    help="Optuna best_params.json (per-model hyper-parameters)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master_csv = out_dir / "master_results.csv"
    fail_csv = out_dir / "failures.csv"
    with open(out_dir / "environment.json", "w", encoding="utf-8") as f:
        json.dump(capture_environment(), f, indent=2)
    t_pipeline = time.time()
    started = datetime.now()
    print(f"Device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))
    print(f"START: {started:%Y-%m-%d %H:%M:%S}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    finetune_epochs = args.epochs
    freeze_epochs = None
    limit = args.limit_per_class
    if args.quick:
        args.img_size = 96
        finetune_epochs = 1
        freeze_epochs = 0
        limit = limit or 40
        seeds = seeds[:1]
        print(f"QUICK mode: img=96, finetune_epochs=1, limit_per_class={limit}, single seed")

    # --- model list + timm validation ---
    requested = BASELINE_MODELS if args.models == "all" else \
        [m.strip() for m in args.models.split(",") if m.strip()]
    available = set(timm.list_models())
    model_list, skipped = [], []
    for m in requested:
        # timm.list_models() returns UNTAGGED architecture names; reduce names of the
        # form 'arch.pretrained_tag' to their root name for the check (otherwise skipped).
        base = m.split(".")[0]
        (model_list if (m in available or base in available) else skipped).append(m)
    if skipped:
        print(f"WARNING: {len(skipped)} model(s) not found in timm were skipped: {skipped}")
    if args.max_models:
        model_list = model_list[:args.max_models]
    print(f"Number of baseline models: {len(model_list)} | seeds={seeds}")

    # --- data index ---
    samples, class_names, n_dup = build_index(args.data_dir, dedup=True)
    print(f"Classes: {len(class_names)} | images: {len(samples)} | duplicates dropped: {n_dup}")
    with open(out_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    # --- external validation (optional) ---
    ext_samples = None
    if args.external_dir and args.external_map:
        ext_samples, info = build_external_index(
            args.external_dir, class_names, args.external_map,
            max_per_class=args.external_max_per_class)
        print(f"External validation: {info['n_images']} images, "
              f"{len(info['covered_classes'])} classes covered: {info['covered_classes']}")
        with open(out_dir / "external_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    shared = dict(data_dir=args.data_dir, img_size=args.img_size,
                  batch_size=args.batch_size, num_workers=args.num_workers)
    if finetune_epochs is not None:
        shared["finetune_epochs"] = finetune_epochs
    if freeze_epochs is not None:
        shared["freeze_epochs"] = freeze_epochs
    if args.no_pretrained:
        shared["use_pretrained"] = False
    if args.deterministic:
        shared["deterministic"] = True
    if args.patience is not None:
        shared["patience"] = args.patience
    if args.select_metric is not None:
        shared["select_metric"] = args.select_metric

    # Optuna-tuned hyper-parameters (per model)
    tuned = {}
    if args.tuned_params:
        p = Path(args.tuned_params)
        if p.exists():
            with open(p, encoding="utf-8-sig") as f:
                tuned = json.load(f)
            print(f"Tuned parameters loaded: {len(tuned)} models")
        else:
            print(f"WARNING: {p} does not exist, continuing with default hyper-parameters.")

    warned = set()

    def tuned_for(name):
        if tuned and name not in tuned and name not in warned:
            warned.add(name)
            print(f"WARNING: '{name}' is NOT in {args.tuned_params} -> it will run with the "
                  f"DEFAULT hyper-parameters (while the other models are tuned).")
        return tuned.get(name, {}).get("params", {})

    # --- build the blocks: K-fold OR hold-out seeds ---
    if args.kfold:
        base = seeds[0]
        work = subsample(samples, limit, base) if limit else samples
        blocks = list(enumerate(stratified_kfold_splits(work, args.kfold, base)))
        print(f"K-FOLD: {args.kfold} folds (base seed={base}), blocks = fold index")
    else:
        blocks = []
        for seed in seeds:
            work = subsample(samples, limit, seed) if limit else samples
            blocks.append((seed, stratified_split(work, seed, val_frac=0.15, test_frac=0.15)))

    # --- main loop: block (same split) -> all models ---
    for block_id, splits in blocks:
        tag_kind = "FOLD" if args.kfold else "SEED"
        print(f"\n===== {tag_kind} {block_id} | train={len(splits['train'])} "
              f"val={len(splits['val'])} test={len(splits['test'])} =====")

        cfgs = []
        if not args.no_proposed:
            cfgs.append(("proposed", PROPOSED_NAME,
                         make_proposed_config(seed=block_id, **shared,
                                              **tuned_for(PROPOSED_NAME))))
        for m in model_list:
            cfgs.append(("benchmark", m,
                         make_baseline_config(m, seed=block_id, **shared,
                                              **tuned_for(m))))

        for exp, name, cfg in cfgs:
            run_dir = out_dir / exp / safe_name(name) / f"seed{block_id}"
            tag = f"{exp}/{name}/seed{block_id}"
            try:
                run_one(cfg, run_dir, class_names, splits, ext_samples,
                        device, master_csv, tag)
            except KeyboardInterrupt:
                print("\nInterrupted by the user. Re-run the same command to resume.")
                raise
            except Exception as e:
                print(f"[ERROR] {tag}: {e}")
                traceback.print_exc()
                _append_master(fail_csv, {"tag": tag, "model": name,
                                          "seed": block_id, "error": str(e)})

    ended = datetime.now()
    elapsed = time.time() - t_pipeline
    timing = {"script": "run_benchmark", "started_at": started.isoformat(timespec="seconds"),
              "ended_at": ended.isoformat(timespec="seconds"),
              "total_sec": round(elapsed, 1), "total_human": fmt_duration(elapsed)}
    with open(out_dir / "timing_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(f"\nDone. Summary: {master_csv}  |  Per-model timing: {out_dir / 'timing.csv'}")
    print(f"START: {started:%Y-%m-%d %H:%M:%S}  END: {ended:%Y-%m-%d %H:%M:%S}  "
          f"TOTAL: {fmt_duration(elapsed)}")
    print("For the analysis:  python analyze.py")


if __name__ == "__main__":
    main()
