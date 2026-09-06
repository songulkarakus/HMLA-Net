"""
Ablation: removes one component at a time from the proposed model (HMLA-Net) and measures its contribution.
Multiple seeds, external validation, full resume support.

Examples:
    python run_ablation.py --seeds 42,1234,2024
    python run_ablation.py --quick
"""
import argparse
import json
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from leafdx.config import make_proposed_config, ablation_variants, PROPOSED_NAME
from leafdx.utils import fmt_duration
from leafdx.data import (build_index, stratified_split, stratified_kfold_splits,
                         build_external_index)
from leafdx.runner import run_one, _append_master
from run_benchmark import safe_name, subsample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="multiple leaf dataset")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=None)
    # --patience and --select_metric existed in run_benchmark but NOT here: when the
    # benchmark was run with --patience 12 the 'full' variant was trained with 8, and
    # every delta in ablation_summary.csv was measured against a DIFFERENTLY trained
    # reference ('full' could not match the proposed-model result of the benchmark).
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--select_metric", default=None)
    ap.add_argument("--seeds", default="42,1234,2024")
    ap.add_argument("--variants", default="all", help="'all' or comma-separated variant names")
    ap.add_argument("--external_dir", default=None)
    ap.add_argument("--external_map", default=None)
    ap.add_argument("--external_max_per_class", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--limit_per_class", type=int, default=None)
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--kfold", type=int, default=None)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--tuned_params", default=None,
                    help="Optuna best_params.json (hyper-parameters of the proposed model)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    master_csv = out_dir / "master_results.csv"
    fail_csv = out_dir / "failures.csv"
    t_pipeline = time.time()
    started = datetime.now()
    print(f"Device: {device}")
    print(f"START: {started:%Y-%m-%d %H:%M:%S}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    finetune_epochs, freeze_epochs, limit = args.epochs, None, args.limit_per_class
    if args.quick:
        args.img_size, finetune_epochs, freeze_epochs = 96, 1, 0
        limit = limit or 40
        seeds = seeds[:1]
        print("QUICK mode")

    all_variants = ablation_variants()
    names = list(all_variants) if args.variants == "all" else \
        [v.strip() for v in args.variants.split(",") if v.strip()]

    samples, class_names, n_dup = build_index(args.data_dir, dedup=True)
    print(f"Classes: {len(class_names)} | images: {len(samples)}")

    ext_samples = None
    if args.external_dir and args.external_map:
        ext_samples, info = build_external_index(
            args.external_dir, class_names, args.external_map,
            max_per_class=args.external_max_per_class)
        print(f"External validation: {info['n_images']} images, "
              f"{len(info['covered_classes'])} classes")

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

    tuned_params = {}
    if args.tuned_params and Path(args.tuned_params).exists():
        with open(args.tuned_params, encoding="utf-8-sig") as f:
            tuned_params = json.load(f).get(PROPOSED_NAME, {}).get("params", {})
        print(f"Tuned parameters (proposed model): {tuned_params}")

    if args.kfold:
        base = seeds[0]
        work = subsample(samples, limit, base) if limit else samples
        blocks = list(enumerate(stratified_kfold_splits(work, args.kfold, base)))
        print(f"K-FOLD: {args.kfold} folds (base seed={base})")
    else:
        blocks = []
        for seed in seeds:
            work = subsample(samples, limit, seed) if limit else samples
            blocks.append((seed, stratified_split(work, seed, val_frac=0.15, test_frac=0.15)))

    for block_id, splits in blocks:
        print(f"\n===== BLOCK {block_id} | ablation =====")
        for vname in names:
            # Precedence: tuned HP < shared run settings < VARIANT.
            # `shared` used to be applied last; with --epochs/--quick the freeze_epochs
            # in shared silently overrode the variant's own freeze_epochs=0
            # (wo_two_phase).
            overrides = dict(tuned_params)
            overrides.update(shared)
            overrides.update(all_variants[vname])       # the variant ALWAYS wins
            overrides.update(seed=block_id, experiment="ablation")
            cfg = make_proposed_config(**overrides)
            run_dir = out_dir / "ablation" / safe_name(vname) / f"seed{block_id}"
            tag = f"ablation/{vname}/seed{block_id}"
            try:
                run_one(cfg, run_dir, class_names, splits, ext_samples,
                        device, master_csv, tag)
            except KeyboardInterrupt:
                print("\nInterrupted. Re-run the same command to resume.")
                raise
            except Exception as e:
                print(f"[ERROR] {tag}: {e}")
                traceback.print_exc()
                _append_master(fail_csv, {"tag": tag, "model": vname,
                                          "seed": block_id, "error": str(e)})

    ended = datetime.now()
    elapsed = time.time() - t_pipeline
    with open(out_dir / "timing_ablation.json", "w", encoding="utf-8") as f:
        json.dump({"script": "run_ablation",
                   "started_at": started.isoformat(timespec="seconds"),
                   "ended_at": ended.isoformat(timespec="seconds"),
                   "total_sec": round(elapsed, 1),
                   "total_human": fmt_duration(elapsed)}, f, indent=2)
    print(f"\nAblation finished: {master_csv}")
    print(f"START: {started:%Y-%m-%d %H:%M:%S}  END: {ended:%Y-%m-%d %H:%M:%S}  "
          f"TOTAL: {fmt_duration(elapsed)}")


if __name__ == "__main__":
    main()
