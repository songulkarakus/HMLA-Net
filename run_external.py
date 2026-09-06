"""
Independent EXTERNAL VALIDATION (cross-dataset) — evaluates the trained models on any
external data set WITHOUT RETRAINING.

Why a separate script?
  run_benchmark/run_ablation perform the external validation right after training.
  Changing the external data set would then require retraining everything. This script
  loads the saved best.pth weights under 'results/' and only runs inference (minutes,
  not hours).

Produces (in every run folder):
  metrics_<split>.json, per_class_<split>.csv, predictions_<split>.npz,
  confusion_<split>.png, roc/pr/calibration_<split>.png
And an aggregate summary:
  results/eval_<split>.csv   (model x metric, averaged over seeds)

Examples:
    # all trained models on the default external set (MangoLeafBD)
    python run_external.py

    # another external set under a separate split name (does not overwrite old results)
    python run_external.py --external_dir "external dataset/ccmt" --external_map ccmt_map.json --split_name external_ccmt

    # the proposed model only
    python run_external.py --experiments proposed
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch

# keep non-ASCII characters from crashing the Windows console (cp1252/cp1254)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from leafdx.config import RunConfig
from leafdx.data import build_external_index, make_external_loader
from leafdx.engine import evaluate
from leafdx.models import build_model
from leafdx.runner import _save_split_results, _append_master
from leafdx.utils import fmt_duration


def find_runs(out_dir: Path, experiments, models=None):
    """Finds the completed runs that have results/<exp>/<model>/seed<N>/best.pth."""
    runs = []
    for exp in experiments:
        for best in sorted((out_dir / exp).glob("*/seed*/best.pth")):
            run_dir = best.parent
            name = run_dir.parent.name
            if models and name not in models:
                continue
            seed = run_dir.name.replace("seed", "")
            runs.append({"exp": exp, "name": name, "seed": seed, "run_dir": run_dir,
                         "best": best, "tag": f"{exp}/{name}/seed{seed}"})
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--external_dir", default="external dataset/MangoLeafBD Dataset")
    ap.add_argument("--external_map", default="mango_map.json")
    ap.add_argument("--external_max_per_class", type=int, default=None,
                    help="max images per class (speed/balance)")
    ap.add_argument("--split_name", default="external",
                    help="output name; use a different name for every external set "
                         "(e.g. external_ccmt) so that they do not overwrite each other")
    ap.add_argument("--experiments", default="benchmark,proposed",
                    help="comma-separated: benchmark,proposed,ablation")
    ap.add_argument("--models", default=None, help="comma-separated model/variant names (optional)")
    ap.add_argument("--batch_size", type=int, default=None, help="overrides the config")
    ap.add_argument("--num_workers", type=int, default=None, help="overrides the config")
    ap.add_argument("--no_figs", action="store_true", help="do not produce figures (faster)")
    ap.add_argument("--force", action="store_true",
                    help="re-evaluate runs that already have results")
    args = ap.parse_args()

    out_dir = Path(args.results_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    experiments = [e.strip() for e in args.experiments.split(",") if e.strip()]
    models = [m.strip() for m in args.models.split(",")] if args.models else None

    cn_path = out_dir / "class_names.json"
    if not cn_path.exists():
        raise SystemExit(f"{cn_path} does not exist. Run run_benchmark.py first.")
    with open(cn_path, encoding="utf-8") as f:
        class_names = json.load(f)

    ext_samples, info = build_external_index(
        args.external_dir, class_names, args.external_map,
        max_per_class=args.external_max_per_class)
    print(f"Device: {device}")
    print(f"External set : '{args.external_dir}'  (mapping: {args.external_map})")
    print(f"Images       : {info['n_images']} | classes covered: {len(info['covered_classes'])}")
    print(f"Classes      : {', '.join(info['covered_classes'])}")
    print(f"Split name   : '{args.split_name}'\n")
    with open(out_dir / f"{args.split_name}_info.json", "w", encoding="utf-8") as f:
        json.dump({**info, "external_dir": args.external_dir,
                   "external_map": args.external_map}, f, ensure_ascii=False, indent=2)

    runs = find_runs(out_dir, experiments, models)
    if not runs:
        raise SystemExit(f"No trained models to evaluate ({experiments}).")
    print(f"{len(runs)} trained runs found.\n")

    master_csv = out_dir / "master_results.csv"
    rows, t0, n_new = [], time.time(), 0
    for i, r in enumerate(runs, 1):
        done_marker = r["run_dir"] / f"metrics_{args.split_name}.json"
        if done_marker.exists() and not args.force:
            # CAUTION: the existing metrics may stem from ANOTHER external data set.
            # If you changed the external set, pass --force or use a different --split_name.
            print(f"[{i}/{len(runs)}] [skip] {r['tag']} (existing '{args.split_name}' metrics "
                  f"reused; pass --force if the external set changed)")
            with open(done_marker, encoding="utf-8") as f:
                m = json.load(f)
            rows.append({"tag": r["tag"], "experiment": r["exp"], "model": r["name"],
                         "seed": r["seed"], **m})
            continue
        try:
            ck = torch.load(r["best"], map_location=device, weights_only=False)
            cfg = RunConfig(**ck["cfg"])
            if args.batch_size:
                cfg.batch_size = args.batch_size
            if args.num_workers is not None:
                cfg.num_workers = args.num_workers
            model = build_model(cfg, len(class_names)).to(device)
            model.load_state_dict(ck["model_state"])
            model.eval()

            loader = make_external_loader(ext_samples, cfg, device)
            yt, yp, yprob, _ = evaluate(model, loader, device,
                                        cfg.use_amp and device == "cuda", tta=cfg.use_tta)
            m = _save_split_results(yt, yp, yprob, class_names, r["run_dir"],
                                    args.split_name, make_figs=not args.no_figs)
            print(f"[{i}/{len(runs)}] {r['tag']}: acc={m['accuracy']:.4f} "
                  f"f1={m['f1_macro']:.4f}")
            row = {"tag": r["tag"], "experiment": r["exp"], "model": r["name"],
                   "seed": r["seed"], "split": args.split_name, "n_test": len(yt),
                   **{k: round(v, 6) if isinstance(v, float) else v for k, v in m.items()}}
            _append_master(master_csv, row)
            rows.append(row)
            n_new += 1
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"[{i}/{len(runs)}] [ERROR] {r['tag']}: {e}")

    # ---- aggregate summary (model x metric, mean over seeds) ----
    if rows:
        df = pd.DataFrame(rows)
        metrics = [c for c in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
                               "precision_macro", "recall_macro", "auc_macro_ovr", "ece"]
                   if c in df.columns]
        agg = df.groupby("model").agg(
            n_seeds=("seed", "nunique"),
            **{f"{m}_{s}": (m, s) for m in metrics for s in ("mean", "std")})
        agg = agg.reset_index().sort_values("accuracy_mean", ascending=False)
        # a separate name so that it does NOT collide with the summary_<split>.csv of analyze.py
        sum_path = out_dir / f"eval_{args.split_name}.csv"
        agg.to_csv(sum_path, index=False)
        print(f"\n=== {args.split_name} summary (top 10) ===")
        cols = [c for c in ["model", "n_seeds", "accuracy_mean", "accuracy_std",
                            "f1_macro_mean"] if c in agg.columns]
        print(agg[cols].head(10).to_string(index=False))
        print(f"\nSummary written: {sum_path}")
        if n_new:
            print(f"{n_new} new rows appended to the master: {master_csv} "
                  f"(split='{args.split_name}') -> refresh the summaries with analyze.py")
        else:
            print(f"No new evaluation was performed ({len(rows)} runs reported with their "
                  f"existing metrics); the master is unchanged.")
    print(f"TOTAL TIME: {fmt_duration(time.time() - t0)}")


if __name__ == "__main__":
    main()
