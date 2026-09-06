"""
Collects the outputs of one external-validation run (split) into a SEPARATE folder.

run_external.py writes its results next to the trained models
(results/<exp>/<model>/seed*/). This script gathers every file that belongs to the
given --split name, copies it into a stand-alone folder and produces a model x metric
summary table. Every external data set is thus kept in its own folder and nothing is
overwritten.

Examples:
    python tools/collect_external.py --split external_mld24 --out results_MLD24
    python tools/collect_external.py --split external --out results_MangoLeafBD
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BASE = Path(__file__).resolve().parent.parent


def find_splits(src: Path):
    """Finds the external-validation split names under results/ (newest first)."""
    seen = {}
    for p in src.glob("*/*/seed*/metrics_external*.json"):
        name = p.stem.replace("metrics_", "")
        seen[name] = max(seen.get(name, 0), p.stat().st_mtime)
    return [k for k, _ in sorted(seen.items(), key=lambda kv: kv[1], reverse=True)]


def default_out(split: str) -> str:
    """external_mld24 -> results_MLD24 ; external -> results_external"""
    tail = split[len("external_"):] if split.startswith("external_") else split
    return f"results_{tail.upper() if tail != split else tail}"


def main():
    ap = argparse.ArgumentParser()
    # PyCharm's Run button passes no arguments; both are optional and auto-detected.
    ap.add_argument("--split", default=None,
                    help="split name (e.g. external_mld24). The newest one is chosen if omitted.")
    ap.add_argument("--out", default=None,
                    help="target folder. Derived from the split name if omitted (results_MLD24).")
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--no_copy", action="store_true",
                    help="do not copy files, only produce the summary")
    args = ap.parse_args()

    src = BASE / args.results_dir
    if not src.exists():
        raise SystemExit(f"'{src}' does not exist.")

    if args.split is None:
        splits = find_splits(src)
        if not splits:
            raise SystemExit(f"No external-validation result under '{src}' "
                             f"(metrics_external*.json). Run run_external.py first.")
        args.split = splits[0]
        extra = f"  (others: {', '.join(splits[1:])})" if len(splits) > 1 else ""
        print(f"[auto] split = '{args.split}'{extra}")

    if args.out is None:
        args.out = default_out(args.split)
        print(f"[auto] target folder = '{args.out}'")

    dst = BASE / args.out
    dst.mkdir(parents=True, exist_ok=True)

    rows, n_files = [], 0
    for mfile in sorted(src.glob(f"*/*/seed*/metrics_{args.split}.json")):
        run = mfile.parent                       # results/<exp>/<model>/seed<N>
        exp, model, seed = run.parent.parent.name, run.parent.name, run.name
        with open(mfile, encoding="utf-8") as f:
            m = json.load(f)
        rows.append({"experiment": exp, "model": model,
                     "seed": seed.replace("seed", ""), **m})

        if not args.no_copy:
            tgt = dst / exp / model / seed
            tgt.mkdir(parents=True, exist_ok=True)
            for p in run.glob(f"*{args.split}*"):
                shutil.copy2(p, tgt / p.name)
                n_files += 1
            for extra in ("config.json", "complexity.json"):
                if (run / extra).exists():
                    shutil.copy2(run / extra, tgt / extra)
                    n_files += 1

    if not rows:
        raise SystemExit(f"No result found for '{args.split}' ({src}).")

    df = pd.DataFrame(rows)
    df.to_csv(dst / f"raw_{args.split}.csv", index=False)

    metrics = [c for c in ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted",
                           "precision_macro", "recall_macro", "auc_macro_ovr", "ece"]
               if c in df.columns]
    agg = (df.groupby(["experiment", "model"])
             .agg(n_seeds=("seed", "nunique"),
                  **{f"{m}_{s}": (m, s) for m in metrics for s in ("mean", "std")})
             .reset_index())
    sort_by = "f1_macro_mean" if "f1_macro_mean" in agg.columns else "accuracy_mean"
    agg = agg.sort_values(sort_by, ascending=False).reset_index(drop=True)
    agg.insert(0, "rank", agg.index + 1)
    agg.to_csv(dst / f"summary_{args.split}.csv", index=False)

    # carry the external-set info over if it exists
    info = src / f"{args.split}_info.json"
    if info.exists():
        shutil.copy2(info, dst / info.name)

    print(f"Collected -> {dst}")
    print(f"  {len(df)} runs, {agg['model'].nunique()} models"
          + ("" if args.no_copy else f", {n_files} files copied"))
    print(f"  summary_{args.split}.csv  |  raw_{args.split}.csv\n")
    cols = [c for c in ["rank", "model", "accuracy_mean", "f1_macro_mean", "n_seeds"]
            if c in agg.columns]
    print(agg[agg.experiment.isin(["benchmark", "proposed"])][cols].head(10)
          .to_string(index=False))


if __name__ == "__main__":
    main()
