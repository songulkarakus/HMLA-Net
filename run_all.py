"""
ONE COMMAND — runs the whole study end to end and produces every result:
  1) Benchmark : 25 baselines + the proposed model  (external validation included)
  2) Ablation  : the component variants of the proposed model
  3) Analysis  : summary tables + statistics (Friedman/Nemenyi, McNemar, Wilcoxon) + CD diagram
  4) Grad-CAM  : explainability heat-maps

The MangoLeafBD external validation set is detected AUTOMATICALLY and its mapping file
is written. Every step supports resume: if it stops, re-run the same command and it
continues where it left off.

Examples:
    python run_all.py                                  # full study (3 seeds)
    python run_all.py --seeds 42 --no_ablation         # fast single seed
    python run_all.py --quick --models resnet18        # end-to-end smoke test
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# keep non-ASCII characters from crashing the Windows console (cp1254)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def fmt_duration(sec):
    sec = float(sec)
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m {s}s" if h else (f"{m}m {s}s" if m else f"{sec:.1f}s")

# MangoLeafBD (Mendeley, DOI 10.17632/hxsnvwty3r.1) folder names -> our mango classes.
# 'Bacterial Canker' and 'Cutting Weevil' have no counterpart in our data and are not mapped.
MANGO_BD_MAP = {
    "Anthracnose": "mango anthracnose",
    "Die Back": "mango die back",
    "Gall Midge": "mango gall midge",
    "Healthy": "mango healthy",
    "Powdery Mildew": "mango powdery mildew",
    "Sooty Mould": "mango sooty mould",
}
DEFAULT_EXTERNAL_DIR = "external dataset/MangoLeafBD Dataset"
DEFAULT_EXTERNAL_MAP = "mango_map.json"


def ensure_external(args):
    """Locates the external-validation folder and prepares the mapping file."""
    if args.no_external:
        return None, None
    ext_dir = args.external_dir
    if ext_dir is None and Path(DEFAULT_EXTERNAL_DIR).exists():
        ext_dir = DEFAULT_EXTERNAL_DIR
    if ext_dir is None:
        print("! External-validation folder not found -> external validation skipped.\n"
              f"  (Put MangoLeafBD under '{DEFAULT_EXTERNAL_DIR}' "
              "or pass --external_dir.)")
        return None, None
    ext_map = args.external_map
    if ext_map is None:
        ext_map = DEFAULT_EXTERNAL_MAP
        if not Path(ext_map).exists():
            Path(ext_map).write_text(
                json.dumps(MANGO_BD_MAP, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"+ External-validation mapping written: {ext_map}")
    print(f"+ External validation: dir='{ext_dir}'  map='{ext_map}'")
    return ext_dir, ext_map


def run_step(name, cmd, fatal_on_fail):
    t0 = time.time()
    print(f"\n{'='*70}\n>>> {name}  ({datetime.now():%H:%M:%S})\n$ {' '.join(cmd)}\n{'='*70}")
    try:
        rc = subprocess.run(cmd).returncode
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run the same command to resume.")
        sys.exit(130)
    dur = time.time() - t0
    print(f">>> '{name}' finished — duration {fmt_duration(dur)}")
    if rc != 0:
        msg = f"[WARNING] step '{name}' returned code {rc}."
        if fatal_on_fail:
            print(msg + " Re-run the same command to resume.")
            sys.exit(rc)
        print(msg + " (non-critical, continuing)")
    return rc, dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="multiple leaf dataset")
    ap.add_argument("--seeds", default="42,1234,2024")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None,
                    help="early-stopping patience (passed to benchmark AND ablation alike)")
    ap.add_argument("--select_metric", default=None,
                    help="checkpoint/early-stopping criterion (default f1_macro)")
    ap.add_argument("--primary_metric", default="f1_macro",
                    help="ranking + statistics metric (analyze.py)")
    ap.add_argument("--models", default="all")
    ap.add_argument("--external_dir", default=None)
    ap.add_argument("--external_map", default=None)
    ap.add_argument("--external_max_per_class", type=int, default=None)
    ap.add_argument("--no_external", action="store_true")
    ap.add_argument("--no_ablation", action="store_true")
    ap.add_argument("--no_gradcam", action="store_true")
    ap.add_argument("--no_dataset_report", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--kfold", type=int, default=None,
                    help="K-fold cross-validation (replaces --seeds when given)")
    ap.add_argument("--deterministic", action="store_true",
                    help="deterministic cuDNN mode (exact reproducibility)")
    ap.add_argument("--optuna", action="store_true",
                    help="run the Optuna HPO first and feed the best parameters to the benchmark")
    ap.add_argument("--n_trials", type=int, default=25, help="Optuna trials per model")
    ap.add_argument("--proposed_trials", type=int, default=None,
                    help="separate trial budget for the proposed model (default n_trials*2)")
    ap.add_argument("--hpo_epochs", type=int, default=8, help="epochs per Optuna trial")
    ap.add_argument("--hpo_limit_per_class", type=int, default=200,
                    help="sub-sample for Optuna (speed)")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    if args.quick:
        args.num_workers = 0
        if args.external_max_per_class is None:
            args.external_max_per_class = 20  # keep the external validation fast

    ext_dir, ext_map = ensure_external(args)

    common = ["--data_dir", args.data_dir,
              "--seeds", args.seeds, "--batch_size", str(args.batch_size),
              "--img_size", str(args.img_size), "--num_workers", str(args.num_workers)]
    if args.epochs is not None:
        common += ["--epochs", str(args.epochs)]
    # patience/select_metric must reach BOTH steps: unless the ablation 'full' variant
    # is trained with the SAME recipe as the proposed model in the benchmark, every
    # delta in the ablation is measured against a different reference.
    if args.patience is not None:
        common += ["--patience", str(args.patience)]
    if args.select_metric is not None:
        common += ["--select_metric", args.select_metric]
    if args.quick:
        common += ["--quick"]
    if args.no_pretrained:
        common += ["--no_pretrained"]
    if args.deterministic:
        common += ["--deterministic"]
    if args.kfold:
        common += ["--kfold", str(args.kfold)]
    ext_args = []
    if ext_dir:
        ext_args = ["--external_dir", ext_dir, "--external_map", ext_map]
        if args.external_max_per_class:
            ext_args += ["--external_max_per_class", str(args.external_max_per_class)]

    # block identifier: the first fold '0' under k-fold, otherwise the first seed
    first_block = "0" if args.kfold else args.seeds.split(",")[0].strip()

    # --- Optuna HPO (optional) -> feed the tuned parameters to benchmark/ablation ---
    tuned_args = []
    optuna_step = None
    if args.optuna:
        best_params = "results/optuna/best_params.json"
        tuned_args = ["--tuned_params", best_params]
        ocmd = [py, "run_optuna.py", "--data_dir", args.data_dir,
                "--img_size", str(args.img_size), "--batch_size", str(args.batch_size),
                "--num_workers", str(args.num_workers), "--n_trials", str(args.n_trials),
                "--hpo_epochs", str(args.hpo_epochs),
                "--limit_per_class", str(args.hpo_limit_per_class),
                "--seed", args.seeds.split(",")[0].strip()]
        if args.proposed_trials is not None:
            ocmd += ["--proposed_trials", str(args.proposed_trials)]
        if args.quick:
            ocmd += ["--quick"]
        if args.no_pretrained:
            ocmd += ["--no_pretrained"]
        if args.models != "all":
            ocmd += ["--models", args.models]
        optuna_step = ("Optuna HPO (all models)", ocmd, True)

    # --- build the steps ---
    steps = []
    if not args.no_dataset_report:
        steps.append(("Data-set report (distribution + samples)",
                      [py, "dataset_report.py", "--data_dir", args.data_dir], False))
    if optuna_step:
        steps.append(optuna_step)

    bcmd = [py, "run_benchmark.py"] + common + ext_args + tuned_args
    if args.models != "all":
        bcmd += ["--models", args.models]
    steps.append(("Benchmark (25 baselines + the proposed model)", bcmd, True))

    if not args.no_ablation:
        from leafdx.config import ablation_variants
        steps.append((f"Ablation ({len(ablation_variants())} variants)",
                      [py, "run_ablation.py"] + common + ext_args + tuned_args, True))

    steps.append(("Analysis + statistics + CD diagram",
                  [py, "analyze.py", "--primary_metric", args.primary_metric], False))

    if not args.no_gradcam:
        steps.append(("Grad-CAM (explainability)",
                      [py, "gradcam.py", "--seed", first_block], False))

    steps.append(("Paper-ready report (REPORT.md)", [py, "make_report.py"], False))

    pipeline_start = time.time()
    started = datetime.now()
    print(f"\n{len(steps)} steps will be run. START: "
          f"{started:%Y-%m-%d %H:%M:%S}. Device check:")
    subprocess.run([py, "-c",
                    "import torch;print('  CUDA:',torch.cuda.is_available())"])

    step_times = []
    for name, cmd, fatal in steps:
        _, dur = run_step(name, cmd, fatal)
        step_times.append({"step": name, "sec": round(dur, 1),
                           "human": fmt_duration(dur)})

    ended = datetime.now()
    total = time.time() - pipeline_start
    Path("results").mkdir(exist_ok=True)
    with open("results/timing_pipeline.json", "w", encoding="utf-8") as f:
        json.dump({"started_at": started.isoformat(timespec="seconds"),
                   "ended_at": ended.isoformat(timespec="seconds"),
                   "total_sec": round(total, 1), "total_human": fmt_duration(total),
                   "steps": step_times}, f, indent=2)

    print(f"\n{'='*70}\nCOMPLETED. Main outputs under 'results/':")
    for f in ["REPORT.md", "summary_internal_test.csv", "summary_external.csv",
              "ranking_test.png", "cd_diagram.png", "proposed_vs_baselines.csv",
              "mcnemar.json", "ablation_summary.csv", "complexity.csv",
              "stats_report.json", "dataset/class_distribution.png",
              "dataset/samples_grid.png"]:
        p = Path("results") / f
        print(f"  [{'OK' if p.exists() else '  '}] results/{f}")
    gc = list(Path("results").rglob("gradcam_grid.png"))
    print(f"  [{'OK' if gc else '  '}] results/proposed/gradcam/gradcam_grid.png")
    print("\n--- Step durations ---")
    for st in step_times:
        print(f"  {st['human']:>12}  {st['step']}")
    print(f"\nSTART: {started:%Y-%m-%d %H:%M:%S}  END: {ended:%Y-%m-%d %H:%M:%S}  "
          f"TOTAL: {fmt_duration(total)}")
    print(">>> Paper-ready summary: results/REPORT.md  |  Timing: results/timing_pipeline.json")
    print("="*70)


if __name__ == "__main__":
    main()
