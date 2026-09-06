"""
Result aggregation + statistical analysis. Run after run_benchmark / run_ablation.

Produces:
  summary_internal_test.csv / .tex   - model x metric (mean±std over seeds)
  summary_external.csv               - external-validation summary (if any)
  ranking_test.png                   - accuracy ranking (with error bars)
  efficiency_test.png                - accuracy vs parameters (efficiency)
  cd_diagram.png                     - Friedman+Nemenyi critical difference diagram
  proposed_vs_baselines.csv          - paired tests (Wilcoxon/t, Cohen's d)
  mcnemar.json                       - proposed vs best baseline (McNemar)
  ablation_summary.csv               - ablation (delta + significance)
  complexity.csv                     - parameters/GFLOPs
  stats_report.json                  - all statistics in one file

Example:  python analyze.py --out_dir results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from leafdx import viz
from leafdx.stats import (friedman_nemenyi, paired_tests, mcnemar_test,
                          bootstrap_ci, holm_bonferroni)
from leafdx.config import PROPOSED_NAME


def safe_name(s):
    return s.replace("/", "_").replace(".", "-")


def parse_name(tag):
    parts = str(tag).split("/")
    return parts[1] if len(parts) >= 3 else str(tag)


def run_dir_for(out_dir, exp, name, seed):
    return Path(out_dir) / exp / safe_name(name) / f"seed{seed}"


def load_preds(out_dir, exp, name, seed, split="test"):
    p = run_dir_for(out_dir, exp, name, seed) / f"predictions_{split}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return d["y_true"], d["y_pred"], d["y_prob"]


def dedupe(df):
    """If several rows exist for the same (tag, split), keep the LAST one.

    Re-running an interrupted run (crashed before DONE) could append a second row to
    the master CSV; `groupby.mean()` counted that seed twice while `seed.nunique()`
    still showed 3, so the problem stayed invisible.
    """
    if not {"tag", "split"}.issubset(df.columns):
        return df
    n0 = len(df)
    df = df.drop_duplicates(subset=["tag", "split"], keep="last").copy()
    if len(df) < n0:
        print(f"WARNING: master_results.csv had {n0 - len(df)} duplicated (tag, split) "
              f"rows; the last ones were kept.")
    return df


def aggregate(df, split, metrics, experiments=None, sort_by="f1_macro"):
    sub = df[df["split"] == split].copy()
    if experiments is not None:
        sub = sub[sub["experiment"].isin(experiments)]
    if sub.empty:
        return None
    sub["name"] = sub["tag"].map(parse_name)
    g = sub.groupby("name")
    rows = []
    for name, grp in g:
        row = {"model": name, "n_seeds": grp["seed"].nunique()}
        for m in metrics:
            if m in grp:
                row[f"{m}_mean"] = grp[m].mean()
                # ddof=1 (sample std). ddof=0 under-states the spread by ~18% at n=3,
                # and that went straight into the error bars of the ranking plot.
                row[f"{m}_std"] = grp[m].std(ddof=1)
                # in how many seeds this metric could actually be computed (NaNs are not
                # counted): because mean() has skipna=True, missing metrics were silently
                # averaged over a smaller sample
                row[f"{m}_n"] = int(grp[m].count())
        for c in ["params_M", "size_MB", "gflops", "latency_ms",
                  "latency_ms_single_pass", "tta_views", "fps"]:
            if c in grp:
                row[c] = grp[c].iloc[0]
        for c in ["select_src", "hier_beta", "best_epoch"]:
            if c in grp:
                row[c] = grp[c].iloc[0]
        if "train_sec" in grp:
            row["train_sec_mean"] = grp["train_sec"].mean()
        rows.append(row)
    out = pd.DataFrame(rows)
    key = f"{sort_by}_mean" if f"{sort_by}_mean" in out.columns else "accuracy_mean"
    out = out.sort_values(key, ascending=False)

    # Models with missing seeds stay in the table but are FLAGGED: ranking the mean of
    # a single lucky seed next to 3-seed means would be misleading.
    if "n_seeds" in out.columns and len(out):
        full = int(out["n_seeds"].max())
        out["complete"] = out["n_seeds"] >= full
        missing = out[~out["complete"]]["model"].tolist()
        if missing:
            print(f"WARNING: {len(missing)} model(s) did not complete all {full} seeds "
                  f"('complete=False' in the ranking): {missing}")
    return out.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results")
    # Primary metric: macro-F1 on 21 imbalanced classes (209 <-> 500). Model selection
    # (engine.py), the HPO objective (run_optuna.py) and the ranking now use the SAME
    # metric; they used to differ (tune by F1, select by accuracy, rank by accuracy).
    ap.add_argument("--primary_metric", default="f1_macro")
    ap.add_argument("--cd_top_n", type=int, default=8,
                    help="number of top models drawn in the CD diagram "
                         "(CD grows with sqrt(k); with 26 models the diagram is meaningless)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    master = out / "master_results.csv"
    if not master.exists():
        raise SystemExit(f"{master} does not exist. Run run_benchmark.py first.")
    df = dedupe(pd.read_csv(master))
    df["name"] = df["tag"].map(parse_name)
    report = {"primary_metric": args.primary_metric}

    metrics = ["accuracy", "balanced_accuracy", "top5_accuracy",
               "f1_macro", "f1_weighted", "precision_macro", "recall_macro",
               "sensitivity_macro", "specificity_macro", "g_mean_macro",
               "cohen_kappa", "mcc", "auc_macro_ovr", "average_precision_macro", "ece"]
    metrics = [m for m in metrics if m in df.columns]

    # ---------- 1) Internal test summary ----------
    pm = args.primary_metric
    summ = aggregate(df, "internal_test", metrics,
                     experiments=["benchmark", "proposed"], sort_by=pm)
    if summ is not None:
        summ.to_csv(out / "summary_internal_test.csv", index=False)
        # LaTeX (main metrics)
        cols = ["model", f"{pm}_mean", f"{pm}_std", "accuracy_mean", "accuracy_std",
                "auc_macro_ovr_mean", "params_M"]
        cols = list(dict.fromkeys(c for c in cols if c in summ.columns))
        summ[cols].to_latex(out / "summary_internal_test.tex", index=False,
                            float_format="%.4f")
        print(f"[summary] internal test (sorted by {pm}): {len(summ)} models "
              f"-> summary_internal_test.csv")
        print(summ[[c for c in ["model", f"{pm}_mean", f"{pm}_std", "accuracy_mean",
                                "n_seeds"] if c in summ.columns]]
              .head(10).to_string(index=False))

        # ranking plot (primary metric)
        mkey = f"{pm}_mean" if f"{pm}_mean" in summ.columns else "accuracy_mean"
        skey = f"{pm}_std" if f"{pm}_std" in summ.columns else "accuracy_std"
        fig_models = summ["model"].tolist()[::-1]
        acc = summ[mkey].tolist()[::-1]
        err = summ[skey].fillna(0.0).tolist()[::-1]
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, max(4, len(fig_models) * 0.3)))
        colors = ["#d62728" if PROPOSED_NAME in m else "#1f77b4" for m in fig_models]
        ax.barh(range(len(fig_models)), acc, xerr=err, color=colors)
        ax.set_yticks(range(len(fig_models)))
        ax.set_yticklabels(fig_models, fontsize=7)
        ax.set_xlabel(f"Test {pm} (mean ± std)")
        ax.set_title(f"Model ranking ({pm})")
        fig.tight_layout(); fig.savefig(out / "ranking_test.png", dpi=150); plt.close(fig)

        # efficiency plot (primary metric vs parameters)
        if "params_M" in summ.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            for _, r in summ.iterrows():
                c = "#d62728" if PROPOSED_NAME in r["model"] else "#1f77b4"
                ax.scatter(r["params_M"], r[mkey], c=c, s=40)
                ax.annotate(r["model"], (r["params_M"], r[mkey]),
                            fontsize=5, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("Parameters (M)"); ax.set_ylabel(f"Test {pm}")
            ax.set_title("Efficiency: performance vs model size")
            fig.tight_layout(); fig.savefig(out / "efficiency_test.png", dpi=150)
            plt.close(fig)
            cx_cols = [c for c in ["model", "params_M", "size_MB", "gflops",
                                   "latency_ms", "latency_ms_single_pass", "tta_views",
                                   "fps", "train_sec_mean", mkey, "accuracy_mean"]
                       if c in summ.columns]
            summ[list(dict.fromkeys(cx_cols))].to_csv(out / "complexity.csv", index=False)
        report["summary_internal_test"] = summ.to_dict(orient="records")

    # ---------- 2) External-validation summary ----------
    ext = aggregate(df, "external", metrics,
                    experiments=["benchmark", "proposed"], sort_by=pm)
    if ext is not None:
        ext.to_csv(out / "summary_external.csv", index=False)
        report["summary_external"] = ext.to_dict(orient="records")
        print(f"[summary] external validation: {len(ext)} models -> summary_external.csv")

    # ---------- 3) Friedman + Nemenyi + CD diagram ----------
    test = df[df["split"] == "internal_test"].copy()
    test["name"] = test["tag"].map(parse_name)
    # benchmark + proposed models (ablation excluded)
    bench = test[test["experiment"].isin(["benchmark", "proposed"])]
    pivot_raw = bench.pivot_table(index="seed", columns="name",
                                  values=pm, aggfunc="mean")
    pivot = pivot_raw.dropna(axis=1)  # drop models that are not present in every seed
    dropped = [c for c in pivot_raw.columns if c not in pivot.columns]
    if dropped:
        # This used to happen silently: if the proposed model lost a single seed (OOM,
        # interruption) it dropped out of the pivot, the following 'if' silently became
        # False and proposed_vs_baselines.csv + mcnemar.json were NEVER written — while
        # the script finished successfully saying "All analyses written".
        print(f"WARNING: {len(dropped)} model(s) excluded from the statistical tests "
              f"because they are not present in every seed: {dropped}")
        report["dropped_incomplete_models"] = dropped

    if pivot.shape[1] >= 3 and pivot.shape[0] >= 2:
        fn = friedman_nemenyi(pivot.values, list(pivot.columns))
        report["friedman_nemenyi"] = fn
        # CD ~ q(k)*sqrt(k(k+1)/6n): with k=26 and n=3 blocks CD~22.8 and NO model looks
        # distinguishable from any other on a 1..26 rank axis. As Demsar suggests, the
        # diagram is drawn for the best N models; the full matrix stays in the report.
        top = list(summ["model"].head(args.cd_top_n)) if summ is not None else []
        sub_cols = [c for c in pivot.columns if c in top] or list(pivot.columns)
        if PROPOSED_NAME in pivot.columns and PROPOSED_NAME not in sub_cols:
            sub_cols.append(PROPOSED_NAME)
        fn_top = friedman_nemenyi(pivot[sub_cols].values, sub_cols)
        report["friedman_nemenyi_top"] = fn_top
        viz.plot_cd_diagram(fn_top["avg_ranks"], fn_top["nemenyi_cd"],
                            out / "cd_diagram.png")
        print(f"[stat] Friedman p={fn['friedman_p']:.3e} (k={pivot.shape[1]}, "
              f"n={pivot.shape[0]}) | CD(all)={fn['nemenyi_cd']:.2f}, "
              f"CD(top {len(sub_cols)})={fn_top['nemenyi_cd']:.2f} -> cd_diagram.png")
        if fn["nemenyi_cd"] > pivot.shape[1] / 2:
            print(f"      WARNING: with {pivot.shape[0]} blocks Nemenyi has no power "
                  f"(CD larger than half of the rank axis). Use more seeds or "
                  f"--kfold 10.")
    else:
        print("[stat] not enough models/seeds for Friedman (>=3 models, >=2 seeds).")

    # ---------- 4) Proposed vs baselines (paired tests) ----------
    if PROPOSED_NAME not in pivot.columns:
        print(f"WARNING: '{PROPOSED_NAME}' is NOT in the paired-test pivot "
              f"({'missing seed' if PROPOSED_NAME in pivot_raw.columns else 'no run at all'}) "
              f"-> proposed_vs_baselines.csv and mcnemar.json NOT PRODUCED.")
        report["proposed_vs_baselines_skipped"] = True
    if PROPOSED_NAME in pivot.columns and pivot.shape[0] >= 2:
        prop = pivot[PROPOSED_NAME]
        rows = []
        for m in pivot.columns:
            if m == PROPOSED_NAME:
                continue
            pt = paired_tests(prop.values, pivot[m].values)
            rows.append({"baseline": m, "proposed_mean": prop.mean(),
                         "baseline_mean": pivot[m].mean(), **pt})
        pv = pd.DataFrame(rows).sort_values("baseline_mean", ascending=False)
        # multiple-comparison correction (Holm-Bonferroni)
        adj_w, rej_w = holm_bonferroni(pv["wilcoxon_p"].tolist())
        adj_t, _ = holm_bonferroni(pv["paired_t_p"].tolist())
        pv["wilcoxon_p_holm"] = adj_w
        pv["paired_t_p_holm"] = adj_t
        pv["significant_holm"] = rej_w
        pv.to_csv(out / "proposed_vs_baselines.csv", index=False)
        report["proposed_vs_baselines"] = pv.to_dict(orient="records")
        n_sig = int(np.nansum(rej_w))
        print(f"[stat] proposed vs {len(pv)} baselines -> proposed_vs_baselines.csv "
              f"({n_sig} with Holm-corrected p<0.05)")

        # ---------- 5) McNemar: proposed vs best baseline ----------
        best_base = pv.iloc[0]["baseline"]
        mc = {"best_baseline": best_base, "per_seed": []}
        for seed in sorted(bench["seed"].unique()):
            pr = load_preds(out, "proposed", PROPOSED_NAME, seed, "test")
            bb = load_preds(out, "benchmark", best_base, seed, "test")
            # Equal length is NOT ENOUGH: a paired test requires both models to have seen
            # the SAME images in the SAME order. A DONE folder left over from an old run
            # (e.g. a different --limit_per_class) can yield a prediction vector of the
            # same length but different content, and McNemar silently becomes invalid.
            if pr is not None and bb is not None and np.array_equal(pr[0], bb[0]):
                res = mcnemar_test(pr[0], pr[1], bb[1])
                mc["per_seed"].append({"seed": int(seed), **res})
                # bootstrap CI (proposed accuracy)
                acc_fn = lambda t, p: float((t == p).mean())
                est, lo, hi = bootstrap_ci(pr[0], pr[1], acc_fn)
                mc["per_seed"][-1]["proposed_acc_ci95"] = [est, lo, hi]
            else:
                why = ("prediction file missing" if pr is None or bb is None
                       else "test sets are NOT IDENTICAL (old/incompatible run)")
                print(f"      WARNING: seed {seed} skipped in McNemar — {why}.")
                mc.setdefault("skipped_seeds", []).append(
                    {"seed": int(seed), "reason": why})
        with open(out / "mcnemar.json", "w", encoding="utf-8") as f:
            json.dump(mc, f, indent=2)
        report["mcnemar"] = mc
        if mc["per_seed"]:
            print(f"[stat] McNemar (vs {best_base}): "
                  f"p={mc['per_seed'][0]['p_value']:.3e} -> mcnemar.json")

    # ---------- 6) Ablation ----------
    abl = df[df["experiment"] == "ablation"].copy()
    if not abl.empty:
        abl["name"] = abl["tag"].map(parse_name)
        piv = abl[abl["split"] == "internal_test"].pivot_table(
            index="seed", columns="name", values=args.primary_metric, aggfunc="mean")
        rows = []
        full = piv["full"] if "full" in piv.columns else None
        for v in piv.columns:
            r = {"variant": v, f"{pm}_mean": piv[v].mean(),
                 f"{pm}_std": piv[v].std(ddof=1), f"{pm}_n": int(piv[v].count())}
            if full is not None and v != "full":
                r["delta_vs_full"] = piv[v].mean() - full.mean()
                common = piv[[v]].join(full.rename("full")).dropna()
                if len(common) >= 2:
                    pt = paired_tests(common["full"].values, common[v].values)
                    r["wilcoxon_p"] = pt["wilcoxon_p"]
                    r["cohens_d"] = pt["cohens_d"]
                    r["cohens_dz"] = pt["cohens_dz"]
                else:
                    r["wilcoxon_p"] = float("nan")
            rows.append(r)
        ab = pd.DataFrame(rows).sort_values(f"{args.primary_metric}_mean",
                                            ascending=False)
        if "wilcoxon_p" in ab.columns:
            adj, rej = holm_bonferroni(ab["wilcoxon_p"].tolist())
            ab["wilcoxon_p_holm"] = adj
            ab["significant_holm"] = rej
        ab.to_csv(out / "ablation_summary.csv", index=False)
        report["ablation"] = ab.to_dict(orient="records")
        print(f"[ablation] {len(ab)} variants -> ablation_summary.csv")
        print(ab.to_string(index=False))

    with open(out / "stats_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nAll analyses written to '{out}'. (stats_report.json = summary)")


if __name__ == "__main__":
    main()
