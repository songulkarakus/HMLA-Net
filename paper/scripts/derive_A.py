"""Replication A (the execution reported in the submitted version): quantities derived from its archived
per-image predictions (replication_A/runs/*/predictions_*.npz).

  1. Zero-shot protocols on MLD24: P1 (six-class macro), P2 (21-label macro), P3 (logits restricted to the six
     shared classes) and the leakage rate (mass assigned to the 15 absent labels), per run and per model.
  2. Clean-test re-scoring: the near-duplicate flags computed for replication B (output/derived/
     _clean_test_pairs_seed*.csv, same partition files) applied to the 128 runs of replication A.
  3. Parameter count versus zero-shot performance (26 models; 24 without VGG): Spearman, linear/quadratic fits.
  4. Zero-recall scan, six-class internal accuracy of the mango classes and the like-for-like drop.
Outputs: output/derived/_A_protocols.csv, _A_protocols_runs.csv, _A_clean_test_scores.csv,
         _A_clean_test_summary.csv, _A_mango6.csv, _A_extra.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score

MAN = Path(__file__).resolve().parents[1]
ROOT = MAN
B = MAN / "replication_B"
A = MAN / "replication_A"
ARCH = MAN / "replication_A" / "runs"
DER = MAN / "output" / "derived"
PROPOSED = "Proposed_HMLA_CAFormerS18"
CLASSES = json.loads((B / "summary" / "class_names.json").read_text())
SHARED = ["mango anthracnose", "mango die back", "mango gall midge", "mango healthy", "mango powdery mildew", "mango sooty mould"]
SIDX = np.array([CLASSES.index(c) for c in SHARED])
ABSENT = np.array([i for i in range(21) if i not in set(SIDX)])
splits = {s: json.loads((MAN / "partitions" / f"split_seed{s}.json").read_text())["splits"] for s in (42, 1234, 2024)}


def norm(m):
    return m.replace("caformer_s18-sail", "caformer_s18.sail")


def run_dirs():
    for exp in ("benchmark", "proposed", "ablation"):
        for d in sorted((ARCH / exp).iterdir()):
            if d.is_dir():
                for sd in sorted(d.glob("seed*")):
                    yield exp, d.name, int(sd.name.replace("seed", "")), sd


rows = []
for exp, model, seed, sd in run_dirs():
    f = sd / "predictions_external_mld24.npz"
    if not f.exists():
        continue
    z = np.load(f)
    yt, yp, pr = z["y_true"], z["y_pred"], z["y_prob"]
    p1_f1 = f1_score(yt, yp, average="macro", labels=SIDX)
    p1_acc = accuracy_score(yt, yp)
    p2_f1 = f1_score(yt, yp, average="macro", labels=np.arange(21), zero_division=0)
    yp3 = SIDX[np.argmax(pr[:, SIDX], axis=1)]
    p3_f1 = f1_score(yt, yp3, average="macro", labels=SIDX)
    p3_acc = accuracy_score(yt, yp3)
    leak = float(np.isin(yp, ABSENT).mean())
    mass = float(pr[:, ABSENT].sum(axis=1).mean())
    rec = np.array([(yp[yt == c] == c).mean() for c in SIDX])
    gmean = float(np.exp(np.mean(np.log(np.clip(rec, 1e-12, None)))))
    rows.append({"experiment": exp, "model": norm(model), "seed": seed, "P1_f1": p1_f1, "P1_acc": p1_acc, "P2_f1": p2_f1,
                 "P3_f1": p3_f1, "P3_acc": p3_acc, "leakage": leak, "mass_absent": mass, "gmean": gmean,
                 "n_zero_recall": int((rec == 0).sum())})
pr_df = pd.DataFrame(rows)
pr_df.to_csv(DER / "_A_protocols_runs.csv", index=False)
agg = pr_df.groupby(["experiment", "model"]).agg(n_seeds=("seed", "count"), P1_f1=("P1_f1", "mean"), P1_acc=("P1_acc", "mean"),
                                                 P2_f1=("P2_f1", "mean"), P3_f1=("P3_f1", "mean"), P3_acc=("P3_acc", "mean"),
                                                 leakage=("leakage", "mean"), leakage_sd=("leakage", "std"),
                                                 gmean=("gmean", "mean"), zero_any=("n_zero_recall", "max"),
                                                 zero_min=("n_zero_recall", "min")).reset_index()
agg.to_csv(DER / "_A_protocols.csv", index=False)
gen = agg[agg.experiment.isin(["benchmark", "proposed"])].copy()
assert len(gen) == 26, len(gen)

flag = {}
for s in (42, 1234, 2024):
    pairs = pd.read_csv(DER / f"_clean_test_pairs_seed{s}.csv")
    f = np.zeros(len(splits[s]["test"]), dtype=bool)
    f[pairs.test_idx.unique()] = True
    flag[s] = f
crows = []
for exp, model, seed, sd in run_dirs():
    f = sd / "predictions_test.npz"
    if not f.exists():
        continue
    z = np.load(f)
    yt, yp = z["y_true"], z["y_pred"]
    labs = np.array([e["label"] for e in splits[seed]["test"]])
    if len(yt) != len(labs) or not (labs == yt).all():
        print("  !! partition order mismatch:", exp, model, seed)
        continue
    keep = ~flag[seed]
    crows.append({"experiment": exp, "model": norm(model), "seed": seed, "n_clean": int(keep.sum()),
                  "f1_full": f1_score(yt, yp, average="macro"), "acc_full": accuracy_score(yt, yp),
                  "f1_clean": f1_score(yt[keep], yp[keep], average="macro"), "acc_clean": accuracy_score(yt[keep], yp[keep]),
                  "acc_flagged": accuracy_score(yt[~keep], yp[~keep])})
cs = pd.DataFrame(crows)
cs.to_csv(DER / "_A_clean_test_scores.csv", index=False)
cg = cs.groupby(["experiment", "model"]).agg(f1_full=("f1_full", "mean"), f1_clean=("f1_clean", "mean"), acc_full=("acc_full", "mean"),
                                             acc_clean=("acc_clean", "mean"), acc_flagged=("acc_flagged", "mean"), n=("seed", "count")).reset_index()
cg["d_f1_pp"] = 100 * (cg.f1_clean - cg.f1_full)
cg["d_acc_pp"] = 100 * (cg.acc_clean - cg.acc_full)
cg.to_csv(DER / "_A_clean_test_summary.csv", index=False)
cgen = cg[cg.experiment.isin(["benchmark", "proposed"])].copy()
cgen["rank_full"] = cgen.f1_full.rank(ascending=False).astype(int)
cgen["rank_clean"] = cgen.f1_clean.rank(ascending=False).astype(int)
pc = cgen[cgen.model == PROPOSED].iloc[0]
clean = {
    "n_configs": int(len(cgen)), "mean_d_f1_pp": round(float(cgen.d_f1_pp.mean()), 3), "min_d_f1_pp": round(float(cgen.d_f1_pp.min()), 3),
    "max_d_f1_pp": round(float(cgen.d_f1_pp.max()), 3), "mean_d_acc_pp": round(float(cgen.d_acc_pp.mean()), 3),
    "n_rises": int((cgen.d_f1_pp > 0).sum()), "n_drops": int((cgen.d_f1_pp < 0).sum()),
    "spread_full_pp": round(float(100 * (cgen.f1_full.max() - cgen.f1_full.min())), 2),
    "spread_clean_pp": round(float(100 * (cgen.f1_clean.max() - cgen.f1_clean.min())), 2),
    "spearman_full_clean": round(float(stats.spearmanr(cgen.f1_full, cgen.f1_clean)[0]), 3),
    "max_rank_move": int((cgen.rank_full - cgen.rank_clean).abs().max()),
    "acc_flagged_mean": round(float(cgen.acc_flagged.mean()), 4), "acc_clean_mean": round(float(cgen.acc_clean.mean()), 4),
    "proposed_f1_full": round(float(pc.f1_full), 4), "proposed_f1_clean": round(float(pc.f1_clean), 4),
    "proposed_rank_full": int(pc.rank_full), "proposed_rank_clean": int(pc.rank_clean), "proposed_d_f1_pp": round(float(pc.d_f1_pp), 2),
    "best_clean_model": cgen.sort_values("f1_clean", ascending=False).iloc[0].model,
    "n_clean_by_seed": {int(s): int((~f).sum()) for s, f in flag.items()},
    "flagged_by_seed": {int(s): int(f.sum()) for s, f in flag.items()},
    "n_runs": int(len(cs)),
}

it = pd.read_csv(A / "summary" / "summary_internal_test.csv")
it["model"] = it.model.map(norm)
ex = pd.read_csv(A / "summary" / "summary_external_mld24.csv")
ex["model"] = ex.model.map(norm)
ex = ex[ex.experiment.isin(["benchmark", "proposed"])]
m = it[["model", "params_M", "f1_macro_mean"]].merge(ex[["model", "accuracy_mean", "f1_macro_mean"]], on="model", suffixes=("_int", "_ext"))
assert len(m) == 26
rng = np.random.default_rng(0)


def param_fit(x_params, y):
    x = np.log10(x_params)
    rho, p = stats.spearmanr(x_params, y)
    lin = np.polyfit(x, y, 1); quad = np.polyfit(x, y, 2)
    ss = ((y - y.mean()) ** 2).sum()
    r2l = 1 - ((y - np.polyval(lin, x)) ** 2).sum() / ss
    r2q = 1 - ((y - np.polyval(quad, x)) ** 2).sum() / ss
    n = len(y)
    fstat = ((r2q - r2l) / 1) / ((1 - r2q) / (n - 3))
    pf = 1 - stats.f.cdf(fstat, 1, n - 3)
    vertex = 10 ** (-quad[1] / (2 * quad[0])) if quad[0] < 0 else None
    inside = bool(quad[0] < 0 and x.min() <= np.log10(vertex) <= x.max()) if vertex else False
    vs = []
    for _ in range(1000):
        idx = rng.integers(0, n, n)
        q = np.polyfit(x[idx], y[idx], 2)
        if q[0] < 0:
            v = -q[1] / (2 * q[0])
            if x.min() <= v <= x.max():
                vs.append(10 ** v)
    bins = {}
    for lab, lo, hi in [("small", 0, 10), ("mid", 10, 50), ("large", 50, 1e9)]:
        sel = (x_params >= lo) & (x_params < hi)
        bins[lab] = {"n": int(sel.sum()), "mean": round(float(y[sel].mean()), 4), "sd": round(float(y[sel].std(ddof=1)), 4) if sel.sum() > 1 else None}
    kw = stats.kruskal(*[y[(x_params >= lo) & (x_params < hi)] for lo, hi in [(0, 10), (10, 50), (50, 1e9)]])
    return {"spearman_rho": round(float(rho), 3), "spearman_p": round(float(p), 3), "linear_R2": round(float(r2l), 3),
            "quadratic_R2": round(float(r2q), 3), "quad_vs_lin_p": float(pf), "concave": bool(quad[0] < 0),
            "vertex_M": round(float(vertex), 1) if vertex else None, "vertex_inside": inside,
            "vertex_ci95": [round(float(np.percentile(vs, 2.5)), 1), round(float(np.percentile(vs, 97.5)), 1)] if vs else None,
            "share_bootstrap_inverted_U": round(len(vs) / 1000, 3), "bins": bins, "kruskal_p": round(float(kw.pvalue), 4)}


params = {"external_f1": param_fit(m.params_M.values, m.f1_macro_mean_ext.values),
          "external_acc": param_fit(m.params_M.values, m.accuracy_mean.values),
          "internal_f1": param_fit(m.params_M.values, m.f1_macro_mean_int.values)}

zr = pr_df[pr_df.experiment.isin(["benchmark", "proposed"])].pivot_table(index="model", columns="seed", values="n_zero_recall")
zero = {"n_models": int(len(zr)), "any_seed": int((zr.max(axis=1) > 0).sum()), "all_seeds": int((zr.min(axis=1) > 0).sum()),
        "never": int((zr.max(axis=1) == 0).sum()), "never_models": sorted(zr.index[zr.max(axis=1) == 0].tolist()),
        "n_model_seed_instances": int((zr > 0).sum().sum())}

rows6 = []
for exp, model, seed, sd in run_dirs():
    if exp not in ("benchmark", "proposed"):
        continue
    f = sd / "predictions_test.npz"
    if not f.exists():
        continue
    z = np.load(f)
    yt, yp = z["y_true"], z["y_pred"]
    mask = np.isin(yt, SIDX)
    rows6.append({"model": norm(model), "seed": seed, "n6": int(mask.sum()),
                  "acc6": float((yp[mask] == yt[mask]).mean()),
                  "macro_recall6": float(np.mean([(yp[yt == c] == c).mean() for c in SIDX]))})
m6 = pd.DataFrame(rows6).groupby("model").agg(acc6=("acc6", "mean"), macro_recall6=("macro_recall6", "mean"),
                                              n6=("n6", "mean"), n_seeds=("seed", "count")).reset_index()
m6 = m6.merge(ex[["model", "accuracy_mean"]].rename(columns={"accuracy_mean": "ext_acc"}), on="model")
m6 = m6.merge(it[["model", "accuracy_mean"]].rename(columns={"accuracy_mean": "int_acc21"}), on="model")
assert len(m6) == 26, len(m6)
m6["drop21_pp"] = 100 * (m6.int_acc21 - m6.ext_acc)
m6["drop6_pp"] = 100 * (m6.acc6 - m6.ext_acc)
m6["rank_drop21"] = m6.drop21_pp.rank(method="min").astype(int)
m6["rank_drop6"] = m6.drop6_pp.rank(method="min").astype(int)
m6 = m6.sort_values("drop6_pp").reset_index(drop=True)
m6.to_csv(DER / "_A_mango6.csv", index=False)
p6 = m6[m6.model == PROPOSED].iloc[0]
mango6 = {"proposed_acc6": round(float(p6.acc6), 4), "proposed_macro_recall6": round(float(p6.macro_recall6), 4),
          "proposed_drop6_pp": round(float(p6.drop6_pp), 1), "proposed_drop21_pp": round(float(p6.drop21_pp), 1),
          "proposed_rank_drop6": int(p6.rank_drop6), "proposed_rank_drop21": int(p6.rank_drop21),
          "median_drop6_pp": round(float(m6.drop6_pp.median()), 1), "median_drop21_pp": round(float(m6.drop21_pp.median()), 1),
          "max_drop6_pp": round(float(m6.drop6_pp.max()), 1), "max_drop6_model": m6.loc[m6.drop6_pp.idxmax(), "model"],
          "min_drop6_pp": round(float(m6.drop6_pp.min()), 1), "min_drop6_model": m6.loc[m6.drop6_pp.idxmin(), "model"],
          "n_acc6_perfect": int((m6.acc6 >= 0.9995).sum()), "acc6_min": round(float(m6.acc6.min()), 4),
          "acc6_min_model": m6.loc[m6.acc6.idxmin(), "model"], "acc6_mean": round(float(m6.acc6.mean()), 4),
          "spearman_drop21_drop6": round(float(stats.spearmanr(m6.drop21_pp, m6.drop6_pp)[0]), 3),
          "n_test_mango6_mean": round(float(m6.n6.mean()), 1)}

mnv = m[~m.model.isin(["vgg16", "vgg19"])]
params["external_acc_novgg"] = param_fit(mnv.params_M.values, mnv.accuracy_mean.values)

pg = gen[gen.model == PROPOSED].iloc[0]
gen = gen.sort_values("P1_f1", ascending=False).reset_index(drop=True)
gen["rank_p1"] = np.arange(1, len(gen) + 1)
gen["rank_p3"] = gen.P3_f1.rank(ascending=False, method="min").astype(int)
extra = {
    "protocols": {
        "proposed_leakage": round(float(pg.leakage), 3), "proposed_leakage_sd": round(float(pg.leakage_sd), 3),
        "proposed_p3_f1": round(float(pg.P3_f1), 4), "proposed_p3_acc": round(float(pg.P3_acc), 4), "proposed_p2_f1": round(float(pg.P2_f1), 4),
        "proposed_rank_p3": int(gen.loc[gen.model == PROPOSED, "rank_p3"].iloc[0]),
        "leakage_min": round(float(gen.leakage.min()), 3), "leakage_min_model": gen.loc[gen.leakage.idxmin(), "model"],
        "leakage_max": round(float(gen.leakage.max()), 3), "leakage_max_model": gen.loc[gen.leakage.idxmax(), "model"],
        "leakage_median": round(float(gen.leakage.median()), 3),
        "best_p3_model": gen.loc[gen.P3_f1.idxmax(), "model"], "best_p3_f1": round(float(gen.P3_f1.max()), 4),
        "spearman_p1_p3": round(float(stats.spearmanr(gen.P1_f1, gen.P3_f1)[0]), 3),
        "p3_minus_p1_mean_pp": round(float(100 * (gen.P3_f1 - gen.P1_f1).mean()), 2),
        "p3_minus_p1_min_pp": round(float(100 * (gen.P3_f1 - gen.P1_f1).min()), 2),
        "p3_minus_p1_max_pp": round(float(100 * (gen.P3_f1 - gen.P1_f1).max()), 2),
        "p2_over_p1": round(float((gen.P2_f1 / gen.P1_f1).mean()), 4),
        "leakage_by_model": {r.model: round(float(r.leakage), 3) for r in gen.itertuples()},
        "p3_by_model": {r.model: round(float(r.P3_f1), 4) for r in gen.itertuples()},
    },
    "clean_test": clean, "params": params, "zero_recall": zero, "mango6": mango6,
}
(DER / "_A_extra.json").write_text(json.dumps(extra, indent=2))
print(json.dumps(extra, indent=1)[:4000])
