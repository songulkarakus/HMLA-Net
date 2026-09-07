"""Derives every number quoted in the manuscript from the archived run outputs -> output/numbers.json.

Two complete executions of the benchmark are used: replication A (submitted version; replication_A/) and
replication B (this revision; replication_B/). Intermediate tables are written to output/derived/.
Run derive_A.py first.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MAN = Path(__file__).resolve().parents[1]
ROOT = MAN
B = MAN / "replication_B"
A = MAN / "replication_A"
ARCH = MAN / "replication_A" / "runs"
OUT = MAN / "output" / "derived"
OUT.mkdir(parents=True, exist_ok=True)

PROPOSED = "Proposed_HMLA_CAFormerS18"
BACKBONE = "caformer_s18.sail_in22k_ft_in1k"
GENERIC = ["benchmark", "proposed"]
N = {}


def r(x, k=4):
    if x is None:
        return None
    x = float(x)
    return None if np.isnan(x) else round(x, k)


def norm(m):
    return str(m).replace("caformer_s18-sail", "caformer_s18.sail")


def cfg_of(tag):
    return norm(tag.rsplit("/", 1)[0])


CLASSES = json.loads((B / "summary" / "class_names.json").read_text())
SHARED = ["mango anthracnose", "mango die back", "mango gall midge", "mango healthy",
          "mango powdery mildew", "mango sooty mould"]

mb = pd.read_csv(B / "summary" / "master_results.csv")
mb["cfg"] = mb.tag.map(cfg_of)
ma = pd.read_csv(A / "summary" / "master_results.csv")
ma["cfg"] = ma.tag.map(cfg_of)
ma_int = ma[ma.split == "internal_test"]
ma_ext = ma[ma.split == "external_mld24"]
mb_int = mb[mb.split == "internal_test"]
mb_ext = mb[mb.split == "external"]

ds = json.loads((A / "dataset_eda" / "dataset_summary.json").read_text())
cls = pd.read_csv(A / "dataset_eda" / "class_distribution.csv")
N["dataset"] = {**ds, "n_test_images": 1077, "classes": cls.to_dict("records")}

it = pd.read_csv(B / "summary" / "summary_internal_test.csv")
it["model"] = it.model.map(norm)
it["base"] = it.model.str.replace(" [matched]", "", regex=False)
it["matched"] = it.model.str.contains("[matched]", regex=False)
it = it.sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)
it["rank31"] = np.arange(1, len(it) + 1)
it26 = it[~it.matched].reset_index(drop=True)
it26["rank26"] = np.arange(1, len(it26) + 1)
it = it.merge(it26[["model", "rank26"]], on="model", how="left")
it.to_csv(OUT / "_internal_B.csv", index=False)
p = it[it.model == PROPOSED].iloc[0]
bb = it[it.model == BACKBONE].iloc[0]
N["internal"] = {
    "n_models": int(len(it)), "n_generic": int(len(it26)),
    "best": it.iloc[0].model, "best_f1": r(it.iloc[0].f1_macro_mean),
    "best_acc": r(it.accuracy_mean.max()), "best_acc_model": it.loc[it.accuracy_mean.idxmax(), "model"],
    "proposed_f1": r(p.f1_macro_mean), "proposed_f1_sd": r(p.f1_macro_std),
    "proposed_acc": r(p.accuracy_mean), "proposed_acc_sd": r(p.accuracy_std),
    "proposed_bacc": r(p.balanced_accuracy_mean), "proposed_mcc": r(p.mcc_mean),
    "proposed_auc": r(p.auc_macro_ovr_mean), "proposed_ece": r(p.ece_mean),
    "proposed_rank31": int(p.rank31), "proposed_rank26": int(p.rank26),
    "f1_min26": r(it26.f1_macro_mean.min()), "f1_max26": r(it26.f1_macro_mean.max()),
    "f1_spread26_pp": r(100 * (it26.f1_macro_mean.max() - it26.f1_macro_mean.min()), 2),
    "f1_spread31_pp": r(100 * (it.f1_macro_mean.max() - it.f1_macro_mean.min()), 2),
    "worst_model": it.iloc[-1].model,
    "backbone_f1": r(bb.f1_macro_mean), "backbone_rank31": int(bb.rank31),
    "backbone_rank26": int(bb.rank26),
    "gap_1_2_pp": r(100 * (it.iloc[0].f1_macro_mean - it.iloc[1].f1_macro_mean), 3),
    "gap_1_8_pp": r(100 * (it26.iloc[0].f1_macro_mean - it26.iloc[7].f1_macro_mean), 3),
    "gap_1_proposed_pp": r(100 * (it.iloc[0].f1_macro_mean - p.f1_macro_mean), 2),
    "n_within_1pp_of_best": int((it.f1_macro_mean >= it.f1_macro_mean.max() - 0.01).sum()),
}

ex = pd.read_csv(B / "summary" / "summary_external.csv")
ex["model"] = ex.model.map(norm)
ex["base"] = ex.model.str.replace(" [matched]", "", regex=False)
ex["matched"] = ex.model.str.contains("[matched]", regex=False)
pr = pd.read_csv(B / "zero_shot_protocols" / "zeroshot_protocols.csv")
pr["model"] = pr.model.map(norm)
pr = pr[pr.experiment.isin(["benchmark", "proposed", "recipe_matched"])].copy()
pr["model"] = np.where(pr.experiment == "recipe_matched", pr.model + " [matched]", pr.model)
ex = ex.merge(pr[["model", "P1_f1_macro_mean", "P2_f1_macro_mean", "P3_accuracy_mean",
                  "P3_f1_macro_mean", "P3_f1_macro_std", "leakage_rate_mean",
                  "leakage_rate_std", "mean_prob_mass_on_absent_mean"]], on="model", how="left")
assert ex.P3_f1_macro_mean.notna().all(), ex[ex.P3_f1_macro_mean.isna()].model.tolist()
gm = mb_ext.groupby("cfg").g_mean_macro.mean()
def _cfgname(m):
    if m == PROPOSED:
        return "proposed/" + PROPOSED
    if m.endswith(" [matched]"):
        return "recipe_matched/" + m.replace(" [matched]", "")
    return "benchmark/" + m
ex["gmean"] = ex.model.map(lambda m: gm[_cfgname(m)])
ex = ex.sort_values("f1_macro_mean", ascending=False).reset_index(drop=True)
ex["rank31_f1"] = np.arange(1, len(ex) + 1)
ex["rank31_acc"] = ex.accuracy_mean.rank(ascending=False, method="min").astype(int)
ex["rank31_p3"] = ex.P3_f1_macro_mean.rank(ascending=False, method="min").astype(int)
ex26 = ex[~ex.matched].reset_index(drop=True)
ex26["rank26_f1"] = np.arange(1, len(ex26) + 1)
ex26["rank26_acc"] = ex26.accuracy_mean.rank(ascending=False, method="min").astype(int)
ex = ex.merge(ex26[["model", "rank26_f1", "rank26_acc"]], on="model", how="left")
ex = ex.merge(it[["model", "rank31", "rank26", "f1_macro_mean", "accuracy_mean"]]
              .rename(columns={"rank31": "int_rank31", "rank26": "int_rank26",
                               "f1_macro_mean": "int_f1", "accuracy_mean": "int_acc"}),
              on="model", how="left")
ex["drop_acc_pp"] = 100 * (ex.int_acc - ex.accuracy_mean)
ex["drop_f1_pp"] = 100 * (ex.int_f1 - ex.f1_macro_mean)
ex.to_csv(OUT / "_external_B.csv", index=False)
pe = ex[ex.model == PROPOSED].iloc[0]
be = ex[ex.model == BACKBONE].iloc[0]
bme = ex[ex.model == BACKBONE + " [matched]"].iloc[0]
ex26s = ex[~ex.matched]
N["external"] = {
    "n_images": 4800, "n_classes": 6, "n_models": int(len(ex)), "n_generic": int(len(ex26s)),
    "proposed_acc": r(pe.accuracy_mean), "proposed_acc_sd": r(pe.accuracy_std),
    "proposed_f1": r(pe.f1_macro_mean), "proposed_f1_sd": r(pe.f1_macro_std),
    "proposed_prec": r(pe.precision_macro_mean), "proposed_auc": r(pe.auc_macro_ovr_mean),
    "proposed_ece": r(pe.ece_mean), "proposed_gmean": r(pe.gmean, 3),
    "proposed_p3_f1": r(pe.P3_f1_macro_mean), "proposed_p3_acc": r(pe.P3_accuracy_mean),
    "proposed_p2_f1": r(pe.P2_f1_macro_mean),
    "proposed_leakage": r(pe.leakage_rate_mean, 3), "proposed_leakage_sd": r(pe.leakage_rate_std, 3),
    "proposed_rank31_f1": int(pe.rank31_f1), "proposed_rank31_acc": int(pe.rank31_acc),
    "proposed_rank26_f1": int(pe.rank26_f1), "proposed_rank26_acc": int(pe.rank26_acc),
    "proposed_rank31_p3": int(pe.rank31_p3),
    "best_f1_model": ex.iloc[0].model, "best_f1": r(ex.iloc[0].f1_macro_mean),
    "best_f1_sd": r(ex.iloc[0].f1_macro_std),
    "best_acc_model": ex.loc[ex.accuracy_mean.idxmax(), "model"], "best_acc": r(ex.accuracy_mean.max()),
    "best_p3_model": ex.loc[ex.P3_f1_macro_mean.idxmax(), "model"],
    "best_p3_f1": r(ex.P3_f1_macro_mean.max()),
    "second_acc_model": ex.sort_values("accuracy_mean", ascending=False).iloc[1].model,
    "second_acc": r(ex.sort_values("accuracy_mean", ascending=False).iloc[1].accuracy_mean),
    "backbone_acc": r(be.accuracy_mean), "backbone_f1": r(be.f1_macro_mean),
    "backbone_rank31_f1": int(be.rank31_f1), "backbone_rank31_acc": int(be.rank31_acc),
    "backbone_leakage": r(be.leakage_rate_mean, 3),
    "backbone_matched_acc": r(bme.accuracy_mean), "backbone_matched_f1": r(bme.f1_macro_mean),
    "backbone_matched_f1_sd": r(bme.f1_macro_std),
    "backbone_matched_leakage": r(bme.leakage_rate_mean, 3),
    "gain_over_backbone_f1_pp": r(100 * (pe.f1_macro_mean - be.f1_macro_mean), 2),
    "gain_over_backbone_acc_pp": r(100 * (pe.accuracy_mean - be.accuracy_mean), 2),
    "gain_over_matched_backbone_f1_pp": r(100 * (pe.f1_macro_mean - bme.f1_macro_mean), 2),
    "gain_over_matched_backbone_acc_pp": r(100 * (pe.accuracy_mean - bme.accuracy_mean), 2),
    "acc_min26": r(ex26s.accuracy_mean.min()), "acc_max26": r(ex26s.accuracy_mean.max()),
    "acc_spread26_pp": r(100 * (ex26s.accuracy_mean.max() - ex26s.accuracy_mean.min()), 1),
    "f1_min26": r(ex26s.f1_macro_mean.min()), "f1_max26": r(ex26s.f1_macro_mean.max()),
    "f1_spread26_pp": r(100 * (ex26s.f1_macro_mean.max() - ex26s.f1_macro_mean.min()), 1),
    "worst_model": ex.iloc[-1].model, "worst_f1": r(ex.iloc[-1].f1_macro_mean),
    "worst_acc": r(ex26s.accuracy_mean.min()),
    "top5_f1_models": ex.head(5).model.tolist(),
    "top5_f1_range_pp": r(100 * (ex.iloc[0].f1_macro_mean - ex.iloc[4].f1_macro_mean), 2),
    "leakage_min": r(ex.leakage_rate_mean.min(), 3),
    "leakage_min_model": ex.loc[ex.leakage_rate_mean.idxmin(), "model"],
    "leakage_max": r(ex.leakage_rate_mean.max(), 3),
    "leakage_max_model": ex.loc[ex.leakage_rate_mean.idxmax(), "model"],
    "leakage_median": r(ex.leakage_rate_mean.median(), 3),
    "leakage_top5": {m: r(ex.loc[ex.model == m, "leakage_rate_mean"].iloc[0], 3)
                     for m in ex.head(5).model},
    "p2_over_p1_ratio": r((ex.P2_f1_macro_mean / ex.P1_f1_macro_mean).mean(), 4),
    "spearman_p1_p3": r(stats.spearmanr(ex.P1_f1_macro_mean, ex.P3_f1_macro_mean)[0], 3),
    "spearman_p1_p3_p": r(stats.spearmanr(ex.P1_f1_macro_mean, ex.P3_f1_macro_mean)[1], 4),
    "p3_minus_p1_mean_pp": r(100 * (ex.P3_f1_macro_mean - ex.P1_f1_macro_mean).mean(), 2),
    "p3_minus_p1_min_pp": r(100 * (ex.P3_f1_macro_mean - ex.P1_f1_macro_mean).min(), 2),
    "p3_minus_p1_max_pp": r(100 * (ex.P3_f1_macro_mean - ex.P1_f1_macro_mean).max(), 2),
}

mg = ex26s.copy()
rho_f1, p_f1 = stats.spearmanr(mg.int_f1, mg.f1_macro_mean)
rho_acc, p_acc = stats.spearmanr(mg.int_f1, mg.accuracy_mean)
tau, p_tau = stats.kendalltau(mg.int_rank26, mg.rank26_f1)
N["transfer"] = {
    "spearman_intf1_extf1": r(rho_f1, 3), "spearman_intf1_extf1_p": r(p_f1, 4),
    "spearman_intf1_extacc": r(rho_acc, 3), "spearman_intf1_extacc_p": r(p_acc, 4),
    "kendall_tau": r(tau, 3), "kendall_p": r(p_tau, 4),
    "proposed_drop_acc_pp": r(pe.drop_acc_pp, 1), "proposed_drop_f1_pp": r(pe.drop_f1_pp, 1),
    "median_drop_acc_pp": r(mg.drop_acc_pp.median(), 1), "median_drop_f1_pp": r(mg.drop_f1_pp.median(), 1),
    "min_drop_acc_pp": r(mg.drop_acc_pp.min(), 1), "min_drop_acc_model": mg.loc[mg.drop_acc_pp.idxmin(), "model"],
    "min_drop_f1_pp": r(mg.drop_f1_pp.min(), 1), "min_drop_f1_model": mg.loc[mg.drop_f1_pp.idxmin(), "model"],
    "max_drop_acc_pp": r(mg.drop_acc_pp.max(), 1), "max_drop_model": mg.loc[mg.drop_acc_pp.idxmax(), "model"],
    "proposed_drop_rank26": int(mg.drop_acc_pp.rank().loc[mg.model == PROPOSED].iloc[0]),
    "vgg16_int_rank26": int(mg.loc[mg.model == "vgg16", "int_rank26"].iloc[0]),
    "vgg16_ext_rank26_f1": int(mg.loc[mg.model == "vgg16", "rank26_f1"].iloc[0]),
    "vgg19_int_rank26": int(mg.loc[mg.model == "vgg19", "int_rank26"].iloc[0]),
    "vgg19_ext_rank26_f1": int(mg.loc[mg.model == "vgg19", "rank26_f1"].iloc[0]),
    "vgg19_ext_acc": r(mg.loc[mg.model == "vgg19", "accuracy_mean"].iloc[0], 3),
    "vgg16_ext_acc": r(mg.loc[mg.model == "vgg16", "accuracy_mean"].iloc[0], 3),
    "smallest4_drop_min": r(mg.nsmallest(4, "params_M").drop_acc_pp.min(), 1),
    "smallest4_drop_max": r(mg.nsmallest(4, "params_M").drop_acc_pp.max(), 1),
    "vgg_drop_min": r(mg[mg.model.isin(["vgg16", "vgg19"])].drop_acc_pp.min(), 1),
    "vgg_drop_max": r(mg[mg.model.isin(["vgg16", "vgg19"])].drop_acc_pp.max(), 1),
}

def paired(a, b, split_a, split_b):
    x = a[(a.split == split_a) & (a.experiment.isin(GENERIC))][["cfg", "seed", "f1_macro", "accuracy"]]
    y = b[(b.split == split_b) & (b.experiment.isin(GENERIC))][["cfg", "seed", "f1_macro", "accuracy"]]
    m = x.merge(y, on=["cfg", "seed"], suffixes=("_A", "_B"))
    assert len(m) == 78, len(m)
    m["dF1_pp"] = 100 * (m.f1_macro_A - m.f1_macro_B)
    m["dAcc_pp"] = 100 * (m.accuracy_A - m.accuracy_B)
    return m

rep_int = paired(ma, mb, "internal_test", "internal_test")
rep_ext = paired(ma, mb, "external_mld24", "external")
rep_int.to_csv(OUT / "_replication_internal_pairs.csv", index=False)
rep_ext.to_csv(OUT / "_replication_external_pairs.csv", index=False)


def repstats(m, metric):
    d = m[f"d{metric}_pp"]
    g = m.groupby("cfg")[[f"{'f1_macro' if metric == 'F1' else 'accuracy'}_A",
                          f"{'f1_macro' if metric == 'F1' else 'accuracy'}_B"]].mean()
    g.columns = ["A", "B"]
    rho, pv = stats.spearmanr(g.A, g.B)
    tau_, pt = stats.kendalltau(g.A, g.B)
    g["rank_A"] = g.A.rank(ascending=False).astype(int)
    g["rank_B"] = g.B.rank(ascending=False).astype(int)
    g["rank_move"] = (g.rank_A - g.rank_B).abs()
    per_cfg = m.groupby("cfg")[f"d{metric}_pp"].apply(lambda s: s.abs().mean())
    return {
        "n_pairs": int(len(d)), "mean_pp": r(d.mean(), 3), "mean_abs_pp": r(d.abs().mean(), 3),
        "median_abs_pp": r(d.abs().median(), 3), "sd_pp": r(d.std(ddof=1), 3),
        "max_abs_pp": r(d.abs().max(), 3), "p90_abs_pp": r(d.abs().quantile(0.9), 3),
        "n_abs_gt_1pp": int((d.abs() > 1).sum()), "n_abs_gt_5pp": int((d.abs() > 5).sum()),
        "spearman_AB": r(rho, 3), "spearman_AB_p": r(pv, 5), "kendall_AB": r(tau_, 3),
        "spread_A_pp": r(100 * (g.A.max() - g.A.min()), 2), "spread_B_pp": r(100 * (g.B.max() - g.B.min()), 2),
        "max_rank_move": int(g.rank_move.max()), "max_rank_move_model": g.rank_move.idxmax(),
        "mean_rank_move": r(g.rank_move.mean(), 2),
        "n_rank_move_ge5": int((g.rank_move >= 5).sum()),
        "ranks": {k: {"A": int(v.rank_A), "B": int(v.rank_B), "A_val": r(v.A), "B_val": r(v.B)}
                  for k, v in g.iterrows()},
        "mean_abs_of_means_pp": r((100 * (g.A - g.B)).abs().mean(), 3),
        "median_abs_of_means_pp": r((100 * (g.A - g.B)).abs().median(), 3),
        "max_abs_of_means_pp": r((100 * (g.A - g.B)).abs().max(), 3),
        "max_abs_of_means_cfg": (100 * (g.A - g.B)).abs().idxmax(),
        "sd_abs_pp": r(d.abs().std(ddof=1), 3),
        "per_cfg_mean_abs_pp": {k: r(v, 2) for k, v in per_cfg.sort_values(ascending=False).items()},
        "worst_cfg": per_cfg.idxmax(), "worst_cfg_mean_abs_pp": r(per_cfg.max(), 2),
        "best_cfg": per_cfg.idxmin(), "best_cfg_mean_abs_pp": r(per_cfg.min(), 2),
    }

N["replication"] = {
    "internal_f1": repstats(rep_int, "F1"), "internal_acc": repstats(rep_int, "Acc"),
    "external_f1": repstats(rep_ext, "F1"), "external_acc": repstats(rep_ext, "Acc"),
}
ri, re_ = N["replication"]["internal_f1"], N["replication"]["external_f1"]
N["replication"]["spread_to_noise_internal"] = r(ri["spread_B_pp"] / ri["mean_abs_pp"], 1)
N["replication"]["spread_to_noise_external"] = r(re_["spread_B_pp"] / re_["mean_abs_pp"], 1)
N["replication"]["noise_ratio_ext_over_int"] = r(re_["mean_abs_pp"] / ri["mean_abs_pp"], 1)

trip = mb[mb.cfg.isin(["proposed/" + PROPOSED, "ablation/full", "repeats/proposed_rep1"])]
def trip_stats(split, metric):
    pv_ = trip[trip.split == split].pivot_table(index="seed", columns="cfg", values=metric)
    pv_ = pv_[["proposed/" + PROPOSED, "ablation/full", "repeats/proposed_rep1"]]
    vals = pv_.values
    diffs = []
    for i in range(3):
        for j in range(i + 1, 3):
            diffs += list(100 * (vals[:, i] - vals[:, j]))
    diffs = np.array(diffs)
    return {
        "per_seed": {int(s): [r(v) for v in row] for s, row in zip(pv_.index, vals)},
        "replicate_means": [r(v) for v in vals.mean(axis=0)],
        "replicate_sds": [r(v) for v in vals.std(axis=0, ddof=1)],
        "mean9": r(vals.mean()), "sd9": r(vals.std(ddof=1)), "min9": r(vals.min()), "max9": r(vals.max()),
        "per_seed_range_pp": [r(100 * (row.max() - row.min()), 2) for row in vals],
        "max_per_seed_range_pp": r(100 * (vals.max(axis=1) - vals.min(axis=1)).max(), 2),
        "pairwise_diffs_pp": [r(v, 2) for v in diffs],
        "mean_abs_pairwise_pp": r(np.abs(diffs).mean(), 2), "max_abs_pairwise_pp": r(np.abs(diffs).max(), 2),
        "range_of_replicate_means_pp": r(100 * (vals.mean(axis=0).max() - vals.mean(axis=0).min()), 2),
    }
N["replication"]["hmla_triplicate"] = {
    "internal_f1": trip_stats("internal_test", "f1_macro"), "internal_acc": trip_stats("internal_test", "accuracy"),
    "external_f1": trip_stats("external", "f1_macro"), "external_acc": trip_stats("external", "accuracy"),
}
cm2 = mb[mb.cfg.isin(["recipe_matched/" + BACKBONE, "repeats/caformer_matched_rep1"])]
def dup_stats(split, metric):
    pv_ = cm2[cm2.split == split].pivot_table(index="seed", columns="cfg", values=metric)
    v = pv_.values
    d = 100 * (v[:, 0] - v[:, 1])
    return {"replicate_means": [r(x) for x in v.mean(axis=0)], "mean6": r(v.mean()),
            "paired_diffs_pp": [r(x, 2) for x in d], "mean_abs_pp": r(np.abs(d).mean(), 2)}
N["replication"]["caformer_matched_duplicate"] = {
    "internal_f1": dup_stats("internal_test", "f1_macro"), "external_f1": dup_stats("external", "f1_macro"),
    "external_acc": dup_stats("external", "accuracy")}
sn = json.loads((B / "statistics" / "stats_report.json").read_text().replace("NaN", "null"))
N["replication"]["package_repeats_internal"] = sn["spread_vs_noise_internal"]

rc = pd.read_csv(B / "recipe_matched" / "recipe_matched_comparison.csv")
rc["model"] = rc.model.map(norm)
rows = []
for t in rc.itertuples():
    rows.append({"split": t.split, "model": t.model,
                 "standard": r(t.standard_mean), "standard_sd": r(t.standard_std),
                 "matched": r(t.matched_mean), "matched_sd": r(t.matched_std),
                 "recipe_effect_pp": r(100 * t.recipe_effect, 2),
                 "proposed": r(t.proposed_mean),
                 "architecture_effect_pp": r(100 * t.architecture_effect, 2)})
rcx = rc[rc.split == "external"]
rci = rc[rc.split == "internal_test"]
N["recipe_matched"] = {
    "models": sorted(rc.model.unique().tolist()), "rows": rows,
    "ext_recipe_effect_min_pp": r(100 * rcx.recipe_effect.min(), 2),
    "ext_recipe_effect_max_pp": r(100 * rcx.recipe_effect.max(), 2),
    "ext_recipe_effect_mean_pp": r(100 * rcx.recipe_effect.mean(), 2),
    "n_ext_recipe_effect_negative": int((rcx.recipe_effect < 0).sum()),
    "int_recipe_effect_min_pp": r(100 * rci.recipe_effect.min(), 2),
    "int_recipe_effect_max_pp": r(100 * rci.recipe_effect.max(), 2),
    "ext_arch_effect_vs_caformer_pp": r(100 * rcx[rcx.model == BACKBONE].architecture_effect.iloc[0], 2),
    "int_arch_effect_vs_caformer_pp": r(100 * rci[rci.model == BACKBONE].architecture_effect.iloc[0], 2),
    "ext_arch_effect_min_pp": r(100 * rcx.architecture_effect.min(), 2),
    "ext_arch_effect_min_model": rcx.loc[rcx.architecture_effect.idxmin(), "model"],
    "ext_arch_effect_max_pp": r(100 * rcx.architecture_effect.max(), 2),
    "ext_arch_effect_max_model": rcx.loc[rcx.architecture_effect.idxmax(), "model"],
    "matched_best_ext_model": rcx.loc[rcx.matched_mean.idxmax(), "model"],
    "matched_best_ext_f1": r(rcx.matched_mean.max()),
    "pooled_hmla_ext_f1": N["replication"]["hmla_triplicate"]["external_f1"]["mean9"],
    "pooled_caformer_matched_ext_f1": N["replication"]["caformer_matched_duplicate"]["external_f1"]["mean6"],
}
N["recipe_matched"]["pooled_arch_effect_pp"] = r(100 * (
    N["recipe_matched"]["pooled_hmla_ext_f1"] - N["recipe_matched"]["pooled_caformer_matched_ext_f1"]), 2)
rp = pd.read_csv(B / "recipe_matched" / "recipe_matched_paired.csv")
N["recipe_matched"]["n_significant_holm"] = int(rp.significant_holm.sum())
N["recipe_matched"]["n_tests"] = int(len(rp))

abi = pd.read_csv(B / "summary" / "ablation_summary_internal.csv").set_index("variant")
abe = pd.read_csv(B / "summary" / "ablation_summary_external.csv").set_index("variant")
abA = pd.read_csv(MAN / "submitted_version" / "_ablation_both.csv").set_index("variant")
ext_acc_B = mb_ext[mb_ext.experiment == "ablation"].groupby("cfg").accuracy.agg(["mean", "std"])
ext_acc_B.index = [c.split("/")[1] for c in ext_acc_B.index]
ref_int = N["replication"]["hmla_triplicate"]["internal_f1"]["mean9"]
ref_ext = N["replication"]["hmla_triplicate"]["external_f1"]["mean9"]
ref_ext_acc = N["replication"]["hmla_triplicate"]["external_acc"]["mean9"]
GROUP = {"wo_msf": "architecture", "wo_lap": "architecture", "wo_lap_gate": "architecture",
         "wo_hier": "architecture", "wo_mixstyle": "architecture"}
rows = []
for v in abi.index:
    if v == "full":
        continue
    rows.append({
        "variant": v, "group": GROUP.get(v, "recipe"),
        "int_f1": r(abi.loc[v, "f1_macro_mean"]), "int_sd": r(abi.loc[v, "f1_macro_std"]),
        "ext_f1": r(abe.loc[v, "f1_macro_mean"]), "ext_sd": r(abe.loc[v, "f1_macro_std"]),
        "ext_acc": r(ext_acc_B.loc[v, "mean"]),
        "d_int_vs_full_pp": r(100 * (abi.loc["full", "f1_macro_mean"] - abi.loc[v, "f1_macro_mean"]), 2),
        "d_ext_vs_full_pp": r(100 * (abe.loc["full", "f1_macro_mean"] - abe.loc[v, "f1_macro_mean"]), 2),
        "d_int_vs_pooled_pp": r(100 * (ref_int - abi.loc[v, "f1_macro_mean"]), 2),
        "d_ext_vs_pooled_pp": r(100 * (ref_ext - abe.loc[v, "f1_macro_mean"]), 2),
        "d_ext_acc_vs_pooled_pp": r(100 * (ref_ext_acc - ext_acc_B.loc[v, "mean"]), 2),
        "A_int_f1": r(abA.loc[v, "f1_macro_mean_int"]), "A_ext_f1": r(abA.loc[v, "f1_macro_mean_ext"]),
        "A_d_int_pp": r(abA.loc[v, "delta_int_pp"], 2), "A_d_ext_pp": r(abA.loc[v, "delta_ext_pp"], 2),
        "n_seeds_int": int(abi.loc[v, "n_seeds"]), "n_seeds_ext": int(abe.loc[v, "n_seeds"]),
        "wilcoxon_p_int": r(abi.loc[v, "wilcoxon_p"], 3), "wilcoxon_p_ext": r(abe.loc[v, "wilcoxon_p"], 3),
    })
abdf = pd.DataFrame(rows)
abdf.to_csv(OUT / "_ablation_AB.csv", index=False)
sub = abdf[abdf.variant != "wo_pretrained"]
N["ablation"] = {
    "n_variants": 17, "n_removals": 16, "n_arch": 5, "n_recipe": 11,
    "full_B_int_f1": r(abi.loc["full", "f1_macro_mean"]), "full_B_int_sd": r(abi.loc["full", "f1_macro_std"]),
    "full_B_ext_f1": r(abe.loc["full", "f1_macro_mean"]), "full_B_ext_sd": r(abe.loc["full", "f1_macro_std"]),
    "full_B_ext_acc": r(ext_acc_B.loc["full", "mean"]),
    "full_A_int_f1": r(abA.loc["full", "f1_macro_mean_int"]), "full_A_ext_f1": r(abA.loc["full", "f1_macro_mean_ext"]),
    "ref_pooled_int_f1": r(ref_int), "ref_pooled_ext_f1": r(ref_ext), "ref_pooled_ext_acc": r(ref_ext_acc),
    "rows": rows,
    "max_abs_d_int_pooled_pp": r(sub.d_int_vs_pooled_pp.abs().max(), 2),
    "max_abs_d_int_full_pp": r(sub.d_int_vs_full_pp.abs().max(), 2),
    "max_abs_d_ext_pooled_pp": r(sub.d_ext_vs_pooled_pp.abs().max(), 2),
    "max_d_ext_pooled_pp": r(sub.d_ext_vs_pooled_pp.max(), 2),
    "max_d_ext_pooled_variant": sub.loc[sub.d_ext_vs_pooled_pp.idxmax(), "variant"],
    "min_d_ext_pooled_pp": r(sub.d_ext_vs_pooled_pp.min(), 2),
    "min_d_ext_pooled_variant": sub.loc[sub.d_ext_vs_pooled_pp.idxmin(), "variant"],
    "max_abs_d_ext_full_pp": r(sub.d_ext_vs_full_pp.abs().max(), 2),
    "n_int_negative_pooled": int((sub.d_int_vs_pooled_pp < 0).sum()),
    "n_int_negative_full": int((sub.d_int_vs_full_pp < 0).sum()),
    "n_ext_positive_pooled": int((sub.d_ext_vs_pooled_pp > 0).sum()),
    "n_ext_positive_full": int((sub.d_ext_vs_full_pp > 0).sum()),
    "n_ext_positive_A": int((sub.A_d_ext_pp > 0).sum()),
    "A_max_d_ext_pp": r(sub.A_d_ext_pp.max(), 2), "A_max_d_ext_variant": sub.loc[sub.A_d_ext_pp.idxmax(), "variant"],
    "spearman_AB_ext_delta": r(stats.spearmanr(sub.A_d_ext_pp, sub.d_ext_vs_pooled_pp)[0], 3),
    "spearman_AB_ext_delta_p": r(stats.spearmanr(sub.A_d_ext_pp, sub.d_ext_vs_pooled_pp)[1], 3),
    "spearman_AB_ext_delta_fullref": r(stats.spearmanr(sub.A_d_ext_pp, sub.d_ext_vs_full_pp)[0], 3),
    "spearman_AB_int_delta": r(stats.spearmanr(sub.A_d_int_pp, sub.d_int_vs_pooled_pp)[0], 3),
    "spearman_AB_int_delta_p": r(stats.spearmanr(sub.A_d_int_pp, sub.d_int_vs_pooled_pp)[1], 3),
    "spearman_AB_ext_f1": r(stats.spearmanr(sub.A_ext_f1, sub.ext_f1)[0], 3),
    "spearman_int_ext_B": r(stats.spearmanr(sub.d_int_vs_pooled_pp, sub.d_ext_vs_pooled_pp)[0], 3),
    "spearman_int_ext_B_p": r(stats.spearmanr(sub.d_int_vs_pooled_pp, sub.d_ext_vs_pooled_pp)[1], 3),
    "pretrain_d_int_pooled_pp": r(abdf.loc[abdf.variant == "wo_pretrained", "d_int_vs_pooled_pp"].iloc[0], 2),
    "pretrain_d_ext_pooled_pp": r(abdf.loc[abdf.variant == "wo_pretrained", "d_ext_vs_pooled_pp"].iloc[0], 2),
    "pretrain_A_d_ext_pp": r(abdf.loc[abdf.variant == "wo_pretrained", "A_d_ext_pp"].iloc[0], 2),
    "pretrain_A_d_int_pp": r(abdf.loc[abdf.variant == "wo_pretrained", "A_d_int_pp"].iloc[0], 2),
    "n_sign_agree_ext": int(((sub.A_d_ext_pp > 0) == (sub.d_ext_vs_pooled_pp > 0)).sum()),
    "n_sign_agree_int": int(((sub.A_d_int_pp > 0) == (sub.d_int_vs_pooled_pp > 0)).sum()),
    "n_significant_holm_int": int(abi.significant_holm.sum()), "n_significant_holm_ext": int(abe.significant_holm.sum()),
    "cooldown_seeds_A": 2, "cooldown_seeds_B": int(abi.loc["wo_mixup_cooldown", "n_seeds"]),
}
N["ablation"]["A_headline"] = {v: r(abA.loc[v, "delta_ext_pp"], 2) for v in
                              ["wo_llrd", "wo_mixup", "wo_mixstyle", "wo_label_smoothing", "wo_tta", "wo_lap"]}
N["ablation"]["B_same_variants_pooled"] = {v: r(abdf.set_index("variant").loc[v, "d_ext_vs_pooled_pp"], 2) for v in
                                           ["wo_llrd", "wo_mixup", "wo_mixstyle", "wo_label_smoothing", "wo_tta", "wo_lap"]}

SEVEN = ["wo_llrd", "wo_mixup", "wo_mixstyle", "wo_label_smoothing", "wo_tta", "wo_mixup_cooldown", "wo_lap"]
_abx = abdf.set_index("variant")
N["ablation"]["B_full_same_variants7"] = {v: r(_abx.loc[v, "d_ext_vs_full_pp"], 2) for v in SEVEN}
N["ablation"]["B_pooled_same_variants7"] = {v: r(_abx.loc[v, "d_ext_vs_pooled_pp"], 2) for v in SEVEN}
N["ablation"]["A_same_variants7"] = {v: r(_abx.loc[v, "A_d_ext_pp"], 2) for v in SEVEN}
N["ablation"]["pooled_minus_full_ext_pp"] = r(100 * (ref_ext - abe.loc["full", "f1_macro_mean"]), 2)
N["ablation"]["n_sign_agree_ext_full"] = int(((sub.A_d_ext_pp > 0) == (sub.d_ext_vs_full_pp > 0)).sum())
N["ablation"]["spearman_AB_ext_delta_full"] = r(stats.spearmanr(sub.A_d_ext_pp, sub.d_ext_vs_full_pp)[0], 3)
N["ablation"]["A_n_ext_gt_mean_rep"] = int((sub.A_d_ext_pp > N["replication"]["external_f1"]["mean_abs_pp"]).sum())
N["ablation"]["A_max_d_ext_pp"] = r(sub.A_d_ext_pp.max(), 2)
abdf[["variant", "group", "A_d_int_pp", "A_d_ext_pp", "d_int_vs_full_pp", "d_int_vs_pooled_pp",
      "d_ext_vs_full_pp", "d_ext_vs_pooled_pp", "n_seeds_int", "n_seeds_ext"]].rename(columns={
          "d_int_vs_full_pp": "B_d_int_vs_full_pp", "d_int_vs_pooled_pp": "B_d_int_vs_pooled_pp",
          "d_ext_vs_full_pp": "B_d_ext_vs_full_pp", "d_ext_vs_pooled_pp": "B_d_ext_vs_pooled_pp"}).to_csv(
    OUT / "_ablation_reference_sensitivity.csv", index=False)

fn, fnt = sn["friedman_nemenyi"], sn["friedman_nemenyi_top"]
pw = sn["statistical_power"]
mc = sn["mcnemar"]
pvb = pd.read_csv(B / "statistics" / "proposed_vs_baselines.csv")
N["stats"] = {
    "friedman_all": {"chi2": r(fn["friedman_stat"], 2), "p": r(fn["friedman_p"], 4), "k": fn["k_models"],
                     "n_blocks": fn["n_blocks"], "cd": r(fn["nemenyi_cd"], 2),
                     "proposed_avg_rank": r(fn["avg_ranks"][PROPOSED], 2),
                     "best_avg_rank": r(min(fn["avg_ranks"].values()), 2),
                     "worst_avg_rank": r(max(fn["avg_ranks"].values()), 2),
                     "rank_range": r(max(fn["avg_ranks"].values()) - min(fn["avg_ranks"].values()), 2),
                     "avg_ranks": {norm(k): r(v, 2) for k, v in fn["avg_ranks"].items()}},
    "friedman_top": {"chi2": r(fnt["friedman_stat"], 2), "p": r(fnt["friedman_p"], 3), "k": fnt["k_models"],
                     "cd": r(fnt["nemenyi_cd"], 2), "models": [norm(m) for m in fnt["model_names"]]},
    "power": {k: pw[k] for k in ["n_blocks", "k_models", "alpha", "wilcoxon_min_attainable_p",
                                 "min_blocks_for_alpha", "holm_threshold_for_most_significant",
                                 "nemenyi_cd", "rank_axis_width", "cd_as_fraction_of_rank_axis"]},
    "mcnemar": {"best_baseline": mc["best_baseline"],
                "per_seed": [{"seed": s["seed"], "n01": s["n01"], "n10": s["n10"], "p": r(s["p_value"], 3),
                              "acc": r(s["proposed_acc_ci95"][0]), "ci_lo": r(s["proposed_acc_ci95"][1]),
                              "ci_hi": r(s["proposed_acc_ci95"][2])} for s in mc["per_seed"]]},
    "wilcoxon": {"n_baselines": int(len(pvb)), "n_significant_holm": int(pvb.significant_holm.sum()),
                 "min_possible_p": 0.25, "min_raw_p": r(pvb.wilcoxon_p.min(), 3),
                 "n_at_floor": int((pvb.wilcoxon_p <= 0.25).sum()),
                 "n_baselines_above_proposed": int((pvb.mean_diff < 0).sum())},
}
N["stats"]["power"]["holm_blocks_needed_25"] = 10
N["stats"]["power"]["holm_blocks_needed_30"] = 10
import math
N["stats"]["power"]["holm_blocks_needed_30"] = int(math.ceil(1 - math.log2(0.05 / 30)))
N["stats"]["power"]["holm_blocks_needed_25"] = int(math.ceil(1 - math.log2(0.05 / 25)))

cx = pd.read_csv(B / "summary" / "complexity.csv")
cx["model"] = cx.model.map(norm)
pc = cx[cx.model == PROPOSED].iloc[0]
top5 = ex.head(5).model.tolist()
N["complexity"] = {
    "proposed_params_M": r(pc.params_M, 1), "proposed_gflops": r(pc.gflops, 2), "proposed_size_MB": r(pc.size_MB, 1),
    "proposed_latency_tta_ms": r(pc.latency_ms, 1), "proposed_latency_single_ms": r(pc.latency_ms_single_pass, 1),
    "proposed_fps": r(pc.fps, 1), "proposed_train_sec_mean": r(pc.train_sec_mean, 0),
    "backbone_params_M": r(cx[cx.model == BACKBONE].params_M.iloc[0], 1),
    "backbone_latency_ms": r(cx[cx.model == BACKBONE].latency_ms.iloc[0], 1),
    "top5_ext_params": {m: r(cx[cx.model == m].params_M.iloc[0], 1) for m in top5},
    "top5_ext_gflops": {m: r(cx[cx.model == m].gflops.iloc[0], 2) for m in top5},
    "top5_ext_latency": {m: r(cx[cx.model == m].latency_ms.iloc[0], 1) for m in top5},
    "smallest_model": cx.loc[cx.params_M.idxmin(), "model"], "smallest_params_M": r(cx.params_M.min(), 2),
    "largest_model": cx.loc[cx.params_M.idxmax(), "model"], "largest_params_M": r(cx.params_M.max(), 1),
}
tsec = mb_int.drop_duplicates("tag").train_sec.sum()
tsecA = ma_int.drop_duplicates("tag").train_sec.sum()
N["complexity"]["gpu_hours_B"] = r(tsec / 3600, 1)
N["complexity"]["gpu_hours_A"] = r(tsecA / 3600, 1)
N["complexity"]["n_runs_B"] = int(mb_int.tag.nunique())
N["complexity"]["n_runs_A"] = int(ma_int.tag.nunique())

pa = json.loads((B / "parameter_analysis" / "param_analysis.json").read_text())
def pick(d):
    return {k: d[k] for k in ["n_models", "spearman_rho", "spearman_p", "pearson_r_log10params", "pearson_p",
                              "linear_R2", "quadratic_R2", "quadratic_vs_linear_p", "inverted_U",
                              "optimum_params_M", "optimum_params_M_ci95", "share_bootstrap_inverted_U",
                              "kruskal_H", "kruskal_p"] if k in d} | {
        "bins": [{"bin": b["bin"], "n": b["n_models"], "mean": r(b["score_mean"]), "sd": r(b["score_std"]),
                  "min": r(b["score_min"]), "max": r(b["score_max"])} for b in d["bins"]],
        "pairwise": {k: {"diff": r(v["mean_diff"], 4), "p": r(v["p"], 4)} for k, v in d["pairwise_bins"].items()}}
N["params"] = {k: pick(pa[k]) for k in ["external__f1_macro", "external__accuracy",
                                         "internal_test__f1_macro", "internal_test__accuracy"]}

tt = json.loads((B / "tta_symmetry" / "tta_symmetry.json").read_text())
N["tta"] = {"run": tt["run_dir"].replace("\\", "/"),
            "rows": tt["performance"],
            "invariance": {d["group"]: {"orbit_dev": d["orbit_mean_max_abs_deviation"],
                                        "raw_dev": r(d["raw_model_max_abs_deviation"], 4)}
                           for d in tt["invariance_proof"]}}
perf = {d["group"]: d for d in tt["performance"]}
N["tta"]["ext_f1_gain_klein_pp"] = r(100 * (perf["klein"]["external_f1_macro"] - perf["none"]["external_f1_macro"]), 2)
N["tta"]["ext_f1_gain_d4_pp"] = r(100 * (perf["dihedral8"]["external_f1_macro"] - perf["none"]["external_f1_macro"]), 2)
N["tta"]["ext_acc_gain_klein_pp"] = r(100 * (perf["klein"]["external_accuracy"] - perf["none"]["external_accuracy"]), 2)
N["tta"]["ext_acc_gain_d4_pp"] = r(100 * (perf["dihedral8"]["external_accuracy"] - perf["none"]["external_accuracy"]), 2)
N["tta"]["int_f1_change_klein_pp"] = r(100 * (perf["klein"]["internal_f1_macro"] - perf["none"]["internal_f1_macro"]), 2)
N["tta"]["latency_ratio_klein"] = r(perf["klein"]["latency_ms"] / perf["none"]["latency_ms"], 1)
N["tta"]["latency_ratio_d4"] = r(perf["dihedral8"]["latency_ms"] / perf["none"]["latency_ms"], 1)

dr = json.loads((B / "duplicate_audit" / "duplicate_report.json").read_text())
cc = dr["cross_corpus"]
wi = json.loads((B / "duplicate_audit" / "within_internal.json").read_text())
sl = pd.read_csv(B / "duplicate_audit" / "split_leakage.csv")
exm = pd.read_csv(B / "duplicate_audit" / "cross_corpus_examples.csv")
ph = exm[exm.method == "perceptual_phash"].copy()
def ext_cls(pth):
    m = re.search(r"MLD24[\\/]([^\\/]+)[\\/]", pth); return m.group(1).lower().replace("_", " ")
def int_cls(pth):
    m = re.search(r"multiple leaf dataset[\\/]([^\\/]+)[\\/]([^\\/]+)[\\/]", pth); return m.group(2).lower()
ph["e"] = ph.external_image.map(ext_cls); ph["i"] = ph.internal_image.map(int_cls)
ph["same_class"] = ph.apply(lambda t: t.i == "mango " + t.e, axis=1)
ph["same_crop"] = ph.i.str.startswith("mango")
hist = {int(k): v for k, v in cc["perceptual_phash"]["min_distance_histogram"].items()}
emb = dr.get("cross_corpus", {}).get("embedding_cosine", {})
N["duplicates"] = {
    "internal_n": dr["internal"]["n_images"], "internal_md5_removed": dr["internal"]["n_md5_duplicates_removed_at_index"],
    "external_n": 4800,
    "cross_exact_md5": cc["exact_md5"]["n_exact_matches"],
    "cross_phash_le6": cc["perceptual_phash"]["n_near_duplicates"],
    "cross_phash_rate_pct": r(100 * cc["perceptual_phash"]["rate_of_b"], 2),
    "cross_phash_min_dist_mean": r(cc["perceptual_phash"]["min_distance_mean"], 2),
    "cross_phash_hist": hist,
    "cross_phash_n_le4": int(sum(v for k, v in hist.items() if k <= 4)),
    "cross_phash_n_le5": int(sum(v for k, v in hist.items() if k <= 5)),
    "cross_phash_n_eq6": int(hist.get(6, 0)),
    "cross_dhash": 113,
    "cross_embedding_ge95": 1, "cross_embedding_max": 0.9516, "cross_embedding_p99": 0.9018,
    "phash_examples_n": int(len(ph)), "phash_examples_same_class": int(ph.same_class.sum()),
    "phash_examples_same_crop_other_class": int((ph.same_crop & ~ph.same_class).sum()),
    "phash_examples_other_crop": int((~ph.same_crop).sum()),
    "phash_examples_hub_images": int(ph.internal_image.nunique()),
    "within_phash_pairs": wi["n_near_duplicate_pairs"], "within_cross_class_pairs": wi["n_cross_group_pairs"],
    "within_same_class_pairs": wi["n_within_group_pairs"],
    "within_possible_pairs": int(7179 * 7178 / 2),
    "within_cross_split_pairs": {int(t.seed): int(t.n_cross_split_pairs) for t in sl.itertuples()},
    "within_test_train_pairs": {int(t.seed): int(t["pairs_test|train"]) for _, t in sl.iterrows()},
    "within_cross_split_frac_mean": r(sl.n_cross_split_pairs.mean() / wi["n_near_duplicate_pairs"], 3),
    "chance_cross_split_frac": r(1 - (0.7 ** 2 + 0.15 ** 2 + 0.15 ** 2), 3),
}

fr = json.loads((B / "class_forensics" / "forensics_report.json").read_text())
zs = pd.read_csv(B / "class_forensics" / "zero_recall_scan.csv")
zsb = zs[zs.experiment.isin(["benchmark", "proposed"])]
piv = zsb.pivot_table(index="model", columns="seed", values="n_zero_recall_classes")
pdst = pd.read_csv(B / "class_forensics" / "prediction_destinations.csv")
pdp = pdst[pdst.model == PROPOSED].sort_values("share", ascending=False)
dsc = pd.read_csv(B / "class_forensics" / "domain_shift_by_class.csv").set_index("class")
sooty = dsc.loc["mango sooty mould"]
N["forensics"] = {
    "n_models_scanned": fr["n_models_scanned"], "n_models_any_zero": fr["n_models_with_zero_recall_class"],
    "n_model_seed_zero": int(fr["zero_recall_class_counts"]["mango sooty mould"]),
    "zero_class_is_always_sooty": list(fr["zero_recall_class_counts"].keys()) == ["mango sooty mould"],
    "n_models_all_seeds_zero": int((piv.min(axis=1) > 0).sum()),
    "n_models_never_zero": int((piv.max(axis=1) == 0).sum()),
    "never_zero_models": [norm(m) for m in piv.index[piv.max(axis=1) == 0]],
    "proposed_zero_seeds": int(piv.loc[PROPOSED].sum()),
    "proposed_sooty_recall": r(pdp[pdp.predicted_class == "mango sooty mould"].share.iloc[0], 3),
    "proposed_sooty_destinations": [{"class": t.predicted_class, "share": r(t.share, 3)} for t in pdp.head(6).itertuples()],
    "proposed_sooty_to_powdery": r(pdp[pdp.predicted_class == "mango powdery mildew"].share.iloc[0], 3),
    "proposed_sooty_to_gallmidge": r(pdp[pdp.predicted_class == "mango gall midge"].share.iloc[0], 3),
    "proposed_sooty_to_jack_sooty": r(pdp[pdp.predicted_class == "jackfruit sooty mold"].share.iloc[0], 3),
    "proposed_sooty_off_mango": r(pdp[~pdp.predicted_class.str.startswith("mango")].share.sum(), 3),
    "insect_confusion_share": r(fr["insect_class_confusion_share"], 3),
    "sooty_descriptors": {
        "brightness_int": r(sooty.mean_brightness_internal, 3), "brightness_ext": r(sooty.mean_brightness_external, 3),
        "brightness_d": r(sooty.mean_brightness_cohens_d, 2),
        "saturation_int": r(sooty.mean_saturation_internal, 3), "saturation_ext": r(sooty.mean_saturation_external, 3),
        "saturation_d": r(sooty.mean_saturation_cohens_d, 2),
        "sharpness_int": r(sooty.sharpness_lapvar_internal, 5), "sharpness_ext": r(sooty.sharpness_lapvar_external, 5),
        "sharpness_d": r(sooty.sharpness_lapvar_cohens_d, 2),
        "dark_frac_int": r(sooty.dark_pixel_fraction_internal, 3), "dark_frac_ext": r(sooty.dark_pixel_fraction_external, 3),
        "dark_frac_d": r(sooty.dark_pixel_fraction_cohens_d, 2),
        "greenness_d": r(sooty.greenness_cohens_d, 2),
    },
    "sharpness_d_all_classes": {c: r(dsc.loc[c, "sharpness_lapvar_cohens_d"], 2) for c in SHARED},
    "brightness_d_all_classes": {c: r(dsc.loc[c, "mean_brightness_cohens_d"], 2) for c in SHARED},
    "dark_frac_d_all_classes": {c: r(dsc.loc[c, "dark_pixel_fraction_cohens_d"], 2) for c in SHARED},
    "n_descriptor_images_per_class": 150,
    "nn_label_transfer_own_class": 0.40, "centroid_cosine": 0.881, "mmd2": 0.1868,
    "internal_sooty_n": 209,
}
N["forensics"]["sharpness_d_min"] = r(min(N["forensics"]["sharpness_d_all_classes"].values()), 2)
N["forensics"]["sharpness_d_max"] = r(max(N["forensics"]["sharpness_d_all_classes"].values()), 2)

RUNS_B = B / "runs"
B_AVAILABLE = (RUNS_B / "proposed" / PROPOSED / "seed42" / "per_class_external.csv").exists()


def per_class_tables(base, ext_name, test_name):
    per, pi = [], []
    for s in (42, 1234, 2024):
        d = pd.read_csv(base / f"seed{s}" / ext_name); d = d[d.support > 0].copy(); d["seed"] = s; per.append(d)
        t = pd.read_csv(base / f"seed{s}" / test_name); t["seed"] = s; pi.append(t)
    per, pi = pd.concat(per), pd.concat(pi)
    g = per.groupby("class").agg(prec=("precision", "mean"), prec_sd=("precision", "std"),
                                 rec=("recall_sensitivity", "mean"), rec_sd=("recall_sensitivity", "std"),
                                 f1=("f1-score", "mean"), f1_sd=("f1-score", "std"),
                                 support=("support", "mean")).reset_index().sort_values("f1", ascending=False)
    gi = pi.groupby("class").agg(prec=("precision", "mean"), rec=("recall_sensitivity", "mean"),
                                 f1=("f1-score", "mean"), support=("support", "mean")).reset_index()
    sooty = per[per["class"] == "mango sooty mould"].sort_values("seed").recall_sensitivity.tolist()
    return g, gi, sooty


def confusion(base, npz_name):
    cms = []
    for s in (42, 1234, 2024):
        z = np.load(base / f"seed{s}" / npz_name)
        cm = np.zeros((21, 21))
        for yt, yp in zip(z["y_true"], z["y_pred"]):
            cm[yt, yp] += 1
        cms.append(cm / cm.sum(axis=1, keepdims=True).clip(min=1))
    cm = np.mean(cms, axis=0)
    idx = [CLASSES.index(c) for c in SHARED]
    cm6 = cm[idx]
    cols_used = [j for j in range(21) if cm6[:, j].max() >= 0.01]
    return cm6, pd.DataFrame(cm6[:, cols_used], index=SHARED, columns=[CLASSES[j] for j in cols_used])


gA, giA, sootyA = per_class_tables(ARCH / "proposed" / PROPOSED, "per_class_external_mld24.csv", "per_class_test.csv")
cm6A, cmA = confusion(ARCH / "proposed" / PROPOSED, "predictions_external_mld24.npz")
gA.to_csv(OUT / "_per_class_external_A.csv", index=False); giA.to_csv(OUT / "_per_class_internal_A.csv", index=False)
cmA.to_csv(OUT / "_confusion_external_A.csv")
if B_AVAILABLE:
    g, gi, sooty_rec = per_class_tables(RUNS_B / "proposed" / PROPOSED, "per_class_external.csv", "per_class_test.csv")
    cm6, cmB = confusion(RUNS_B / "proposed" / PROPOSED, "predictions_external.npz")
    g.to_csv(OUT / "_per_class_external_B.csv", index=False); gi.to_csv(OUT / "_per_class_internal_B.csv", index=False)
    cmB.to_csv(OUT / "_confusion_external_B.csv")
    src = "B"
else:
    g, gi, sooty_rec, cm6 = gA, giA, sootyA, cm6A
    src = "A"
gi6 = gi[gi["class"].isin(SHARED)].set_index("class")
N["per_class"] = {
    "source": src, "B_available": bool(B_AVAILABLE),
    "external": [{"class": t._1, "precision": r(t.prec), "recall": r(t.rec), "f1": r(t.f1), "f1_sd": r(t.f1_sd),
                  "support": int(t.support)} for t in g.itertuples()],
    "internal6": {c: {"precision": r(gi6.loc[c, "prec"]), "recall": r(gi6.loc[c, "rec"]), "f1": r(gi6.loc[c, "f1"]),
                      "support": r(gi6.loc[c, "support"], 1)} for c in SHARED},
    "internal_n_perfect_of_6": int((gi6.f1 >= 0.9995).sum()),
    "internal_min_f1_6": r(gi6.f1.min()), "internal_min_f1_class_6": gi6.f1.idxmin(),
    "internal_n_perfect_of_21": int((gi.f1 >= 0.9995).sum()), "internal_min_f1_21": r(gi.f1.min()),
    "internal_min_f1_class_21": gi.loc[gi.f1.idxmin(), "class"],
    "sooty_recall_per_seed": [r(v, 4) for v in sooty_rec], "sooty_recall_mean": r(float(np.mean(sooty_rec)), 4),
    "sooty_recall_per_seed_A": [r(v, 4) for v in sootyA],
    "sooty_f1": r(g[g["class"] == "mango sooty mould"].f1.iloc[0], 3),
    "sooty_f1_A": r(gA[gA["class"] == "mango sooty mould"].f1.iloc[0], 3),
    "other5_mean_recall": r(g[g["class"] != "mango sooty mould"].rec.mean(), 3),
    "other5_mean_f1": r(g[g["class"] != "mango sooty mould"].f1.mean(), 3),
    "other5_mean_f1_A": r(gA[gA["class"] != "mango sooty mould"].f1.mean(), 3),
    "sooty_to_powdery": r(cm6[SHARED.index("mango sooty mould"), CLASSES.index("mango powdery mildew")], 3),
    "sooty_to_jack_sooty": r(cm6[SHARED.index("mango sooty mould"), CLASSES.index("jackfruit sooty mold")], 3),
    "sooty_to_gallmidge": r(cm6[SHARED.index("mango sooty mould"), CLASSES.index("mango gall midge")], 3),
    "sooty_off_mango": r(float(sum(cm6[SHARED.index("mango sooty mould"), j] for j in range(21) if not CLASSES[j].startswith("mango"))), 3),
    "sooty_to_powdery_A": r(cm6A[SHARED.index("mango sooty mould"), CLASSES.index("mango powdery mildew")], 3),
    "sooty_to_jack_sooty_A": r(cm6A[SHARED.index("mango sooty mould"), CLASSES.index("jackfruit sooty mold")], 3),
    "sooty_to_gallmidge_A": r(cm6A[SHARED.index("mango sooty mould"), CLASSES.index("mango gall midge")], 3),
    "powdery_precision": r(g[g["class"] == "mango powdery mildew"].prec.iloc[0], 3),
    "powdery_recall": r(g[g["class"] == "mango powdery mildew"].rec.iloc[0], 3),
    "powdery_f1": r(g[g["class"] == "mango powdery mildew"].f1.iloc[0], 3),
    "external_by_class": {t._1: {"f1": r(t.f1), "rec": r(t.rec), "prec": r(t.prec)} for t in g.itertuples()},
}
N["forensics"]["proposed_sooty_recall"] = N["per_class"]["sooty_recall_mean"]

ct = OUT / "_clean_test_summary.json"
N["clean_test"] = json.loads(ct.read_text()) if ct.exists() else None

N["calibration"] = {"int_ece": r(p.ece_mean, 3), "ext_ece": r(pe.ece_mean, 3), "ext_ece_sd": r(pe.ece_std, 3),
                    "int_ece_best": r(it.ece_mean.min(), 3), "int_ece_best_model": it.loc[it.ece_mean.idxmin(), "model"]}
N["environment"] = json.loads((B / "summary" / "environment.json").read_text())
selA = json.loads((ARCH / "proposed" / PROPOSED / "seed1234" / "selection.json").read_text())
N["beta"] = {"selected_all_seeds_B": [float(v) for v in mb_int[mb_int.cfg == "proposed/" + PROPOSED].sort_values("seed").hier_beta],
             "grid": [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0],
             "grid_scores_identical_A_seed1234": len(set(round(v, 9) for v in selA["hier_beta_scores"].values())) == 1}
N["timing"] = {"B_first_run": "2026-08-31", "A_first_run": "2026-08-06"}

NA = json.loads((MAN / "submitted_version" / "numbers.json").read_text())
XA = json.loads((OUT / "_A_extra.json").read_text())
itA = pd.read_csv(A / "summary" / "summary_internal_test.csv"); itA["model"] = itA.model.map(norm)
itA = itA.sort_values("f1_macro_mean", ascending=False).reset_index(drop=True); itA["rank_int"] = np.arange(1, len(itA) + 1)
exA = pd.read_csv(A / "summary" / "summary_external_mld24.csv"); exA["model"] = exA.model.map(norm)
exA = exA[exA.experiment.isin(["benchmark", "proposed"])].copy()
prA = pd.read_csv(OUT / "_A_protocols.csv"); prA = prA[prA.experiment.isin(["benchmark", "proposed"])]
exA = exA.merge(prA[["model", "P3_f1", "P3_acc", "leakage", "gmean"]], on="model").merge(
    itA[["model", "rank_int", "f1_macro_mean", "accuracy_mean", "params_M", "gflops"]].rename(
        columns={"f1_macro_mean": "int_f1", "accuracy_mean": "int_acc"}), on="model")
exA["drop_acc_pp"] = 100 * (exA.int_acc - exA.accuracy_mean)
exA = exA.sort_values("accuracy_mean", ascending=False).reset_index(drop=True); exA["rank_acc"] = np.arange(1, len(exA) + 1)
exA["rank_f1"] = exA.f1_macro_mean.rank(ascending=False, method="min").astype(int)
assert len(exA) == 26
itA.to_csv(OUT / "_internal_A.csv", index=False); exA.to_csv(OUT / "_external_A.csv", index=False)
ctA = pd.read_csv(OUT / "_A_clean_test_summary.csv")
N["A"] = {k: NA[k] for k in ["internal", "external", "transfer", "noise_floor", "stats", "complexity", "calibration", "per_class_internal"]}
N["A"]["per_class_external"] = NA["per_class_external"]
N["A"]["ablation"] = NA["ablation"]
N["A"]["zero_recall"] = XA["zero_recall"]
N["A"]["protocols"] = XA["protocols"]
N["A"]["clean_test"] = XA["clean_test"]
N["A"]["params"] = XA["params"]
N["A"]["n_within_mean_band"] = int((itA.f1_macro_mean >= itA.f1_macro_mean.max() - ri["mean_abs_pp"] / 100).sum())
N["A"]["n_within_max_band"] = int((itA.f1_macro_mean >= itA.f1_macro_mean.max() - ri["max_abs_pp"] / 100).sum())
N["A"]["ext_second_acc_model"] = exA.iloc[1].model; N["A"]["ext_second_acc"] = r(exA.iloc[1].accuracy_mean)
N["A"]["ext_best_f1_model"] = exA.loc[exA.f1_macro_mean.idxmax(), "model"]; N["A"]["ext_best_f1"] = r(exA.f1_macro_mean.max())
N["A"]["ext_top5_f1_range_pp"] = r(100 * (exA.f1_macro_mean.nlargest(5).max() - exA.f1_macro_mean.nlargest(5).min()), 2)
N["A"]["families"] = {"cnn_max": r(itA[itA.model.isin(["vgg16","vgg19","resnet50","resnet101","densenet121","densenet201","inception_v3","legacy_xception","resnext50_32x4d","legacy_seresnext50_32x4d"])].f1_macro_mean.max()),
                      "cnn_min": r(itA[itA.model.isin(["vgg16","vgg19","resnet50","resnet101","densenet121","densenet201","inception_v3","legacy_xception","resnext50_32x4d","legacy_seresnext50_32x4d"])].f1_macro_mean.min()),
                      "vit_max": r(itA[itA.model.isin(["vit_small_patch16_224","vit_base_patch16_224","swin_tiny_patch4_window7_224","swin_small_patch4_window7_224","deit3_small_patch16_224"])].f1_macro_mean.max()),
                      "vit_min": r(itA[itA.model.isin(["vit_small_patch16_224","vit_base_patch16_224","swin_tiny_patch4_window7_224","swin_small_patch4_window7_224","deit3_small_patch16_224"])].f1_macro_mean.min())}
N["A"]["ext_smallest4_drop"] = [r(exA.nsmallest(4, "params_M").drop_acc_pp.min(), 1), r(exA.nsmallest(4, "params_M").drop_acc_pp.max(), 1)]
N["A"]["ext_vgg_drop"] = [r(exA[exA.model.isin(["vgg16", "vgg19"])].drop_acc_pp.min(), 1), r(exA[exA.model.isin(["vgg16", "vgg19"])].drop_acc_pp.max(), 1)]
N["A"]["proposed_ext_rank_p3"] = XA["protocols"]["proposed_rank_p3"]

N["A"]["mango6"] = XA["mango6"]
_hA = N["replication"]["internal_f1"]["ranks"]["proposed/" + PROPOSED]
N["replication"]["hmla_benchmark_AB_int_diff_pp"] = r(100 * abs(_hA["A_val"] - _hA["B_val"]), 2)
N["A"]["cd_axis_fraction"] = r(N["A"]["stats"]["friedman_all"]["cd"] / 25, 3)
(MAN / "output" / "numbers.json").write_text(json.dumps(N, indent=2, ensure_ascii=False, default=float))
print("numbers.json written:", len(json.dumps(N)), "characters")
print("INT  proposed F1 %.4f rank %d/31 (%d/26); best %s %.4f; spread26 %.2f pp" % (
    N["internal"]["proposed_f1"], N["internal"]["proposed_rank31"], N["internal"]["proposed_rank26"],
    N["internal"]["best"], N["internal"]["best_f1"], N["internal"]["f1_spread26_pp"]))
print("EXT  proposed acc %.4f (rank %d) F1 %.4f (rank %d); best F1 %s %.4f; spread26 F1 %.1f acc %.1f" % (
    N["external"]["proposed_acc"], N["external"]["proposed_rank31_acc"], N["external"]["proposed_f1"],
    N["external"]["proposed_rank31_f1"], N["external"]["best_f1_model"], N["external"]["best_f1"],
    N["external"]["f1_spread26_pp"], N["external"]["acc_spread26_pp"]))
for k in ["internal_f1", "external_f1", "internal_acc", "external_acc"]:
    s = N["replication"][k]
    print("REP %-12s mean|d| %.3f sd %.3f max %.3f spearmanAB %.3f maxmove %d (%s)" % (
        k, s["mean_abs_pp"], s["sd_pp"], s["max_abs_pp"], s["spearman_AB"], s["max_rank_move"], s["max_rank_move_model"]))
h = N["replication"]["hmla_triplicate"]
print("HMLA x3 ext F1 means", h["external_f1"]["replicate_means"], "mean9", h["external_f1"]["mean9"], "sd9", h["external_f1"]["sd9"])
print("ABL  spearman A-B ext delta %.3f (p %.3f); int %.3f; max|d_ext_pooled| %.2f; pretrain %.2f/%.2f" % (
    N["ablation"]["spearman_AB_ext_delta"], N["ablation"]["spearman_AB_ext_delta_p"], N["ablation"]["spearman_AB_int_delta"],
    N["ablation"]["max_abs_d_ext_pooled_pp"], N["ablation"]["pretrain_d_int_pooled_pp"], N["ablation"]["pretrain_d_ext_pooled_pp"]))
print("TRANSFER spearman intF1-extF1 %.3f p %.4f; intF1-extAcc %.3f p %.4f; proposed drop acc %.1f median %.1f" % (
    N["transfer"]["spearman_intf1_extf1"], N["transfer"]["spearman_intf1_extf1_p"], N["transfer"]["spearman_intf1_extacc"],
    N["transfer"]["spearman_intf1_extacc_p"], N["transfer"]["proposed_drop_acc_pp"], N["transfer"]["median_drop_acc_pp"]))
print("RECIPE ext arch effect vs caformer %.2f pp; pooled %.2f pp; recipe effects ext %s" % (
    N["recipe_matched"]["ext_arch_effect_vs_caformer_pp"], N["recipe_matched"]["pooled_arch_effect_pp"],
    [x["recipe_effect_pp"] for x in N["recipe_matched"]["rows"] if x["split"] == "external"]))
print("DUP", {k: N["duplicates"][k] for k in ["cross_exact_md5", "cross_phash_le6", "cross_embedding_ge95", "within_phash_pairs", "within_cross_split_frac_mean", "chance_cross_split_frac"]})
print("FOR", {k: N["forensics"][k] for k in ["n_models_any_zero", "n_model_seed_zero", "proposed_sooty_recall", "proposed_sooty_to_powdery", "n_models_never_zero"]})
print("PERCLASS A sooty recall", N["per_class"]["sooty_recall_per_seed_A"], "other5 F1", N["per_class"]["other5_mean_f1_A"])
print("GPU h: A %.1f (%d runs)  B %.1f (%d runs)" % (N["complexity"]["gpu_hours_A"], N["complexity"]["n_runs_A"], N["complexity"]["gpu_hours_B"], N["complexity"]["n_runs_B"]))
print("STATS", N["stats"]["friedman_all"]["chi2"], N["stats"]["friedman_all"]["p"], "CD", N["stats"]["friedman_all"]["cd"], "range", N["stats"]["friedman_all"]["rank_range"], "top", N["stats"]["friedman_top"])
print("TTA", N["tta"]["ext_f1_gain_klein_pp"], N["tta"]["ext_f1_gain_d4_pp"], N["tta"]["latency_ratio_klein"], N["tta"]["latency_ratio_d4"])
print("BETA", N["beta"])
