"""Near-duplicate-free internal test re-scoring (replication B).

  1. 64-bit perceptual hash of the 7,179 internal images (same definition as imagehash.phash: 32x32 grey
     LANCZOS -> 2-D DCT -> top-left 8x8 -> median threshold). Needs the internal corpus zip
     (set LEAF_ZIP or place the Mendeley zip at the package root); the cached hashes are shipped in
     output/derived/_phash_internal.csv so the rest of the chain runs without the corpus.
  2. Test-train pairs per seed (Hamming <= 6) and the flagged test images.
  3. Macro-F1 / accuracy of the 150 replication-B runs on the full test partition and on the clean subset.
Outputs: output/derived/_phash_internal.csv, _clean_test_pairs_seed*.csv, _clean_test_scores.csv,
         _clean_test_summary.csv, _clean_test_summary.json
"""
import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fftpack import dct
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score

MAN = Path(__file__).resolve().parents[1]
ROOT = MAN
B = MAN / "replication_B"
RUNS = B / "runs"
DER = MAN / "output" / "derived"
import os
ZIP = Path(os.environ.get("LEAF_ZIP", str(MAN / "Multi-Crop Leaf Disease Dataset for Deep Learning-.zip")))
THR = 6
CLASSES = json.loads((B / "summary" / "class_names.json").read_text())


def phash_bits(img):
    im = img.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    px = np.asarray(im, dtype=np.float64)
    d = dct(dct(px, axis=0), axis=1)
    low = d[:8, :8]
    return (low > np.median(low)).flatten()


def rel(p):
    p = p.replace("\\", "/")
    i = p.find("multiple leaf dataset/")
    return p[i:]


cache = DER / "_phash_internal.csv"
if cache.exists():
    ph = pd.read_csv(cache)
else:
    outer = zipfile.ZipFile(ZIP)
    inner_names = [n for n in outer.namelist() if n.endswith("multiple leaf dataset.zip")]
    inner = zipfile.ZipFile(outer.open(inner_names[0])) if inner_names else outer
    names = [n for n in inner.namelist() if not n.endswith("/") and n.lower().endswith((".jpg", ".jpeg", ".png"))]
    rows = []
    for k, n in enumerate(names):
        img = Image.open(io.BytesIO(inner.read(n)))
        bits = phash_bits(img)
        h = int("".join("1" if b else "0" for b in bits), 2)
        rows.append({"path": n, "hash": h, "class": n.split("/")[2]})
        if k % 1000 == 0:
            print(f"  phash {k}/{len(names)}", file=sys.stderr)
    ph = pd.DataFrame(rows)
    ph.to_csv(cache, index=False)
print("hashed images:", len(ph))

ph["key"] = ph.path.map(lambda s: s[s.find("multiple leaf dataset/"):] if "multiple leaf dataset/" in s else "multiple leaf dataset/" + s.lstrip("/"))
key2hash = dict(zip(ph.key, ph.hash.astype(np.uint64)))
key2class = dict(zip(ph.key, ph["class"]))

splits = {s: json.loads((MAN / "partitions" / f"split_seed{s}.json").read_text())["splits"] for s in (42, 1234, 2024)}


def hashes_for(entries):
    keys = [rel(e["path"]) for e in entries]
    miss = [k for k in keys if k not in key2hash]
    assert not miss, f"{len(miss)} paths not found in the zip, e.g. {miss[:2]}"
    return keys, np.array([key2hash[k] for k in keys], dtype=np.uint64)


def hamming_matrix(a, b):
    x = np.bitwise_xor(a[:, None], b[None, :])
    cnt = np.zeros(x.shape, dtype=np.uint8)
    for _ in range(8):
        cnt += np.unpackbits((x & np.uint64(0xFF)).astype(np.uint8)).reshape(x.shape + (8,)).sum(-1).astype(np.uint8)
        x = x >> np.uint64(8)
    return cnt


allkeys = list(key2hash.keys())
H = np.array([key2hash[k] for k in allkeys], dtype=np.uint64)
n = len(H)
tot = 0
cross_cls = 0
cls = np.array([key2class[k] for k in allkeys])
for i0 in range(0, n, 500):
    hm = hamming_matrix(H[i0:i0 + 500], H)
    for r in range(hm.shape[0]):
        i = i0 + r
        js = np.where(hm[r] <= THR)[0]
        js = js[js > i]
        tot += len(js)
        cross_cls += int((cls[js] != cls[i]).sum())
print(f"within-corpus pHash<= {THR} pairs: {tot} (replication-B package: 1423) | cross-class: {cross_cls} (package: 424)")

pairs_by_seed, flagged_by_seed = {}, {}
for s, sp in splits.items():
    tk, th = hashes_for(sp["test"])
    trk, trh = hashes_for(sp["train"])
    vk, vh = hashes_for(sp["val"])
    hm = hamming_matrix(th, trh)
    ii, jj = np.where(hm <= THR)
    pairs = pd.DataFrame({"test_idx": ii, "test_path": [tk[i] for i in ii], "train_path": [trk[j] for j in jj],
                          "hamming": hm[ii, jj], "same_class": [key2class[tk[i]] == key2class[trk[j]] for i, j in zip(ii, jj)]})
    pairs.to_csv(DER / f"_clean_test_pairs_seed{s}.csv", index=False)
    hv = hamming_matrix(th, vh)
    n_tv = int((hv <= THR).sum())
    flagged = np.zeros(len(tk), dtype=bool)
    flagged[np.unique(ii)] = True
    flagged_same = np.zeros(len(tk), dtype=bool)
    flagged_same[np.unique(ii[pairs.same_class.values])] = True
    pairs_by_seed[s] = pairs
    flagged_by_seed[s] = {"any": flagged, "same_class": flagged_same}
    print(f"seed {s}: test-train pairs {len(pairs)} (package {dict(zip([42,1234,2024],[283,317,268]))[s]}), test-val pairs {n_tv}; "
          f"flagged test images {flagged.sum()} ({100*flagged.mean():.1f}%), same-class only {flagged_same.sum()}")

rows = []
for p in sorted(RUNS.glob("*/*/seed*/predictions_test.npz")):
    exp, model, seed = p.parts[-4], p.parts[-3], int(p.parts[-2].replace("seed", ""))
    z = np.load(p)
    yt, yp = z["y_true"], z["y_pred"]
    assert len(yt) == len(splits[seed]["test"])
    labs = np.array([e["label"] for e in splits[seed]["test"]])
    assert (labs == yt).all(), (exp, model, seed)
    fl = flagged_by_seed[seed]
    keep = ~fl["any"]
    keep_sc = ~fl["same_class"]
    rows.append({"experiment": exp, "model": model, "seed": seed, "n_test": len(yt), "n_clean": int(keep.sum()),
                 "f1_full": f1_score(yt, yp, average="macro"), "acc_full": accuracy_score(yt, yp),
                 "f1_clean": f1_score(yt[keep], yp[keep], average="macro"), "acc_clean": accuracy_score(yt[keep], yp[keep]),
                 "f1_flagged": f1_score(yt[~keep], yp[~keep], average="macro") if (~keep).sum() else np.nan,
                 "acc_flagged": accuracy_score(yt[~keep], yp[~keep]) if (~keep).sum() else np.nan,
                 "f1_clean_sc": f1_score(yt[keep_sc], yp[keep_sc], average="macro"), "acc_clean_sc": accuracy_score(yt[keep_sc], yp[keep_sc])})
sc = pd.DataFrame(rows)
sc.to_csv(DER / "_clean_test_scores.csv", index=False)
sc["cfg"] = sc.experiment + "/" + sc.model
g = sc.groupby(["experiment", "model", "cfg"]).agg(f1_full=("f1_full", "mean"), f1_full_sd=("f1_full", "std"),
                                                   f1_clean=("f1_clean", "mean"), f1_clean_sd=("f1_clean", "std"),
                                                   acc_full=("acc_full", "mean"), acc_clean=("acc_clean", "mean"),
                                                   acc_flagged=("acc_flagged", "mean"), f1_flagged=("f1_flagged", "mean"),
                                                   f1_clean_sc=("f1_clean_sc", "mean"), n_seeds=("seed", "count")).reset_index()
g["d_f1_pp"] = 100 * (g.f1_clean - g.f1_full)
g["d_acc_pp"] = 100 * (g.acc_clean - g.acc_full)
main = g[g.experiment.isin(["benchmark", "proposed", "recipe_matched"])].copy()
main["rank_full"] = main.f1_full.rank(ascending=False).astype(int)
main["rank_clean"] = main.f1_clean.rank(ascending=False).astype(int)
main = main.sort_values("f1_clean", ascending=False)
main.to_csv(DER / "_clean_test_summary.csv", index=False)
gen = main[main.experiment.isin(["benchmark", "proposed"])]
rho, pv = stats.spearmanr(main.f1_full, main.f1_clean)
rho26, _ = stats.spearmanr(gen.f1_full, gen.f1_clean)
prop = main[main.model == "Proposed_HMLA_CAFormerS18"].iloc[0]
summary = {
    "threshold": THR,
    "corpus_pairs_recomputed": int(tot), "corpus_cross_class_pairs_recomputed": int(cross_cls),
    "test_train_pairs_by_seed": {int(s): int(len(p_)) for s, p_ in pairs_by_seed.items()},
    "flagged_test_images_by_seed": {int(s): int(f["any"].sum()) for s, f in flagged_by_seed.items()},
    "flagged_test_pct_mean": round(float(np.mean([100 * f["any"].mean() for f in flagged_by_seed.values()])), 1),
    "flagged_same_class_by_seed": {int(s): int(f["same_class"].sum()) for s, f in flagged_by_seed.items()},
    "n_clean_by_seed": {int(s): int((~f["any"]).sum()) for s, f in flagged_by_seed.items()},
    "n_runs": int(len(sc)), "n_configs_main": int(len(main)),
    "mean_d_f1_pp_all31": round(float(main.d_f1_pp.mean()), 3), "min_d_f1_pp": round(float(main.d_f1_pp.min()), 3),
    "max_d_f1_pp": round(float(main.d_f1_pp.max()), 3), "mean_d_acc_pp_all31": round(float(main.d_acc_pp.mean()), 3),
    "n_configs_f1_drops": int((main.d_f1_pp < 0).sum()), "n_configs_f1_rises": int((main.d_f1_pp > 0).sum()),
    "spread_full_pp": round(float(100 * (gen.f1_full.max() - gen.f1_full.min())), 2),
    "spread_clean_pp": round(float(100 * (gen.f1_clean.max() - gen.f1_clean.min())), 2),
    "spearman_full_clean_31": round(float(rho), 3), "spearman_full_clean_26": round(float(rho26), 3),
    "best_clean_model": main.iloc[0].model, "best_clean_f1": round(float(main.iloc[0].f1_clean), 4),
    "proposed_f1_full": round(float(prop.f1_full), 4), "proposed_f1_clean": round(float(prop.f1_clean), 4),
    "proposed_acc_full": round(float(prop.acc_full), 4), "proposed_acc_clean": round(float(prop.acc_clean), 4),
    "proposed_rank_full": int(prop.rank_full), "proposed_rank_clean": int(prop.rank_clean),
    "proposed_d_f1_pp": round(float(prop.d_f1_pp), 2),
    "acc_flagged_mean_all": round(float(main.acc_flagged.mean()), 4), "acc_clean_mean_all": round(float(main.acc_clean.mean()), 4),
    "max_rank_move": int((main.rank_full - main.rank_clean).abs().max()),
    "n_rank_move_ge3": int(((main.rank_full - main.rank_clean).abs() >= 3).sum()),
}
(DER / "_clean_test_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=1))
print(main[["model", "f1_full", "f1_clean", "d_f1_pp", "acc_full", "acc_clean", "acc_flagged", "rank_full", "rank_clean"]].round(4).to_string())
