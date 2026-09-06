"""Single-run orchestration: train -> evaluate -> save ALL outputs. Resume/skip capable."""
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from . import viz
from .data import (make_loaders, make_external_loader, LeafDataset,
                   build_transforms, plant_index)
from .engine import train_run, evaluate, TTA_VIEWS
from .metrics import compute_metrics, per_class_dataframe, confusion
from .models import build_model, complexity, ProposedModel
from .utils import measure_latency, fmt_duration


def _load_best(cfg, num_classes, best_path, device):
    model = build_model(cfg, num_classes).to(device)
    ck = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model


@torch.no_grad()
def _extract_embeddings(model, loader, device, max_n=1500):
    """Penultimate feature vectors of the proposed model (for t-SNE)."""
    if not (hasattr(model, "forward_features") and hasattr(model, "pool")):
        return None, None
    feats, labs, count = [], [], 0
    for imgs, tgts in loader:
        imgs = imgs.to(device)
        f = model.pool(model.forward_features(imgs)).flatten(1)
        feats.append(f.cpu().numpy())
        labs.append(np.asarray(tgts))
        count += len(tgts)
        if count >= max_n:
            break
    return np.concatenate(feats)[:max_n], np.concatenate(labs)[:max_n]


def _save_split_results(y_true, y_pred, y_prob, class_names, out_dir: Path,
                        split: str, make_figs=True) -> Dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = compute_metrics(y_true, y_pred, y_prob, class_names)

    with open(out_dir / f"metrics_{split}.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    per_class_dataframe(y_true, y_pred, class_names).to_csv(
        out_dir / f"per_class_{split}.csv", index=False)
    np.savez_compressed(out_dir / f"predictions_{split}.npz",
                        y_true=y_true, y_pred=y_pred, y_prob=y_prob)

    if make_figs:
        cm = confusion(y_true, y_pred, len(class_names))
        viz.plot_confusion(cm, class_names, out_dir / f"confusion_{split}.png",
                           title=f"{split} (acc={m['accuracy']:.3f})")
        try:
            viz.plot_roc(y_true, y_prob, class_names, out_dir / f"roc_{split}.png")
            viz.plot_pr(y_true, y_prob, class_names, out_dir / f"pr_{split}.png")
            viz.plot_reliability(y_true, y_prob, out_dir / f"calibration_{split}.png")
        except Exception as e:
            print(f"    (warning: {split} curve plots skipped: {e})")
    return m


def _append_master(csv_path: Path, row: Dict):
    """Appends a row to the master CSV, aligned to the existing header (missing fields empty, extras dropped)."""
    exists = csv_path.exists()
    if exists:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None) or list(row.keys())
        # if the row has keys that are not in the header, extend the header (rewrite)
        new_keys = [k for k in row.keys() if k not in header]
        if new_keys:
            import pandas as pd
            df = pd.read_csv(csv_path)
            for k in new_keys:
                df[k] = ""
            df.to_csv(csv_path, index=False)
            header = header + new_keys
    else:
        header = list(row.keys())
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in header})


def run_one(cfg, run_dir: Path, class_names: List[str], splits: Dict,
            ext_samples: Optional[List], device: str, master_csv: Path,
            tag: str, log=print) -> Optional[Dict]:
    """Runs one (model, seed) experiment end to end. Skips it if DONE exists."""
    run_dir = Path(run_dir)
    done = run_dir / "DONE"
    if done.exists():
        log(f"[skip] {tag} (already complete)")
        return None
    run_dir.mkdir(parents=True, exist_ok=True)

    # class -> plant mapping for the hierarchical head (BEFORE config.json is written,
    # so that gradcam.py can rebuild the same model from the checkpoint).
    if getattr(cfg, "use_hier", False) and not getattr(cfg, "plant_of", None):
        po, plant_names = plant_index(class_names)
        if po is None:
            log("    (warning: plant groups could not be derived -> hierarchical head disabled)")
            cfg.use_hier = False
        else:
            cfg.plant_of = po
            log(f"    plant groups: {plant_names}")

    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    t_start = time.time()
    started_at = datetime.now()
    num_classes = len(class_names)
    loaders = make_loaders(splits, cfg, device)

    # --- training (resume-capable) ---
    log(f"[train] {tag}  (started {started_at:%Y-%m-%d %H:%M:%S})")
    history, best_path = train_run(cfg, num_classes, loaders, device, run_dir, log=log)
    if history:
        import pandas as pd
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        viz.plot_history(history, run_dir / "history.png")

    # --- complexity + inference speed ---
    model = _load_best(cfg, num_classes, best_path, device)
    cx = complexity(model, cfg.img_size)
    # with TTA enabled the latency is measured with the real number of views (see measure_latency)
    cx.update(measure_latency(model, cfg.img_size, device,
                              n_views=TTA_VIEWS if cfg.use_tta else 1))
    with open(run_dir / "complexity.json", "w", encoding="utf-8") as f:
        json.dump(cx, f, indent=2)

    # --- model-selection / calibration record (reported in the paper) ---
    ck_meta = torch.load(best_path, map_location="cpu", weights_only=False)
    sel = {k: ck_meta.get(k) for k in
           ("epoch", "score", "val_acc", "select_metric", "source",
            "hier_beta", "hier_beta_scores")}
    with open(run_dir / "selection.json", "w", encoding="utf-8") as f:
        json.dump(sel, f, indent=2)

    # --- internal test evaluation ---
    yt, yp, yprob, _ = evaluate(model, loaders[2], device,
                                cfg.use_amp and device == "cuda", tta=cfg.use_tta)
    train_sec = sum(h.get("epoch_sec", 0) for h in history)  # cumulative (resume-safe)
    cx_row = {"params_M": round(cx["params_M"], 3), "size_MB": round(cx["size_MB"], 2),
              "gflops": cx["gflops"], "latency_ms": round(cx["latency_ms"], 3),
              "latency_ms_single_pass": round(cx.get("latency_ms_single_pass",
                                                     cx["latency_ms"]), 3),
              "tta_views": cx.get("tta_views", 1),
              "fps": round(cx["fps"], 1), "train_sec": round(train_sec, 1),
              "n_epochs": len(history),
              "best_epoch": sel.get("epoch"), "select_src": sel.get("source"),
              "hier_beta": sel.get("hier_beta")}
    m_test = _save_split_results(yt, yp, yprob, class_names, run_dir, "test")
    log(f"    TEST acc={m_test['accuracy']:.4f}  f1={m_test['f1_macro']:.4f}  "
        f"auc={m_test.get('auc_macro_ovr', float('nan')):.4f}")
    # Rows are BUFFERED and written just before DONE. They used to be written here; an
    # interruption during the t-SNE that follows (proposed model only, can take minutes)
    # left the run without DONE, it restarted from scratch and a second row for the SAME
    # (tag, seed) was appended to master_results.csv -> analyze.py counted that seed
    # twice, and since n_seeds still showed 3 nobody noticed.
    pending_rows = [{
        "tag": tag, "experiment": cfg.experiment, "model": cfg.model_name,
        "seed": cfg.seed, "split": "internal_test", "n_test": len(yt),
        **{k: round(v, 6) if isinstance(v, float) else v for k, v in m_test.items()},
        **cx_row}]

    # --- external validation ---
    if ext_samples:
        ext_loader = make_external_loader(ext_samples, cfg, device)
        yt2, yp2, yprob2, _ = evaluate(model, ext_loader, device,
                                       cfg.use_amp and device == "cuda", tta=cfg.use_tta)
        m_ext = _save_split_results(yt2, yp2, yprob2, class_names, run_dir,
                                    "external", make_figs=True)
        log(f"    EXTERNAL acc={m_ext['accuracy']:.4f}  f1={m_ext['f1_macro']:.4f}")
        pending_rows.append({
            "tag": tag, "experiment": cfg.experiment, "model": cfg.model_name,
            "seed": cfg.seed, "split": "external", "n_test": len(yt2),
            **{k: round(v, 6) if isinstance(v, float) else v for k, v in m_ext.items()},
            **cx_row})

    # --- t-SNE (proposed model only) ---
    if isinstance(model, ProposedModel):
        try:
            emb, labs = _extract_embeddings(model, loaders[2], device)
            if emb is not None:
                np.savez_compressed(run_dir / "embeddings_test.npz", emb=emb, labels=labs)
                viz.plot_tsne(emb, labs, class_names, run_dir / "tsne_test.png",
                              seed=cfg.seed)
        except Exception as e:
            log(f"    (warning: t-SNE skipped: {e})")

    # --- timing (file + console) ---
    ended_at = datetime.now()
    run_sec = time.time() - t_start
    timing = {"tag": tag, "model": cfg.model_name, "seed": cfg.seed,
              "started_at": started_at.isoformat(timespec="seconds"),
              "ended_at": ended_at.isoformat(timespec="seconds"),
              "run_duration_sec": round(run_sec, 1),
              "train_sec_cumulative": round(train_sec, 1),
              "n_epochs": len(history)}
    with open(run_dir / "timing.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    log(f"    TIME: {fmt_duration(run_sec)} (training {fmt_duration(train_sec)}, "
        f"{len(history)} epochs) | {started_at:%H:%M:%S} -> {ended_at:%H:%M:%S}")

    # --- master rows + DONE (one block: an interrupted run leaves no row behind) ---
    for row in pending_rows:
        _append_master(master_csv, row)
    _append_master(master_csv.parent / "timing.csv", timing)
    done.write_text("ok", encoding="utf-8")
    return {"test": m_test}
