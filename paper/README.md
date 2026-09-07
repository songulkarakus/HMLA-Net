# Paper archive: partitions, run outputs, derived numbers and the scripts that produce the tables

Everything the manuscript quotes can be checked from this folder without retraining anything.

| Folder | Contents |
|---|---|
| `partitions/` | The exact train / validation / test partition of the internal corpus for seeds 42, 1234 and 2024, one JSON per seed (relative image path and label index for every image), and `INDEX.json` with the SHA-256 fingerprint of every partition and the definition of the fingerprint. All models of a seed share the partition. |
| `replication_A/` | The execution reported in the submitted version of the paper (replication A). `runs/` holds the 128 archived runs (25 backbones, the proposed model and 16 single-component removals, three seeds): `config.json`, `metrics_test.json`, `metrics_external_mld24.json`, per-class CSVs, `selection.json` (chosen epoch, EMA or raw weights, calibrated β), `complexity.json`, `history.csv`, plus the summary files written by `analyze.py`; `summary/` the curated summary tables and statistics of that execution; `dataset_eda/` the corpus statistics and figures. |
| `replication_B/` | The execution of the revision (replication B, 150 runs): `runs/` with the same per-run files for the 26 generic configurations, 16 removals, five recipe-matched baselines and the repeat runs, and the Grad-CAM / LAP attention grids of the proposed model; `summary/`, `statistics/`, `recipe_matched/` (including the machine-readable recipe matrix), `zero_shot_protocols/`, `duplicate_audit/`, `class_forensics/` (with the side-by-side image panels of the six shared classes), `tta_symmetry/`, `parameter_analysis/`, `figures/`. |
| `submitted_version/` | Number store and ablation summary of the submitted version (used to keep Tables 3, 7 and 8 identical to it). |
| `architecture/` | `component_locations.csv`: module path, input, output shape and role of every HMLA-Net component. |
| `scripts/` | `derive_A.py`, `derive_numbers_rev.py`, `make_figures_rev.py`, `make_tables_rev.py`, `make_tables_submitted.py`, `clean_test_rescoring.py`, `compare_numbers.py`, `reproduce.sh` and the preserved table bodies of the submitted version (`tables_submitted/`). |
| `output/` | What the scripts produce: `derived/` (intermediate CSV/JSON), `numbers.json` (every quantity quoted in the paper) and `tables/` (LaTeX bodies of the generated tables). The 600 dpi figures are not committed; the scripts regenerate them into `output/figures/`. |
| `reference/numbers.json` | Frozen copy of `output/numbers.json` against which `compare_numbers.py` checks a regeneration. |

`MANIFEST.md` maps every table and figure of the manuscript to the script function and the source files it
is derived from, and states which execution each one reports.

## Per-image predictions

The per-image predictions of the 278 runs (`predictions_test.npz`, `predictions_external*.npz` and the
t-SNE embeddings; 712 files, about 240 MB) are too large for the git tree and are attached to the release of this
version as `predictions.zip`. `scripts/reproduce.sh` downloads and unpacks them into place automatically
when they are missing; they can also be fetched by hand from
https://github.com/songulkarakus/HMLA-Net/releases and unpacked in this folder.

## Reproduction

```bash
bash paper/scripts/reproduce.sh
```

runs, in order, `derive_A.py` (zero-shot protocols P1–P3 and leakage from the replication-A predictions,
clean-test re-scoring, parameter fits, six-class internal accuracy), `derive_numbers_rev.py` (every number
→ `output/numbers.json`, `output/derived/*.csv`), `make_figures_rev.py` (`output/figures/*.png`),
`make_tables_rev.py` (`output/tables/*.tex`), `make_tables_submitted.py` (the bodies of Tables 3, 7 and 8,
which the revision keeps from the submitted version, regenerated from `replication_A/summary/`,
`submitted_version/` and the per-class CSVs of the proposed model with the generator of the submitted
version, and checked to be identical to `scripts/tables_submitted/`) and `compare_numbers.py`, which reports
whether the regenerated `numbers.json` is identical to `reference/numbers.json`. The chain was verified in a clean directory; it
takes a few minutes on a laptop and needs no GPU. Requirements: Python ≥ 3.10 with numpy, pandas, scipy,
scikit-learn, matplotlib and Pillow.

Only the within-corpus near-duplicate scan (`clean_test_rescoring.py`) needs the internal corpus zip
(doi:10.17632/3xd9n7jpc8.1); its outputs (`output/derived/_phash_internal.csv`, `_clean_test_pairs_seed*.csv`,
`_clean_test_summary.json`) are committed so the rest of the chain runs without it. To re-run the scan, place
the Mendeley zip in this folder or set `LEAF_ZIP`.

## Regenerating the statistics from the archived runs

The statistical results of a replication (summary tables, Friedman/Nemenyi with the critical-difference
diagram, Wilcoxon and paired t tests with Holm correction, McNemar with bootstrap intervals, ablation
deltas) are produced by `analyze.py` of this repository from the run folders and `master_results.csv`.
Because the archived run folders keep that layout, the statistics can be recomputed in place:

```bash
python analyze.py --out_dir paper/replication_A/runs      # or paper/replication_B/runs
```

For replication A this rewrites `summary_internal_test.csv`, `proposed_vs_baselines.csv`, `mcnemar.json`,
`ablation_summary.csv`, `stats_report.json` and `cd_diagram.png` in that folder; the values equal the
archived ones (checked: Friedman statistic, p and CD, the per-seed McNemar p values and the Holm-corrected
Wilcoxon table agree to floating-point precision). The McNemar test needs the per-image predictions, i.e.
the release asset described above.

## Replication A versus B

Tables 3, 5, 7 and 8 and Figures 5, 6, 8, 9, 10 and 12 of the paper report replication A, the execution of
the submitted version; Table 5 adds the P3 and leakage columns computed from its archived predictions.
Table 4 and Figure 7 compare the two executions (78 paired runs). Tables 6 and 9 and Figures 11, 13, 14 and
15 report experiments that exist only in replication B (recipe-matched baselines, orbit averaging,
forensics, attention maps). `output/derived/_ablation_reference_sensitivity.csv` gives the replication-B
ablation deltas under both reference definitions discussed in Section 4.6; `output/derived/_A_mango6.csv`
gives the six-class internal accuracy and like-for-like drop of Section 4.4.
