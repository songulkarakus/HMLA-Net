# Manifest: manuscript tables and figures → scripts and sources

All paths are relative to the package root. `N` denotes `output/numbers.json`; functions are in `scripts/`.
"A" = replication A (submitted execution), "B" = replication B (this revision).

## Tables

| Table | Reports | Generator | Sources |
|---|---|---|---|
| 1 Literature | – | static block in the manuscript; the two numbers of the "This work" row are written to `output/tables/tab_literature_this_work.json` | `N.A.internal.proposed_acc`, `N.A.external.proposed_acc` |
| 2 Corpus composition | – | `make_tables_rev.t_dataset` → `output/tables/tab_dataset.tex` | `replication_A/dataset_eda/class_distribution.csv` |
| 3 Internal benchmark | A | `t_internal` (submitted body, unchanged) | `scripts/tables_submitted/tab_internal.tex`; source `replication_A/summary/summary_internal_test.csv` |
| 4 Replicate variation | A vs B | `t_replication` ← `N.replication` | `replication_A/summary/master_results.csv`, `replication_B/summary/master_results.csv` (`derive_numbers_rev.py`) |
| 5 Zero-shot MLD24 | A | `t_external` | `output/derived/_external_A.csv` ← `replication_A/summary/summary_external_mld24.csv` + `output/derived/_A_protocols.csv` (P3, leakage; `derive_A.py` from `replication_A/runs/*/predictions_external_mld24.npz`) |
| 6 Recipe-matched | B | `t_recipe` ← `N.recipe_matched`, `N.replication` | `replication_B/recipe_matched/recipe_matched_comparison.csv`, `recipe_matched_paired.csv`, `recipe_matrix.csv`; HMLA-Net internal value `N.internal` (B) |
| 7 Ablation | A (note: B) | `t_ablation` (submitted body; caption and note regenerated) | `scripts/tables_submitted/tab_ablation.tex`; source `replication_A/runs/ablation_summary.csv`; B comparison `N.ablation` ← `replication_B/summary/ablation_summary_*.csv`, `submitted_version/_ablation_both.csv` |
| 8 Per-class | A | `t_perclass` (submitted body, unchanged) | `scripts/tables_submitted/tab_perclass.tex`; source `replication_A/runs/proposed/*/per_class_*.csv` |
| 9 Orbit averaging | B | `t_tta` ← `N.tta` | `replication_B/tta_symmetry/tta_symmetry.json` |

## Figures

| Figure | Reports | Generator | Sources |
|---|---|---|---|
| 1 Pipeline | – | drawn in the manuscript source | – |
| 2 Corpus | – | `make_figures_rev.fig_dataset` → `output/figures/fig1_dataset.png` | `replication_A/dataset_eda/class_distribution.csv`, `N.dataset` |
| 3 Samples | – | prepared image | corpus images |
| 4 Architecture | – | drawn in the manuscript source | – |
| 5 Internal ranking | A | `fig_internal_ranking` | `output/derived/_internal_A.csv`, bands from `N.replication.internal_f1` |
| 6 Critical-difference | A | `fig_cd` | `replication_A/summary/stats_report.json` |
| 7 Two executions | A vs B | `fig_replication` | `output/derived/_replication_internal_pairs.csv`, `_replication_external_pairs.csv` |
| 8 Transfer | A | `fig_transfer` | `output/derived/_external_A.csv`, `N.A.transfer` |
| 9 Ablation | A | `fig_ablation` | `N.A.ablation.rows` |
| 10 Per-class | A | `fig_perclass` | `output/derived/_per_class_external_A.csv`, `_per_class_internal_A.csv`, `_confusion_external_A.csv` (from `replication_A/runs/proposed/*`) |
| 11 Sooty mould panel | – | `fig_forensics` | `replication_B/class_forensics/panels/mango_sooty_mould.png` |
| 12 Parameters vs accuracy | A | `fig_efficiency` | `output/derived/_external_A.csv`, `N.A.params` (`derive_A.py`; VGG-excluded fit `external_acc_novgg`) |
| 13 Grad-CAM | B | `fig_gradcam` | `replication_B/runs/proposed/gradcam/gradcam_grid.png` |
| 14 Grad-CAM vs LAP | B | `fig_lap` | same folder, `lap_attention_grid.png` |
| 15 t-SNE | B | `fig_tsne` | `replication_B/runs/proposed/Proposed_HMLA_CAFormerS18/seed1234/embeddings_test.npz` |

## Text-only quantities

| Quantity | Where | Source |
|---|---|---|
| Duplicate audit (MD5, pHash, embeddings) | §3.1 | `replication_B/duplicate_audit/duplicate_report.json`, `cross_corpus_examples.csv`, `split_leakage.csv` |
| Within-corpus near-duplicates and clean-test re-scoring | §3.2 | `output/derived/_clean_test_*` (B), `output/derived/_A_clean_test_summary.csv` (A) |
| Zero-shot protocols P1–P3, leakage | §3.7, §4.4 | `output/derived/_A_protocols.csv` (A), `replication_B/zero_shot_protocols/zeroshot_protocols.csv` (B) |
| Six-class internal accuracy and like-for-like drop | §4.4 | `output/derived/_A_mango6.csv` |
| Ablation deltas of B under both references | §4.6 | `output/derived/_ablation_reference_sensitivity.csv`, `_ablation_AB.csv` |
| Class-level forensics, descriptors, destinations | §4.7 | `replication_B/class_forensics/*` |
| Parameter-count fits (26 models; 24 without VGG) | §4.8 | `N.A.params` (`derive_A.py`); B bins in `replication_B/parameter_analysis/` |
| Statistical power, Friedman/Nemenyi, McNemar | §3.8, §4.2 | `replication_A/summary/stats_report.json`, `replication_B/statistics/stats_report.json`, `statistical_power.json` |
| Compute cost | §3.10 | `train_sec` in both `master_results.csv` files |
| Partition fingerprints | §3.2 | `partitions/INDEX.json` |
