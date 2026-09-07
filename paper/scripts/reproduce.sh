#!/bin/bash
# Regenerates every derived number, table body and figure of the manuscript from the archived run outputs.
# Requires: python3 >= 3.10 with numpy, pandas, scipy, scikit-learn, matplotlib, Pillow. No GPU, no training.
set -e
cd "$(dirname "$0")/.."
if [ ! -f replication_A/runs/proposed/Proposed_HMLA_CAFormerS18/seed42/predictions_test.npz ]; then
    # The per-image predictions of the 278 runs (about 240 MB) are attached to the release of this version.
    echo "downloading predictions.zip (about 240 MB) from the repository release ..."
    curl -L -o predictions.zip https://github.com/songulkarakus/HMLA-Net/releases/download/v1.1.2/predictions.zip
    unzip -q -o predictions.zip && rm predictions.zip
fi
python3 scripts/derive_A.py            # replication A: zero-shot protocols, clean-test re-scoring, parameter fits, mango-6 drop
python3 scripts/derive_numbers_rev.py  # every number -> output/numbers.json, output/derived/*.csv
python3 scripts/make_figures_rev.py    # figures -> output/figures/*.png
python3 scripts/make_tables_rev.py     # table bodies -> output/tables/*.tex
python3 scripts/make_tables_submitted.py  # Tables 3, 7, 8 of the submitted version from the replication-A CSVs; must equal scripts/tables_submitted/
python3 scripts/compare_numbers.py     # regenerated numbers == frozen reference copy
