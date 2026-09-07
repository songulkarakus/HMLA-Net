#!/bin/bash
# Regenerates every derived number, table body and figure of the manuscript from the archived run outputs.
# Requires: python3 >= 3.10 with numpy, pandas, scipy, scikit-learn, matplotlib, Pillow. No GPU, no training.
set -e
cd "$(dirname "$0")/.."
python3 scripts/derive_A.py            # replication A: zero-shot protocols, clean-test re-scoring, parameter fits, mango-6 drop
python3 scripts/derive_numbers_rev.py  # every number -> output/numbers.json, output/derived/*.csv
python3 scripts/make_figures_rev.py    # figures -> output/figures/*.png
python3 scripts/make_tables_rev.py     # table bodies -> output/tables/*.tex
python3 scripts/compare_numbers.py     # regenerated numbers == frozen reference copy
