# Paper reproduction folder

This folder holds, unchanged, the parts of the paper's supplementary material that fit in a code
repository:

- `partitions/` — the exact train / validation / test partition of the internal corpus for seeds 42, 1234
  and 2024 (one JSON per seed: relative image path and label index for every image) and `INDEX.json` with
  the SHA-256 fingerprint of every partition and the definition of the fingerprint.
- `submitted_version/` — number store and ablation summary of the submitted version of the paper (used to
  keep Tables 3, 7 and 8 identical to it).
- `scripts/` — the scripts that derive every number quoted in the manuscript (`derive_A.py`,
  `derive_numbers_rev.py`), regenerate the figures (`make_figures_rev.py`) and the LaTeX bodies of the
  generated tables (`make_tables_rev.py`), the within-corpus near-duplicate scan (`clean_test_rescoring.py`),
  `compare_numbers.py` and `reproduce.sh`; `tables_submitted/` holds the preserved table bodies of the
  submitted version.
- `output/` — what the scripts produce: `derived/` (intermediate CSV/JSON), `numbers.json` (every quantity
  quoted in the paper) and `tables/` (LaTeX table bodies). The 600 dpi figures are not committed; the
  scripts regenerate them into `output/figures/`.
- `reference/numbers.json` — frozen copy of `output/numbers.json` used by `compare_numbers.py`.

The scripts read the archived per-run outputs of the two complete executions of the benchmark
(`replication_A/`, `replication_B/`: 128 + 150 runs with their per-image predictions, about 280 MB),
which are distributed as the paper's supplementary material rather than in this repository. To run the
chain, unpack the supplementary package and copy its `replication_A/` and `replication_B/` folders into
this `paper/` folder (its other folders are identical to the ones here), then

```bash
bash paper/scripts/reproduce.sh
```

which rewrites `output/` and checks the regenerated `numbers.json` against `reference/numbers.json`.
Requirements: Python >= 3.10 with numpy, pandas, scipy, scikit-learn, matplotlib and Pillow; no GPU.
