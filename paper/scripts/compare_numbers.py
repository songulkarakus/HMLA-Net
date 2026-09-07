"""Compares the regenerated output/numbers.json with the frozen copy in reference/numbers.json."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
a = json.loads((ROOT / "output" / "numbers.json").read_text())
b = json.loads((ROOT / "reference" / "numbers.json").read_text())
diffs = []


def walk(x, y, path=""):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k not in x or k not in y:
                diffs.append((path + "/" + k, "missing on one side"))
            else:
                walk(x[k], y[k], path + "/" + k)
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            diffs.append((path, f"length {len(x)} vs {len(y)}"))
        else:
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]")
    else:
        same = (x == y) or (isinstance(x, float) and isinstance(y, float)
                            and ((math.isnan(x) and math.isnan(y)) or abs(x - y) < 1e-9))
        if not same:
            diffs.append((path, f"{x!r} vs {y!r}"))


walk(a, b)
if diffs:
    print(f"{len(diffs)} difference(s) between output/numbers.json and reference/numbers.json:")
    for p, d in diffs[:50]:
        print("  ", p, d)
    sys.exit(1)
print("output/numbers.json is identical to reference/numbers.json "
      f"({len(json.dumps(a))} characters).")
