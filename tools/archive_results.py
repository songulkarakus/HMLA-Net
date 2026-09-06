"""
Result archiver: copies a snapshot of the 'results/' folder to 'archive/<label>/'.
Run it BEFORE starting a new study (to preserve the current results) or once a study
is FINISHED, so that every study is kept in its own folder.

By default the huge .pth weight files are EXCLUDED (they are reproducible); all
metrics, predictions (.npz), figures (.png), tables (.csv) and REPORT.md are copied.

Examples:
    python archive_results.py --label 01_HMLA_full
    python archive_results.py --label 02_HMLA_wo_mixstyle --timestamp
    python archive_results.py --label 03_final --include_weights   # .pth included (very large)
"""
import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

# keep non-ASCII characters from crashing the Windows console (cp1252/cp1254)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--archive_dir", default="archive")
    ap.add_argument("--label", required=True, help="study name (becomes the archive folder name)")
    ap.add_argument("--include_weights", action="store_true",
                    help="also copy the .pth weight files (very large)")
    ap.add_argument("--timestamp", action="store_true",
                    help="prefix the folder name with the date and time")
    args = ap.parse_args()

    src = Path(args.results_dir)
    if not src.exists():
        raise SystemExit(f"'{src}' not found — nothing to copy.")

    name = f"{datetime.now():%Y%m%d_%H%M}_{args.label}" if args.timestamp else args.label
    dst = Path(args.archive_dir) / name
    if dst.exists():
        raise SystemExit(f"'{dst}' already exists. Use a different --label or delete the old folder.")

    ignore = None if args.include_weights else shutil.ignore_patterns("*.pth")
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying: '{src}'  ->  '{dst}' ...")
    shutil.copytree(src, dst, ignore=ignore)

    files = [f for f in dst.rglob("*") if f.is_file()]
    size_mb = sum(f.stat().st_size for f in files) / (1024 ** 2)
    note = "" if args.include_weights else "  (.pth weights excluded)"
    print(f"Archived ✓  {len(files)} files, {size_mb:.1f} MB{note}")
    print(f"Location: {dst}")


if __name__ == "__main__":
    main()
