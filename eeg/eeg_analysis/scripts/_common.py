"""Common CLI / path utilities shared by the pipeline scripts."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Ensure parent (eeg/) is on sys.path so `import eeg_analysis` works even
# when scripts are launched with `python scripts/01_...py`.
_HERE = Path(__file__).resolve()
_EEG_DIR = _HERE.parents[2]           # .../eeg
if str(_EEG_DIR) not in sys.path:
    sys.path.insert(0, str(_EEG_DIR))

from eeg_analysis import config as cfg  # noqa: E402


def arg_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--subjects", nargs="*", default=None,
                   help="Subset of sub-XX ids (default: all cohort subjects).")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute and overwrite cached derivatives.")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Parallel workers where supported.")
    return p


def select_subjects(args) -> list[str]:
    if args.subjects:
        bad = [s for s in args.subjects if s not in cfg.ALL_SUBJECTS]
        if bad:
            raise SystemExit(f"Unknown subject ids: {bad}")
        return args.subjects
    return list(cfg.SUBJECTS)


def log_failure(script: str, tag: str, msg: str) -> None:
    fp = cfg.LOGS_DIR / "failures.csv"
    new = not fp.exists()
    with open(fp, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["script", "tag", "message"])
        w.writerow([script, tag, msg[:500]])
