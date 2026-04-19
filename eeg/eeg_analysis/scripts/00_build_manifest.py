"""Enumerate subjects x runs and stage coverage; write a manifest CSV."""
from __future__ import annotations

import pandas as pd
from tqdm import tqdm

from _common import arg_parser, select_subjects
from eeg_analysis import config as cfg
from eeg_analysis.src import io_eeg, staging


def main() -> None:
    args = arg_parser(__doc__).parse_args()
    subjects = select_subjects(args)

    rows = []
    for sub in tqdm(subjects, desc="manifest"):
        for run in io_eeg.list_runs(sub):
            try:
                tbl = staging.run_stage_table(run)
            except Exception as e:  # noqa: BLE001
                rows.append({"subject": sub, "run": run.tag,
                             "task": run.task, "run_idx": run.run,
                             "n_epochs_total": 0, "Wake": 0, "N1": 0, "N2": 0,
                             "error": str(e)})
                continue
            counts = tbl["stage"].value_counts().to_dict()
            rows.append({
                "subject": sub, "run": run.tag,
                "task": run.task, "run_idx": run.run,
                "n_epochs_total": int(len(tbl)),
                "Wake": int(counts.get("Wake", 0)),
                "N1":   int(counts.get("N1", 0)),
                "N2":   int(counts.get("N2", 0)),
                "error": "",
            })
    df = pd.DataFrame(rows)
    out = cfg.TABLES_DIR / "manifest_runs_stages.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {out}   ({len(df)} runs)")


if __name__ == "__main__":
    main()
