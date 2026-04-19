"""Build 30-s stage-labelled epochs from cleaned derivatives."""
from __future__ import annotations

from pathlib import Path

import mne
from tqdm import tqdm

from _common import arg_parser, log_failure, select_subjects
from eeg_analysis import config as cfg
from eeg_analysis.src import io_eeg, staging


def _epoch_path(run) -> Path:
    return cfg.subject_deriv_dir(run.subject) / f"{run.tag}_desc-stages_epo.fif"


def _clean_path(run) -> Path:
    return cfg.subject_deriv_dir(run.subject) / f"{run.tag}_desc-clean_raw.fif"


def main() -> None:
    args = arg_parser(__doc__).parse_args()
    subjects = select_subjects(args)
    runs = [r for s in subjects for r in io_eeg.list_runs(s)]

    ok = 0
    for run in tqdm(runs, desc="epochs"):
        raw_fp = _clean_path(run)
        out = _epoch_path(run)
        if out.exists() and not args.overwrite:
            ok += 1
            continue
        if not raw_fp.exists():
            log_failure("02_stage_epochs", run.tag,
                        f"Missing cleaned raw: {raw_fp}")
            continue
        try:
            raw = mne.io.read_raw_fif(str(raw_fp), preload=True,
                                       verbose="ERROR")
            epochs = staging.make_stage_epochs(run, raw)
            epochs.save(str(out), overwrite=True, verbose="ERROR")
            ok += 1
        except Exception as e:  # noqa: BLE001
            log_failure("02_stage_epochs", run.tag, f"{type(e).__name__}: {e}")

    print(f"Wrote {ok}/{len(runs)} stage-epoch files.")


if __name__ == "__main__":
    main()
