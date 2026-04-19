"""AAS + BCG + filter + ICA preprocessing for every run."""
from __future__ import annotations

from joblib import Parallel, delayed
from tqdm import tqdm

from _common import arg_parser, log_failure, select_subjects
from eeg_analysis.src import io_eeg, preprocess


def _run_one(run, overwrite):
    try:
        res = preprocess.preprocess_run(run, overwrite=overwrite)
        return run.tag, res.get("status", "ok"), ""
    except Exception as e:  # noqa: BLE001
        return run.tag, "failed", f"{type(e).__name__}: {e}"


def main() -> None:
    args = arg_parser(__doc__).parse_args()
    subjects = select_subjects(args)

    runs = [r for s in subjects for r in io_eeg.list_runs(s)]
    print(f"Preprocessing {len(runs)} runs from {len(subjects)} subjects "
          f"(n_jobs={args.n_jobs})")

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_run_one)(r, args.overwrite) for r in tqdm(runs, desc="runs"))

    ok = sum(1 for _, s, _ in results if s in ("ok", "cached"))
    for tag, status, msg in results:
        if status == "failed":
            log_failure("01_preprocess", tag, msg)
    print(f"Done. ok/cached = {ok}/{len(results)}. "
          f"See logs/failures.csv for failures.")


if __name__ == "__main__":
    main()
