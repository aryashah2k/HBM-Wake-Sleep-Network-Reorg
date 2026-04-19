"""Cycle-by-cycle alpha waveform shape (bycycle) — Wake vs N2."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from _common import arg_parser, log_failure, select_subjects
from eeg_analysis import config as cfg
from eeg_analysis.src import io_eeg, bycycle_alpha
from eeg_analysis.src import stats as stats_mod
from eeg_analysis.src.plotting import default_style, save_fig


FEATURES = ["time_rdsym", "time_ptsym", "volt_amp", "period"]


def _epo(run) -> Path:
    return cfg.subject_deriv_dir(run.subject) / f"{run.tag}_desc-stages_epo.fif"


def _concat(subject: str) -> mne.Epochs | None:
    eps = []
    for run in io_eeg.list_runs(subject):
        fp = _epo(run)
        if fp.exists():
            try:
                eps.append(mne.read_epochs(str(fp), preload=True,
                                           verbose="ERROR"))
            except Exception:
                continue
    return mne.concatenate_epochs(eps, add_offset=True) if eps else None


def main() -> None:
    args = arg_parser(__doc__).parse_args()
    subjects = select_subjects(args)
    default_style()

    per_subject_medians: list[dict] = []
    all_cycles: list[pd.DataFrame] = []

    for sub in tqdm(subjects, desc="subjects"):
        ep = _concat(sub)
        if ep is None:
            log_failure("05_bycycle_alpha", sub, "No epochs")
            continue
        dfs = []
        for stage in ("Wake", "N2"):
            try:
                df = bycycle_alpha.cycle_features_run(
                    ep, stage, channels=cfg.OCCIPITAL_PRIMARY)
            except Exception as e:  # noqa: BLE001
                log_failure("05_bycycle_alpha", f"{sub}:{stage}", str(e))
                continue
            if df.empty or "error" in df.columns:
                continue
            dfs.append(df)
        if not dfs:
            continue
        sub_df = pd.concat(dfs, ignore_index=True)
        sub_df["subject"] = sub
        all_cycles.append(sub_df)

        for stage in ("Wake", "N2"):
            for ch in cfg.OCCIPITAL_PRIMARY:
                s = sub_df[(sub_df["stage"] == stage) &
                           (sub_df["channel"] == ch)]
                row = {"subject": sub, "stage": stage, "channel": ch,
                       "n_cycles": len(s)}
                for feat in FEATURES:
                    row[feat] = float(s[feat].median()) if feat in s and len(s) else np.nan
                per_subject_medians.append(row)

    if not per_subject_medians:
        raise SystemExit("No bycycle data.")
    med = pd.DataFrame(per_subject_medians)
    med.to_csv(cfg.TABLES_DIR / "bycycle_per_subject_medians.csv", index=False)

    # Paired tests per feature per channel, FDR over (features x channels)
    tests = []
    for feat in FEATURES:
        for ch in cfg.OCCIPITAL_PRIMARY:
            wide = med[med["channel"] == ch].pivot(index="subject",
                                                    columns="stage",
                                                    values=feat).dropna()
            if wide.empty or "Wake" not in wide or "N2" not in wide:
                continue
            r = stats_mod.wilcoxon_paired(wide["Wake"].to_numpy(),
                                          wide["N2"].to_numpy())
            r.update({"feature": feat, "channel": ch,
                      "Wake_median": float(wide["Wake"].median()),
                      "N2_median": float(wide["N2"].median())})
            tests.append(r)
    tests_df = pd.DataFrame(tests)
    tests_df["q_fdr"] = stats_mod.fdr_bh(tests_df["p"].values)
    tests_df.to_csv(cfg.TABLES_DIR / "bycycle_wake_vs_n2_stats.csv", index=False)
    print(tests_df.to_string(index=False))

    # Violin plots per feature for Oz
    fig, axes = plt.subplots(1, len(FEATURES), figsize=(3.2 * len(FEATURES), 3.4))
    for ax, feat in zip(axes, FEATURES):
        sns.violinplot(data=med[med["channel"] == "Oz"],
                       x="stage", y=feat, order=["Wake", "N2"], ax=ax,
                       inner="box", cut=0)
        ax.set_title(f"Oz · {feat}")
        ax.set_xlabel("")
    save_fig(fig, "bycycle_oz_wake_vs_n2", subdir="group")

    # Save concatenated cycle-level table (large; parquet-style CSV)
    all_df = pd.concat(all_cycles, ignore_index=True)
    all_df.to_csv(cfg.TABLES_DIR / "bycycle_cycles_all.csv.gz",
                  index=False, compression="gzip")


if __name__ == "__main__":
    main()
