"""Wake vs N2 occipital alpha test (paired Wilcoxon, FDR across channels)."""
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
from eeg_analysis.src import io_eeg, alpha_occipital
from eeg_analysis.src import stats as stats_mod
from eeg_analysis.src.plotting import default_style, save_fig


def _epo(run) -> Path:
    return cfg.subject_deriv_dir(run.subject) / f"{run.tag}_desc-stages_epo.fif"


def _concat_subject_epochs(subject: str) -> mne.Epochs | None:
    eps = []
    for run in io_eeg.list_runs(subject):
        fp = _epo(run)
        if fp.exists():
            try:
                eps.append(mne.read_epochs(str(fp), preload=True,
                                           verbose="ERROR"))
            except Exception:
                continue
    if not eps:
        return None
    return mne.concatenate_epochs(eps, add_offset=True)


def main() -> None:
    args = arg_parser(__doc__).parse_args()
    subjects = select_subjects(args)
    default_style()

    rows = []
    for sub in tqdm(subjects, desc="subjects"):
        ep = _concat_subject_epochs(sub)
        if ep is None:
            log_failure("04_alpha_wake_vs_n2", sub, "No epochs")
            continue
        df = alpha_occipital.per_subject_alpha(ep)
        df["subject"] = sub
        rows.append(df)

    if not rows:
        raise SystemExit("No alpha data.")
    long_df = pd.concat(rows, ignore_index=True)
    long_df.to_csv(cfg.TABLES_DIR / "alpha_power_by_stage.csv", index=False)

    # Paired Wake vs N2 per channel
    wide = long_df.pivot_table(index=["subject", "channel"], columns="stage",
                               values="alpha_power_db").reset_index()
    tests = []
    for ch in cfg.OCCIPITAL_ALL:
        sub = wide[wide["channel"] == ch].dropna(subset=["Wake", "N2"])
        if sub.empty:
            continue
        res = stats_mod.wilcoxon_paired(sub["Wake"].to_numpy(),
                                        sub["N2"].to_numpy())
        res.update({"channel": ch,
                    "Wake_median_dB": float(sub["Wake"].median()),
                    "N2_median_dB": float(sub["N2"].median())})
        tests.append(res)
    stat_df = pd.DataFrame(tests)
    stat_df["q_fdr"] = stats_mod.fdr_bh(stat_df["p"].values)
    stat_df = stat_df[["channel", "n", "Wake_median_dB", "N2_median_dB",
                       "W", "p", "q_fdr", "hedges_g"]]
    stat_df.to_csv(cfg.TABLES_DIR / "alpha_wake_vs_n2_stats.csv", index=False)
    print(stat_df.to_string(index=False))

    # Group figure: per-subject lines for O1/O2/Oz
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    for ax, ch in zip(axes, cfg.OCCIPITAL_PRIMARY):
        sub = wide[wide["channel"] == ch].dropna(subset=["Wake", "N2"])
        for _, r in sub.iterrows():
            ax.plot(["Wake", "N2"], [r["Wake"], r["N2"]],
                    color="grey", alpha=0.4, lw=0.8)
        sns.pointplot(data=long_df[(long_df["channel"] == ch) &
                                   (long_df["stage"].isin(["Wake", "N2"]))],
                      x="stage", y="alpha_power_db", order=["Wake", "N2"],
                      ax=ax, color="black", errorbar=("ci", 95))
        ax.set_title(f"{ch}  (α 8–13 Hz)")
        ax.set_xlabel(""); ax.set_ylabel("10·log10 power")
    save_fig(fig, "alpha_wake_vs_n2_occipital", subdir="group")


if __name__ == "__main__":
    main()
