"""Per-stage Welch PSD + FOOOF fits (per subject and group)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from tqdm import tqdm

from _common import arg_parser, log_failure, select_subjects
from eeg_analysis import config as cfg
from eeg_analysis.src import io_eeg, psd as psd_mod, fooof_analysis
from eeg_analysis.src.plotting import default_style, save_fig


def _epo(run) -> Path:
    return cfg.subject_deriv_dir(run.subject) / f"{run.tag}_desc-stages_epo.fif"


def _accumulate_subject_psd(subject: str):
    """Return {stage: (sum_psd[ch,freq], n_epochs, freqs, ch_names)}."""
    runs = io_eeg.list_runs(subject)
    agg: dict[str, dict] = {}
    for run in runs:
        fp = _epo(run)
        if not fp.exists():
            continue
        try:
            ep = mne.read_epochs(str(fp), preload=True, verbose="ERROR")
        except Exception:
            continue
        sp = psd_mod.stage_psd(ep)
        for stage, (psd, freqs, chs) in sp.items():
            if stage not in agg:
                agg[stage] = {"sum": np.zeros(psd.shape[1:]),
                              "n": 0, "freqs": freqs, "chs": chs}
            if agg[stage]["chs"] != chs:
                # skip runs with mismatched channels rather than silently
                # averaging them
                continue
            agg[stage]["sum"] += psd.sum(axis=0)
            agg[stage]["n"] += psd.shape[0]
    return agg


def main() -> None:
    args = arg_parser(__doc__).parse_args()
    subjects = select_subjects(args)
    default_style()

    fooof_rows = []
    psd_store: dict[str, dict] = {}  # subject -> stage -> mean psd

    for sub in tqdm(subjects, desc="subjects"):
        agg = _accumulate_subject_psd(sub)
        if not agg:
            log_failure("03_psd_fooof", sub, "No epochs found")
            continue
        psd_store[sub] = {}
        out_npz = cfg.subject_deriv_dir(sub) / f"{sub}_psd_by_stage.npz"
        to_save = {}
        for stage, d in agg.items():
            if d["n"] == 0:
                continue
            mean_psd = d["sum"] / d["n"]
            psd_store[sub][stage] = (mean_psd, d["freqs"], d["chs"])
            to_save[f"{stage}_psd"] = mean_psd
            to_save[f"{stage}_n"] = d["n"]
            # FOOOF
            df = fooof_analysis.fit_all_channels(
                mean_psd, d["freqs"], d["chs"], sub, stage)
            df["n_epochs"] = d["n"]
            fooof_rows.append(df)
        to_save["freqs"] = list(agg.values())[0]["freqs"]
        to_save["channels"] = np.array(list(agg.values())[0]["chs"])
        np.savez(out_npz, **to_save)

        # Per-subject FOOOF figure (key channels)
        try:
            _plot_subject_fooof(sub, psd_store[sub])
        except Exception as e:  # noqa: BLE001
            log_failure("03_psd_fooof", sub, f"plot: {e}")

    if not fooof_rows:
        raise SystemExit("No FOOOF rows produced.")
    fooof_df = pd.concat(fooof_rows, ignore_index=True)
    out_csv = cfg.TABLES_DIR / "fooof_params.csv"
    fooof_df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}  ({len(fooof_df)} rows)")

    _plot_group_fooof(fooof_df)


def _plot_subject_fooof(subject: str, stages: dict) -> None:
    key_chs = [c for c in ("O1", "Oz", "Cz", "Fz") if c in
               list(stages.values())[0][2]]
    if not key_chs:
        return
    fig, axes = plt.subplots(1, len(key_chs), figsize=(3.5 * len(key_chs), 3.2),
                             squeeze=False)
    for j, ch in enumerate(key_chs):
        ax = axes[0, j]
        for stage, (psd, freqs, chs) in stages.items():
            i = chs.index(ch)
            ax.semilogy(freqs, psd[i], color=cfg.STAGE_COLORS.get(stage, "k"),
                        label=stage, lw=1.2)
        ax.set_title(f"{subject} {ch}")
        ax.set_xlabel("Hz"); ax.set_xlim(1, 40)
        if j == 0:
            ax.set_ylabel("PSD (V²/Hz)")
            ax.legend(frameon=False, fontsize=8)
    save_fig(fig, f"{subject}_psd_by_stage", subdir=f"per_subject/{subject}")


def _plot_group_fooof(df: pd.DataFrame) -> None:
    # Group mean exponent & alpha CF at occipital channels
    occ = df[df["channel"].isin(cfg.OCCIPITAL_PRIMARY)]
    summary = occ.groupby(["stage", "channel"]).agg(
        exponent_mean=("exponent", "mean"),
        exponent_sem=("exponent", lambda s: s.sem()),
        alpha_cf_mean=("alpha_cf", "mean"),
        alpha_pw_mean=("alpha_pw", "mean"),
    ).reset_index()
    out = cfg.TABLES_DIR / "fooof_occipital_summary.csv"
    summary.to_csv(out, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    sub = df[df["channel"].isin(cfg.OCCIPITAL_PRIMARY)]
    import seaborn as sns
    sns.boxplot(data=sub, x="stage", y="exponent", hue="channel",
                order=cfg.STAGES, ax=axes[0])
    axes[0].set_title("Aperiodic exponent (occipital)")
    sns.boxplot(data=sub, x="stage", y="alpha_pw", hue="channel",
                order=cfg.STAGES, ax=axes[1])
    axes[1].set_title("Alpha peak power (FOOOF)")
    for a in axes:
        a.legend(frameon=False, fontsize=8)
    save_fig(fig, "group_fooof_occipital", subdir="group")


if __name__ == "__main__":
    main()
