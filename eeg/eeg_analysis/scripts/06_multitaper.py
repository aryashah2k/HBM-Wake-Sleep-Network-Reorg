"""Per-run multitaper spectrograms (Prerau algorithm, Python port).

Saves a per-run figure (spectrogram + hypnogram) and a per-subject
stage-mean band-power summary.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from tqdm import tqdm

from _common import arg_parser, log_failure, select_subjects
from eeg_analysis import config as cfg
from eeg_analysis.src import io_eeg, staging
from eeg_analysis.src.multitaper_spec import multitaper_spectrogram
from eeg_analysis.src.plotting import default_style, save_fig

CHANNELS = ["Oz", "Cz", "C3", "C4"]
STAGE_CODE = {"Wake": 0, "N1": 1, "N2": 2}


def _clean_path(run) -> Path:
    return cfg.subject_deriv_dir(run.subject) / f"{run.tag}_desc-clean_raw.fif"


def _hypnogram(run, duration_s: float) -> tuple[np.ndarray, np.ndarray]:
    tbl = staging.run_stage_table(run)
    t = tbl["onset_s"].to_numpy(float)
    codes = np.array([STAGE_CODE.get(s, -1) for s in tbl["stage"]])
    mask = (t + cfg.EPOCH_DURATION_S) <= duration_s
    return t[mask], codes[mask]


def _plot_run(run, raw, S_chans, t, f, outdir) -> None:
    fig, axes = plt.subplots(len(CHANNELS) + 1, 1,
                             figsize=(10, 2.1 * (len(CHANNELS) + 1)),
                             sharex=True,
                             gridspec_kw={"height_ratios": [0.8] +
                                          [1] * len(CHANNELS)})
    # hypnogram
    hx, hy = _hypnogram(run, raw.times[-1])
    axes[0].step(hx, hy, where="post", color="black", lw=1)
    axes[0].set_yticks([0, 1, 2]); axes[0].set_yticklabels(["W", "N1", "N2"])
    axes[0].set_ylim(-0.5, 2.5); axes[0].set_title(run.tag)
    for ax, (ch, S) in zip(axes[1:], S_chans.items()):
        im = ax.pcolormesh(t, f, 10 * np.log10(S + 1e-24),
                            shading="auto", cmap="magma",
                            vmin=np.percentile(10 * np.log10(S + 1e-24), 5),
                            vmax=np.percentile(10 * np.log10(S + 1e-24), 99))
        ax.set_ylabel(f"{ch}  Hz")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="dB")
    axes[-1].set_xlabel("Time (s)")
    save_fig(fig, f"{run.tag}_multitaper",
             subdir=f"per_subject/{run.subject}")


def main() -> None:
    args = arg_parser(__doc__).parse_args()
    subjects = select_subjects(args)
    default_style()

    band_rows = []
    for sub in tqdm(subjects, desc="subjects"):
        for run in io_eeg.list_runs(sub):
            fp = _clean_path(run)
            if not fp.exists():
                continue
            try:
                raw = mne.io.read_raw_fif(str(fp), preload=True,
                                          verbose="ERROR")
            except Exception as e:  # noqa: BLE001
                log_failure("06_multitaper", run.tag, str(e))
                continue
            sfreq = raw.info["sfreq"]
            S_by_ch = {}
            f_axis = t_axis = None
            for ch in CHANNELS:
                if ch not in raw.ch_names:
                    continue
                sig = raw.get_data(picks=[ch])[0]
                try:
                    S, t, f = multitaper_spectrogram(sig, sfreq)
                except Exception as e:  # noqa: BLE001
                    log_failure("06_multitaper", f"{run.tag}:{ch}", str(e))
                    continue
                S_by_ch[ch] = S
                f_axis, t_axis = f, t
                # Stage-aligned band power
                tbl = staging.run_stage_table(run)
                for _, row in tbl.iterrows():
                    t0 = float(row["onset_s"])
                    t1 = t0 + cfg.EPOCH_DURATION_S
                    wmask = (t >= t0) & (t < t1)
                    if wmask.sum() == 0:
                        continue
                    for band_name, (lo, hi) in cfg.BANDS.items():
                        fmask = (f >= lo) & (f <= hi)
                        if fmask.sum() < 2:
                            continue
                        bp = np.trapz(S[fmask][:, wmask], f[fmask], axis=0).mean()
                        band_rows.append({
                            "subject": sub, "run": run.tag, "channel": ch,
                            "stage": row["stage"], "band": band_name,
                            "power": float(bp),
                        })
            if S_by_ch:
                _plot_run(run, raw, S_by_ch, t_axis, f_axis, None)

    if band_rows:
        bp_df = pd.DataFrame(band_rows)
        bp_df.to_csv(cfg.TABLES_DIR / "multitaper_band_power.csv.gz",
                     index=False, compression="gzip")
        # Group stage mean
        summ = bp_df.groupby(["channel", "band", "stage"])["power"] \
            .agg(["mean", "median", "std", "count"]).reset_index()
        summ.to_csv(cfg.TABLES_DIR / "multitaper_band_power_summary.csv",
                    index=False)
        print("Wrote multitaper_band_power{_summary}.csv")


if __name__ == "__main__":
    main()
