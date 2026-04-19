"""DYNAM-O TF-peak extraction + SO-power / SO-phase histograms (N2).

Implements the Stokes et al. (2022) pipeline: multitaper spectrogram ->
watershed TF-peak segmentation -> SO power/phase histograms.

Processes channels C3, C4, Cz on runs that contain any N2 time.
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
from eeg_analysis.src.dynamo_tfpeaks import (
    extract_tf_peaks, so_power_phase,
    so_power_histogram, so_phase_histogram,
)
from eeg_analysis.src.plotting import default_style, save_fig


CHANNELS = cfg.CENTRAL_SPINDLE


def _clean_path(run) -> Path:
    return cfg.subject_deriv_dir(run.subject) / f"{run.tag}_desc-clean_raw.fif"


def _n2_bouts(run) -> list[tuple[float, float]]:
    """Return list of (t_start_s, t_end_s) for contiguous N2 segments."""
    tbl = staging.run_stage_table(run).sort_values("onset_s").reset_index(drop=True)
    bouts: list[tuple[float, float]] = []
    cur_start = None
    cur_end = None
    for _, row in tbl.iterrows():
        if row["stage"] != "N2":
            if cur_start is not None:
                bouts.append((cur_start, cur_end))
                cur_start = cur_end = None
            continue
        t0 = float(row["onset_s"])
        t1 = t0 + cfg.EPOCH_DURATION_S
        if cur_start is None:
            cur_start, cur_end = t0, t1
        elif abs(t0 - cur_end) < 1e-6:
            cur_end = t1
        else:
            bouts.append((cur_start, cur_end))
            cur_start, cur_end = t0, t1
    if cur_start is not None:
        bouts.append((cur_start, cur_end))
    return bouts


def main() -> None:
    args = arg_parser(__doc__).parse_args()
    subjects = select_subjects(args)
    default_style()

    peak_rows = []
    group_so_power = {ch: None for ch in CHANNELS}
    group_so_phase = {ch: None for ch in CHANNELS}
    group_counts = {ch: 0 for ch in CHANNELS}
    power_edges = phase_edges = freq_edges = None

    for sub in tqdm(subjects, desc="subjects"):
        sub_hist_power = {ch: None for ch in CHANNELS}
        sub_hist_phase = {ch: None for ch in CHANNELS}
        for run in io_eeg.list_runs(sub):
            fp = _clean_path(run)
            if not fp.exists():
                continue
            try:
                raw = mne.io.read_raw_fif(str(fp), preload=True,
                                          verbose="ERROR")
            except Exception as e:  # noqa: BLE001
                log_failure("07_dynamo_tfpeaks", run.tag, str(e))
                continue
            sfreq = raw.info["sfreq"]
            bouts = _n2_bouts(run)
            if not bouts:
                continue
            for ch in CHANNELS:
                if ch not in raw.ch_names:
                    continue
                sig = raw.get_data(picks=[ch])[0]
                # Extract peaks from each contiguous N2 bout at fine
                # time resolution (Stokes et al. 2022 parameters).
                run_peaks = []
                for (t0, t1) in bouts:
                    i0 = max(0, int(round(t0 * sfreq)))
                    i1 = min(sig.size, int(round(t1 * sfreq)))
                    if (i1 - i0) < int(cfg.MT_WINDOW_S_TFPEAK * sfreq) * 2:
                        continue
                    seg = sig[i0:i1]
                    try:
                        S, t_loc, f = multitaper_spectrogram(
                            seg, sfreq,
                            frequency_range=(0.3, 20.0),
                            time_bandwidth=cfg.MT_NW_TFPEAK,
                            num_tapers=cfg.MT_K_TFPEAK,
                            window_s=cfg.MT_WINDOW_S_TFPEAK,
                            step_s=cfg.MT_STEP_S_TFPEAK)
                    except Exception as e:  # noqa: BLE001
                        log_failure("07_dynamo_tfpeaks",
                                    f"{run.tag}:{ch}:MT", str(e))
                        continue
                    try:
                        pk = extract_tf_peaks(S, t_loc, f)
                    except Exception as e:  # noqa: BLE001
                        log_failure("07_dynamo_tfpeaks",
                                    f"{run.tag}:{ch}:peaks", str(e))
                        continue
                    if pk.empty:
                        continue
                    pk = pk.assign(time_s=pk["time_s"] + t0)
                    run_peaks.append(pk)
                if not run_peaks:
                    continue
                peaks = pd.concat(run_peaks, ignore_index=True)
                try:
                    so_p, so_ph = so_power_phase(sig, sfreq,
                                                 peaks["time_s"].to_numpy())
                except Exception as e:  # noqa: BLE001
                    log_failure("07_dynamo_tfpeaks",
                                f"{run.tag}:{ch}:so", str(e))
                    continue
                peaks = peaks.assign(so_power_pct=so_p,
                                     so_phase_rad=so_ph,
                                     subject=sub, run=run.tag, channel=ch)
                peak_rows.append(peaks)

                Hp, p_edges, f_edges = so_power_histogram(peaks, so_p)
                Hph, ph_edges, _ = so_phase_histogram(peaks, so_ph)
                power_edges, phase_edges, freq_edges = p_edges, ph_edges, f_edges

                sub_hist_power[ch] = Hp if sub_hist_power[ch] is None \
                    else sub_hist_power[ch] + Hp
                sub_hist_phase[ch] = Hph if sub_hist_phase[ch] is None \
                    else sub_hist_phase[ch] + Hph

        # Save per-subject histograms + figure
        any_data = False
        for ch in CHANNELS:
            if sub_hist_power[ch] is None:
                continue
            any_data = True
            group_so_power[ch] = sub_hist_power[ch] if group_so_power[ch] is None \
                else group_so_power[ch] + sub_hist_power[ch]
            group_so_phase[ch] = sub_hist_phase[ch] if group_so_phase[ch] is None \
                else group_so_phase[ch] + sub_hist_phase[ch]
            group_counts[ch] += 1
        if any_data:
            np.savez(cfg.subject_deriv_dir(sub) / f"{sub}_so_histograms.npz",
                     **{f"power_{ch}": sub_hist_power[ch]
                        for ch in CHANNELS if sub_hist_power[ch] is not None},
                     **{f"phase_{ch}": sub_hist_phase[ch]
                        for ch in CHANNELS if sub_hist_phase[ch] is not None},
                     power_edges=power_edges, phase_edges=phase_edges,
                     freq_edges=freq_edges)
            _plot_subject_hists(sub, sub_hist_power, sub_hist_phase,
                                 power_edges, phase_edges, freq_edges)

    if peak_rows:
        peaks_all = pd.concat(peak_rows, ignore_index=True)
        peaks_all.to_csv(cfg.TABLES_DIR / "tfpeaks_all.csv.gz",
                        index=False, compression="gzip")
    if any(v is not None for v in group_so_power.values()):
        np.savez(cfg.TABLES_DIR / "group_so_histograms.npz",
                 **{f"power_{ch}": group_so_power[ch] for ch in CHANNELS
                    if group_so_power[ch] is not None},
                 **{f"phase_{ch}": group_so_phase[ch] for ch in CHANNELS
                    if group_so_phase[ch] is not None},
                 power_edges=power_edges, phase_edges=phase_edges,
                 freq_edges=freq_edges,
                 counts=np.array([group_counts[ch] for ch in CHANNELS]))
        _plot_group_hists(group_so_power, group_so_phase,
                          power_edges, phase_edges, freq_edges)


def _plot_subject_hists(sub, Hp, Hph, p_edges, ph_edges, f_edges):
    fig, axes = plt.subplots(2, len(CHANNELS), figsize=(3.5 * len(CHANNELS), 6),
                             squeeze=False)
    for j, ch in enumerate(CHANNELS):
        ax = axes[0, j]
        if Hp[ch] is not None:
            ax.pcolormesh(f_edges, p_edges, Hp[ch], cmap="viridis",
                          shading="auto")
        ax.set_title(f"{sub} {ch} · SO-power × freq")
        ax.set_xlabel("Freq (Hz)"); ax.set_ylabel("SO power %")
        ax = axes[1, j]
        if Hph[ch] is not None:
            ax.pcolormesh(f_edges, ph_edges, Hph[ch], cmap="twilight",
                          shading="auto")
        ax.set_title(f"{sub} {ch} · SO-phase × freq")
        ax.set_xlabel("Freq (Hz)"); ax.set_ylabel("SO phase (rad)")
    save_fig(fig, f"{sub}_so_histograms", subdir=f"per_subject/{sub}")


def _plot_group_hists(Hp, Hph, p_edges, ph_edges, f_edges):
    fig, axes = plt.subplots(2, len(CHANNELS), figsize=(3.5 * len(CHANNELS), 6),
                             squeeze=False)
    for j, ch in enumerate(CHANNELS):
        ax = axes[0, j]
        if Hp[ch] is not None:
            ax.pcolormesh(f_edges, p_edges, Hp[ch], cmap="viridis",
                          shading="auto")
        ax.set_title(f"Group {ch} · SO-power × freq")
        ax.set_xlabel("Freq (Hz)"); ax.set_ylabel("SO power %")
        ax = axes[1, j]
        if Hph[ch] is not None:
            ax.pcolormesh(f_edges, ph_edges, Hph[ch], cmap="twilight",
                          shading="auto")
        ax.set_title(f"Group {ch} · SO-phase × freq")
        ax.set_xlabel("Freq (Hz)"); ax.set_ylabel("SO phase (rad)")
    save_fig(fig, "group_so_histograms", subdir="group")


if __name__ == "__main__":
    main()
