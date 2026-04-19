"""MR-gradient (Allen 1998 AAS) and BCG (Niazy OBS) artifact correction.

These implementations follow the canonical algorithms:

* Allen PJ, Josephs O, Turner R (2000) *A method for removing imaging
  artifact from continuous EEG recorded during functional MRI*,
  NeuroImage 12:230-239.
* Niazy RK et al (2005) *Removal of FMRI environment artifacts from EEG
  data using optimal basis sets*, NeuroImage 28:720-737.

No silent fallbacks: if triggers are missing or irregular the routine
raises. Callers log the run as failed in `logs/failures.csv`.
"""
from __future__ import annotations

import numpy as np
import mne
from scipy.signal import find_peaks

from eeg_analysis import config as cfg


# ---------------------------------------------------------------------------
# Gradient artifact subtraction (AAS)
# ---------------------------------------------------------------------------
def _regularize_triggers(triggers: np.ndarray, tol_frac: float = 0.02
                         ) -> tuple[np.ndarray, int]:
    """Return (clean_triggers, modal_spacing) after dropping outliers.

    Keeps only triggers whose spacing to the previous kept trigger is
    within `tol_frac` (default 2%) of the modal diff. Raises if fewer
    than 10 regular triggers remain, or if < 80% of triggers survive.
    """
    if triggers.size < 2:
        raise ValueError("Need >= 2 volume triggers for AAS.")
    diffs = np.diff(triggers)
    modal = int(np.median(diffs))
    tol = max(2, int(tol_frac * modal))
    kept = [int(triggers[0])]
    for t in triggers[1:]:
        if abs(int(t) - kept[-1] - modal) <= tol:
            kept.append(int(t))
    kept_arr = np.asarray(kept, dtype=int)
    survived = kept_arr.size / triggers.size
    if kept_arr.size < 10 or survived < 0.8:
        raise ValueError(
            f"Could not regularize volume triggers: "
            f"modal={modal}, kept={kept_arr.size}/{triggers.size} "
            f"({survived:.0%}).")
    return kept_arr, modal


def apply_aas(raw: mne.io.BaseRaw, triggers: list[int],
              window_volumes: int = cfg.AAS_WINDOW_VOLUMES
              ) -> mne.io.BaseRaw:
    """Remove MR gradient artifact via sliding-window template subtraction.

    Parameters
    ----------
    raw : mne.io.BaseRaw (preloaded)
    triggers : list[int] sample indices of volume onsets
    window_volumes : number of consecutive volumes in the sliding template
    """
    if not raw.preload:
        raw.load_data()
    trig_raw = np.asarray(sorted(set(int(t) for t in triggers)), dtype=int)
    if trig_raw.size < window_volumes + 1:
        raise ValueError(
            f"Need at least {window_volumes + 1} triggers for AAS; "
            f"got {trig_raw.size}.")
    trig, tpl_len = _regularize_triggers(trig_raw)
    n_trig = trig.size
    # Preloaded raw data; edit in place via ._data (documented contract).
    data = raw._data                            # (n_ch, n_samp), V
    n_samp = data.shape[1]

    # Clip triggers whose window would run past end of recording
    valid = (trig + tpl_len) <= n_samp
    trig = trig[valid]
    n_trig = trig.size
    if n_trig < window_volumes + 1:
        raise ValueError(
            f"After clipping, only {n_trig} usable triggers; need >= "
            f"{window_volumes + 1}.")

    # Stack all per-volume segments: shape (n_ch, n_trig, tpl_len)
    # Memory-aware: built lazily channel-by-channel if very large.
    picks = mne.pick_types(raw.info, eeg=True, ecg=True, eog=True)
    half = window_volumes // 2
    for ch in picks:
        segs = np.empty((n_trig, tpl_len), dtype=np.float64)
        for i, t in enumerate(trig):
            segs[i] = data[ch, t:t + tpl_len]
        # Sliding-window template: for each volume i, template is the mean
        # of the `window_volumes` volumes centered on i (clamped at edges).
        templates = np.empty_like(segs)
        for i in range(n_trig):
            lo = max(0, i - half)
            hi = min(n_trig, lo + window_volumes)
            lo = max(0, hi - window_volumes)
            templates[i] = segs[lo:hi].mean(axis=0)
        # Subtract
        for i, t in enumerate(trig):
            data[ch, t:t + tpl_len] -= templates[i]

    # Pre-scan and post-scan periods contain no gradient artifact
    # (scanner off), so they are left as-is and usable for analysis.
    return raw


# ---------------------------------------------------------------------------
# Ballistocardiogram (BCG) correction via Optimal Basis Set (Niazy 2005)
# ---------------------------------------------------------------------------
def _detect_r_peaks(ecg: np.ndarray, sfreq: float) -> np.ndarray:
    """Return sample indices of R-peaks from a 1-D ECG signal."""
    # Band-pass 5-15 Hz (Pan-Tompkins-style) and rectify
    from scipy.signal import butter, filtfilt
    b, a = butter(3, [5 / (sfreq / 2), 15 / (sfreq / 2)], btype="band")
    filt = filtfilt(b, a, ecg)
    env = np.abs(filt)
    min_dist = int(0.4 * sfreq)                  # 150 bpm ceiling
    height = np.percentile(env, 98) * 0.3
    peaks, _ = find_peaks(env, distance=min_dist, height=height)
    return peaks


def apply_obs_bcg(raw: mne.io.BaseRaw, n_basis: int = 4,
                  win_pre_s: float = 0.21, win_post_s: float = 0.49
                  ) -> mne.io.BaseRaw:
    """Remove BCG via Optimal Basis Set regression (Niazy 2005).

    For each EEG channel, peri-R-peak windows are collected; the first
    `n_basis` principal components span the artifact subspace and are
    regressed out of every window.
    """
    if cfg.ECG_CH not in raw.ch_names:
        raise RuntimeError("ECG channel absent; cannot run BCG correction.")
    if not raw.preload:
        raw.load_data()
    sfreq = raw.info["sfreq"]
    ecg_idx = raw.ch_names.index(cfg.ECG_CH)
    ecg = raw._data[ecg_idx].copy()
    r_peaks = _detect_r_peaks(ecg, sfreq)
    if r_peaks.size < 20:
        raise RuntimeError(
            f"Only {r_peaks.size} R-peaks detected; BCG correction aborted.")

    pre = int(win_pre_s * sfreq)
    post = int(win_post_s * sfreq)
    win = pre + post
    data = raw._data
    n_samp = data.shape[1]
    valid = (r_peaks - pre >= 0) & (r_peaks + post <= n_samp)
    r_peaks = r_peaks[valid]
    if r_peaks.size < 20:
        raise RuntimeError("Too few in-bound R-peaks after clipping.")

    picks = mne.pick_types(raw.info, eeg=True)
    for ch in picks:
        segs = np.stack([data[ch, p - pre:p + post] for p in r_peaks],
                        axis=0)                  # (n_peaks, win)
        # Demean per segment
        segs_dm = segs - segs.mean(axis=1, keepdims=True)
        # SVD -> principal basis set
        U, S, Vt = np.linalg.svd(segs_dm, full_matrices=False)
        basis = Vt[:n_basis]                     # (n_basis, win)
        # Project each segment onto basis and subtract
        proj = segs @ basis.T @ basis            # (n_peaks, win)
        cleaned = segs - proj
        for i, p in enumerate(r_peaks):
            data[ch, p - pre:p + post] = cleaned[i]
    return raw
