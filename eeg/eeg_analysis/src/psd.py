"""Welch PSD computation per stage per channel for a preprocessed run."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import mne

from eeg_analysis import config as cfg


def stage_psd(epochs: mne.Epochs, fmin: float = cfg.PSD_FMIN,
              fmax: float = cfg.PSD_FMAX,
              win_sec: float = cfg.WELCH_WIN_SEC,
              overlap: float = cfg.WELCH_OVERLAP
              ) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    """Return {stage: (psd[epochs, ch, freq], freqs, ch_names)}."""
    sfreq = epochs.info["sfreq"]
    n_fft = int(win_sec * sfreq)
    n_overlap = int(n_fft * overlap)
    out: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {}
    for stage in cfg.STAGES:
        if stage not in epochs.event_id:
            continue
        sel = epochs[stage]
        if len(sel) == 0:
            continue
        spec = sel.compute_psd(method="welch", fmin=fmin, fmax=fmax,
                               n_fft=n_fft, n_overlap=n_overlap,
                               picks="eeg", verbose="ERROR")
        psd = spec.get_data()           # (n_epochs, n_ch, n_freq)
        freqs = spec.freqs
        ch_names = spec.ch_names
        out[stage] = (psd, freqs, ch_names)
    return out


def band_power(psd: np.ndarray, freqs: np.ndarray,
               band: Tuple[float, float]) -> np.ndarray:
    """Integrate PSD over the band along the last axis -> return same
    shape minus freq axis."""
    lo, hi = band
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() < 2:
        raise ValueError(f"Band {band} not represented in freqs.")
    return np.trapz(psd[..., mask], freqs[mask], axis=-1)
