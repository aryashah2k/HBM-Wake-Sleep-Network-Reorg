"""Occipital alpha (8-13 Hz) Wake vs N2 analysis."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import mne

from eeg_analysis import config as cfg
from eeg_analysis.src import psd as psd_mod


def per_subject_alpha(epochs: mne.Epochs,
                      channels: List[str] = cfg.OCCIPITAL_ALL
                      ) -> pd.DataFrame:
    """For a subject's concatenated stage-epochs, return median alpha
    power per stage per channel (10*log10 uV^2/Hz)."""
    stage_psds = psd_mod.stage_psd(epochs)
    rows: List[dict] = []
    for stage, (psd, freqs, ch_names) in stage_psds.items():
        # Channel index map
        idx_map = {ch: i for i, ch in enumerate(ch_names)}
        bp = psd_mod.band_power(psd, freqs, cfg.ALPHA_BAND)
        # mean PSD across epochs, in dB
        mean_psd_db = 10 * np.log10(psd.mean(axis=0) + 1e-24)
        for ch in channels:
            if ch not in idx_map:
                continue
            i = idx_map[ch]
            rows.append({
                "stage": stage, "channel": ch,
                "alpha_power": float(bp[:, i].mean()),
                "alpha_power_db": float(10 * np.log10(bp[:, i].mean() + 1e-24)),
                "n_epochs": int(psd.shape[0]),
            })
    return pd.DataFrame(rows)
