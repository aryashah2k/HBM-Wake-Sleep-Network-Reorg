"""Cycle-by-cycle alpha waveform shape (bycycle)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
import mne

try:
    from bycycle.features import compute_features
except ImportError:  # pragma: no cover
    raise

from eeg_analysis import config as cfg


FEATURES_OF_INTEREST = [
    "time_rdsym", "time_ptsym",
    "volt_amp", "period", "band_amp",
]


def cycle_features_run(epochs: mne.Epochs, stage: str,
                       channels: List[str] = cfg.OCCIPITAL_PRIMARY
                       ) -> pd.DataFrame:
    """Concatenate epochs of `stage`, compute bycycle features on each
    channel, return tidy DataFrame of burst-classified cycles only."""
    if stage not in epochs.event_id:
        return pd.DataFrame()
    ep = epochs[stage]
    if len(ep) == 0:
        return pd.DataFrame()
    sfreq = ep.info["sfreq"]
    data = ep.get_data(picks=channels)  # (n_ep, n_ch, n_samp)
    n_ep, n_ch, n_samp = data.shape

    rows: List[pd.DataFrame] = []
    for ci, ch in enumerate(channels):
        # Concatenate epochs to a single trace for this channel
        sig = data[:, ci, :].reshape(-1)
        try:
            df = compute_features(
                sig, sfreq, cfg.BYCYCLE_FREQ,
                center_extrema="peak",
                burst_method="cycles",
                threshold_kwargs=cfg.BYCYCLE_THRESHOLDS,
                return_samples=False,
            )
        except Exception as e:  # noqa: BLE001
            # Hard fail rather than silent skip: record the error row
            rows.append(pd.DataFrame([{
                "channel": ch, "stage": stage, "error": str(e),
            }]))
            continue
        df = df[df["is_burst"] == True].copy()  # noqa: E712
        df["channel"] = ch
        df["stage"] = stage
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
