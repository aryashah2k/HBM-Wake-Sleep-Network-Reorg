"""FOOOF / specparam fitting on stage-averaged PSDs."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from fooof import FOOOF
except ImportError as e:  # noqa: F401
    raise

from eeg_analysis import config as cfg


def fit_fooof(psd: np.ndarray, freqs: np.ndarray) -> dict:
    """Fit a FOOOF model on a single 1-D PSD; return param dict."""
    fm = FOOOF(peak_width_limits=cfg.FOOOF_PEAK_WIDTH_LIMITS,
               max_n_peaks=cfg.FOOOF_MAX_N_PEAKS,
               min_peak_height=cfg.FOOOF_MIN_PEAK_HEIGHT,
               peak_threshold=cfg.FOOOF_PEAK_THRESHOLD,
               aperiodic_mode=cfg.FOOOF_APERIODIC_MODE,
               verbose=False)
    fm.fit(freqs, psd, cfg.FOOOF_FREQ_RANGE)
    ap = fm.aperiodic_params_  # offset, (knee), exponent
    if cfg.FOOOF_APERIODIC_MODE == "fixed":
        offset, exponent = float(ap[0]), float(ap[1])
        knee = np.nan
    else:
        offset, knee, exponent = float(ap[0]), float(ap[1]), float(ap[2])
    peaks = fm.peak_params_  # each row: CF, PW, BW
    r2 = float(fm.r_squared_)
    err = float(fm.error_)
    return {
        "offset": offset, "knee": knee, "exponent": exponent,
        "r2": r2, "error": err,
        "peaks": [(float(cf), float(pw), float(bw)) for cf, pw, bw in peaks],
    }


def band_peak(peaks, band: Tuple[float, float]) -> dict:
    """Return the tallest peak inside band, or NaNs."""
    lo, hi = band
    in_band = [p for p in peaks if lo <= p[0] <= hi]
    if not in_band:
        return {"cf": np.nan, "pw": np.nan, "bw": np.nan}
    p = max(in_band, key=lambda q: q[1])
    return {"cf": p[0], "pw": p[1], "bw": p[2]}


def fit_all_channels(mean_psd: np.ndarray, freqs: np.ndarray,
                     ch_names: List[str], subject: str, stage: str
                     ) -> pd.DataFrame:
    """Fit FOOOF per channel on stage-averaged PSD."""
    rows = []
    for i, ch in enumerate(ch_names):
        try:
            f = fit_fooof(mean_psd[i], freqs)
        except Exception as e:
            rows.append({"subject": subject, "stage": stage, "channel": ch,
                         "offset": np.nan, "exponent": np.nan, "r2": np.nan,
                         "alpha_cf": np.nan, "alpha_pw": np.nan,
                         "alpha_bw": np.nan, "fit_error": str(e)})
            continue
        alpha = band_peak(f["peaks"], cfg.BANDS["alpha"])
        sigma = band_peak(f["peaks"], cfg.BANDS["sigma"])
        theta = band_peak(f["peaks"], cfg.BANDS["theta"])
        delta = band_peak(f["peaks"], cfg.BANDS["delta"])
        rows.append({
            "subject": subject, "stage": stage, "channel": ch,
            "offset": f["offset"], "exponent": f["exponent"],
            "r2": f["r2"], "fit_error": "",
            "alpha_cf": alpha["cf"], "alpha_pw": alpha["pw"],
            "alpha_bw": alpha["bw"],
            "sigma_cf": sigma["cf"], "sigma_pw": sigma["pw"],
            "theta_cf": theta["cf"], "theta_pw": theta["pw"],
            "delta_cf": delta["cf"], "delta_pw": delta["pw"],
        })
    return pd.DataFrame(rows)
