"""TF-peak detection and SO-power / SO-phase histograms.

Python implementation following Stokes, Dang, Coleman, Prerau (2022)
*Transient oscillations as the basis for sleep oscillation analysis*,
SLEEP. Code logic mirrors the DYNAM-O MATLAB toolbox:

1. Multitaper spectrogram in the spindle/sigma band.
2. Watershed segmentation on the log-power image to extract TF-peaks.
3. Per-peak features: peak freq, peak time, area, duration, bandwidth,
   prominence.
4. Filter peaks by frequency / duration / bandwidth thresholds.
5. Slow-oscillation: 0.3-1.5 Hz bandpass -> Hilbert phase; compute SO
   power (log) in a sliding window normalized to the subject's whole
   N2 distribution (percentile rank).
6. Build SO-power x frequency and SO-phase x frequency histograms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import label as nd_label
from scipy.signal import butter, filtfilt, hilbert
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from eeg_analysis import config as cfg
from eeg_analysis.src.multitaper_spec import multitaper_spectrogram


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _bandpass(sig: np.ndarray, sfreq: float, band: tuple[float, float],
              order: int = 4) -> np.ndarray:
    b, a = butter(order, [band[0] / (sfreq / 2), band[1] / (sfreq / 2)],
                  btype="band")
    return filtfilt(b, a, sig)


@dataclass
class TFPeak:
    time_s: float
    freq_hz: float
    duration_s: float
    bandwidth_hz: float
    prominence: float
    area: float


# ---------------------------------------------------------------------------
# Peak extraction
# ---------------------------------------------------------------------------
def extract_tf_peaks(S: np.ndarray, t: np.ndarray, f: np.ndarray,
                     freq_range: tuple[float, float] = cfg.TFPEAK_FREQ_RANGE,
                     dur_range: tuple[float, float] = cfg.TFPEAK_DUR_RANGE,
                     bw_range: tuple[float, float] = cfg.TFPEAK_BW_RANGE,
                     min_prom_db: float = 1.0
                     ) -> pd.DataFrame:
    """Watershed-based TF-peak extraction on a spectrogram.

    Parameters
    ----------
    S : (n_freqs, n_times) power spectrogram (linear)
    t : time axis (s), f : freq axis (Hz)

    Returns a DataFrame of peak features (one row per peak).
    """
    # Restrict to analysis band
    f_mask = (f >= freq_range[0]) & (f <= freq_range[1])
    if f_mask.sum() < 3:
        raise ValueError("Too few freq bins in analysis band.")
    Sb = S[f_mask]
    fb = f[f_mask]
    logS = 10 * np.log10(Sb + 1e-24)

    # Seed points: local maxima
    coordinates = peak_local_max(logS, min_distance=2,
                                 threshold_abs=logS.mean())
    if coordinates.size == 0:
        return pd.DataFrame(columns=["time_s", "freq_hz", "duration_s",
                                     "bandwidth_hz", "prominence", "area"])
    markers = np.zeros(logS.shape, dtype=int)
    for i, (fi, ti) in enumerate(coordinates, start=1):
        markers[fi, ti] = i

    # Watershed on inverted log-power
    labels = watershed(-logS, markers=markers, mask=logS > np.percentile(logS, 40))

    dt = float(np.median(np.diff(t)))
    df = float(np.median(np.diff(fb)))

    rows = []
    for peak_id in np.unique(labels):
        if peak_id == 0:
            continue
        mask = labels == peak_id
        if mask.sum() < 4:
            continue
        f_idx, t_idx = np.where(mask)
        peak_fi, peak_ti = coordinates[peak_id - 1]
        pf = float(fb[peak_fi])
        pt = float(t[peak_ti])
        dur = float((t_idx.max() - t_idx.min() + 1) * dt)
        bw = float((f_idx.max() - f_idx.min() + 1) * df)
        region_power = logS[mask]
        prom = float(region_power.max() - region_power.min())
        area = float(mask.sum() * dt * df)
        if not (dur_range[0] <= dur <= dur_range[1]):
            continue
        if not (bw_range[0] <= bw <= bw_range[1]):
            continue
        if prom < min_prom_db:
            continue
        rows.append({"time_s": pt, "freq_hz": pf,
                     "duration_s": dur, "bandwidth_hz": bw,
                     "prominence": prom, "area": area})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SO power / phase at peak times
# ---------------------------------------------------------------------------
def so_power_phase(signal: np.ndarray, sfreq: float,
                   peak_times_s: np.ndarray,
                   so_band: tuple[float, float] = cfg.SO_BAND,
                   power_win_s: float = 2.0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return (so_power_pct[n_peaks], so_phase_rad[n_peaks]).

    SO power is computed as 10*log10 of the RMS of the SO-bandpassed
    signal in a `power_win_s` window centered on each peak, then
    converted to within-signal percentile rank (0-100).
    SO phase is the Hilbert-transform phase of the SO-bandpassed
    signal at the peak sample.
    """
    so = _bandpass(signal, sfreq, so_band, order=4)
    analytic = hilbert(so)
    so_phase = np.angle(analytic)
    # Smooth power
    half = int(power_win_s * sfreq / 2)
    power_env = np.convolve(so ** 2, np.ones(2 * half + 1) / (2 * half + 1),
                            mode="same")
    log_power = 10 * np.log10(power_env + 1e-24)

    # Peak -> sample index
    idx = np.clip(np.round(peak_times_s * sfreq).astype(int), 0,
                  signal.size - 1)
    peak_so_db = log_power[idx]
    peak_so_phase = so_phase[idx]

    # Percentile rank of peak SO power w.r.t. full log_power distribution
    srt = np.sort(log_power)
    ranks = np.searchsorted(srt, peak_so_db, side="right") / srt.size * 100.0
    return ranks, peak_so_phase


# ---------------------------------------------------------------------------
# Histogram builders
# ---------------------------------------------------------------------------
def so_power_histogram(peaks: pd.DataFrame, so_power_pct: np.ndarray,
                       freq_range: tuple[float, float] = cfg.TFPEAK_FREQ_RANGE,
                       n_power_bins: int = cfg.SO_POWER_BINS,
                       n_freq_bins: int = cfg.TFPEAK_FREQ_BINS
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (hist[power, freq], power_edges, freq_edges)."""
    power_edges = np.linspace(0, 100, n_power_bins + 1)
    freq_edges = np.linspace(freq_range[0], freq_range[1], n_freq_bins + 1)
    H, _, _ = np.histogram2d(so_power_pct, peaks["freq_hz"].to_numpy(),
                             bins=[power_edges, freq_edges])
    return H, power_edges, freq_edges


def so_phase_histogram(peaks: pd.DataFrame, so_phase_rad: np.ndarray,
                       freq_range: tuple[float, float] = cfg.TFPEAK_FREQ_RANGE,
                       n_phase_bins: int = cfg.SO_PHASE_BINS,
                       n_freq_bins: int = cfg.TFPEAK_FREQ_BINS
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (hist[phase, freq], phase_edges, freq_edges)."""
    phase_edges = np.linspace(-np.pi, np.pi, n_phase_bins + 1)
    freq_edges = np.linspace(freq_range[0], freq_range[1], n_freq_bins + 1)
    H, _, _ = np.histogram2d(so_phase_rad, peaks["freq_hz"].to_numpy(),
                             bins=[phase_edges, freq_edges])
    return H, phase_edges, freq_edges
