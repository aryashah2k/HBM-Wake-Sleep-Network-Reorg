"""Multitaper spectrogram — faithful Python implementation of the
Prerau Lab algorithm (Prerau et al., Physiology 2017).

Reference: https://github.com/preraulab/multitaper_toolbox (Python port).

Returns power in units of (V^2 / Hz); callers typically convert to dB.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.signal.windows import dpss

from eeg_analysis import config as cfg


def multitaper_spectrogram(
    signal: np.ndarray,
    sfreq: float,
    frequency_range: Tuple[float, float] = (cfg.MT_FMIN, cfg.MT_FMAX),
    time_bandwidth: float = cfg.MT_TIME_BANDWIDTH,
    num_tapers: int | None = cfg.MT_NUM_TAPERS,
    window_s: float = cfg.MT_WINDOW_S,
    step_s: float = cfg.MT_STEP_S,
    detrend: str = "linear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a multitaper spectrogram.

    Parameters
    ----------
    signal : 1-D array of samples (float)
    sfreq : sampling rate (Hz)
    frequency_range : (fmin, fmax) of interest
    time_bandwidth : NW product (time-half-bandwidth product)
    num_tapers : number of tapers; default = floor(2*NW - 1)
    window_s, step_s : spectrogram window length and hop in seconds

    Returns
    -------
    S : (n_freqs, n_windows) spectrogram, power V^2/Hz
    t : (n_windows,) centered window times (s)
    f : (n_freqs,) frequency axis (Hz)
    """
    signal = np.asarray(signal, dtype=float).ravel()
    n = signal.size
    n_win = int(round(window_s * sfreq))
    n_step = int(round(step_s * sfreq))
    if n_win <= 0 or n_step <= 0:
        raise ValueError("window_s and step_s must yield positive sample counts.")
    if n < n_win:
        raise ValueError("Signal shorter than one window.")
    NW = float(time_bandwidth)
    K = int(num_tapers) if num_tapers is not None else int(np.floor(2 * NW - 1))
    if K < 1:
        raise ValueError("num_tapers must be >= 1.")
    tapers, eigvals = dpss(n_win, NW, Kmax=K, return_ratios=True)
    # FFT length: next pow of 2 for efficiency, but preserve freq resolution
    nfft = int(2 ** np.ceil(np.log2(n_win)))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sfreq)
    f_mask = (freqs >= frequency_range[0]) & (freqs <= frequency_range[1])
    f_out = freqs[f_mask]

    # window start indices
    starts = np.arange(0, n - n_win + 1, n_step, dtype=int)
    S = np.empty((f_out.size, starts.size), dtype=float)

    for wi, start in enumerate(starts):
        seg = signal[start:start + n_win].copy()
        if detrend == "linear":
            # linear detrend
            x = np.arange(n_win)
            p = np.polyfit(x, seg, 1)
            seg = seg - np.polyval(p, x)
        elif detrend == "constant":
            seg = seg - seg.mean()
        # Tapered FFTs
        tapered = tapers * seg                              # (K, n_win)
        Fk = np.fft.rfft(tapered, n=nfft, axis=1)           # (K, nfreq)
        pk = (np.abs(Fk) ** 2) / sfreq                      # power density
        # eigenvalue-weighted average
        w = eigvals / eigvals.sum()
        Sxx = (w[:, None] * pk).sum(axis=0)
        # one-sided: double all non-DC, non-Nyquist bins
        dbl = np.ones_like(Sxx)
        dbl[1:-1] = 2.0
        Sxx = Sxx * dbl
        S[:, wi] = Sxx[f_mask]

    t_out = (starts + n_win / 2) / sfreq
    return S, t_out, f_out
