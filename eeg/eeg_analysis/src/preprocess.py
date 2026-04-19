"""Preprocessing orchestration: AAS -> resample -> BCG -> filter -> ICA.

Produces two derivative files per run:
* `*_desc-aasbcg_raw.fif` : after MR+BCG correction, no filtering/ICA
* `*_desc-clean_raw.fif`  : filtered, average-referenced, ICA-cleaned
"""
from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np

from eeg_analysis import config as cfg
from eeg_analysis.src import io_eeg, mr_artifact


def _json_default(obj):
    """Coerce numpy scalars / arrays / Paths to JSON-native types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _save_raw(raw: mne.io.BaseRaw, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    raw.save(str(out), overwrite=True, fmt="single", verbose="ERROR")
    return out


def _provenance(run: io_eeg.RunInfo, steps: dict) -> Path:
    out = cfg.subject_deriv_dir(run.subject) / f"{run.tag}_provenance.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"run": run.tag, "steps": steps}, f, indent=2,
                  default=_json_default)
    return out


def preprocess_run(run: io_eeg.RunInfo,
                   overwrite: bool = False
                   ) -> dict:
    """Run the full preprocessing pipeline for one run.

    Returns a dict with output paths and key stats.
    """
    out_aas = cfg.subject_deriv_dir(run.subject) / \
        f"{run.tag}_desc-aasbcg_raw.fif"
    out_clean = cfg.subject_deriv_dir(run.subject) / \
        f"{run.tag}_desc-clean_raw.fif"
    if out_clean.exists() and not overwrite:
        return {"status": "cached", "aas": str(out_aas),
                "clean": str(out_clean)}

    steps: dict = {}

    # 1) Load raw at native 5 kHz
    raw = io_eeg.load_raw(run, preload=True)
    steps["raw_sfreq"] = raw.info["sfreq"]
    steps["raw_duration_s"] = raw.times[-1]

    # 2) AAS
    triggers = io_eeg.scanner_triggers(run)
    steps["n_volume_triggers"] = len(triggers)
    mr_artifact.apply_aas(raw, triggers,
                          window_volumes=cfg.AAS_WINDOW_VOLUMES)
    steps["aas"] = "applied"

    # 3) Resample to 250 Hz after AAS (Allen's recommendation)
    raw.resample(cfg.TARGET_SFREQ, npad="auto", verbose="ERROR")
    steps["resampled_to"] = cfg.TARGET_SFREQ

    # 4) BCG via OBS
    mr_artifact.apply_obs_bcg(raw)
    steps["bcg_obs"] = "applied"

    _save_raw(raw, out_aas)

    # 5) Filter: HPF 0.3, LPF 45, notch 50
    raw.filter(cfg.HPF_HZ, cfg.LPF_HZ, picks="eeg",
               fir_design="firwin", phase="zero", verbose="ERROR")
    if cfg.NOTCH_HZ:
        raw.notch_filter(freqs=list(cfg.NOTCH_HZ), picks="eeg",
                         verbose="ERROR")
    steps["filter_hz"] = [cfg.HPF_HZ, cfg.LPF_HZ]
    steps["notch_hz"] = list(cfg.NOTCH_HZ)

    # 6) Re-reference: average of EEG channels (exclude EOG/ECG)
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    steps["reference"] = "average"

    # 7) Bad channel detection (simple: channel RMS > 5 MAD)
    eeg_picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=eeg_picks)
    rms = np.sqrt((data ** 2).mean(axis=1))
    med = np.median(rms)
    mad = np.median(np.abs(rms - med)) + 1e-18
    z = (rms - med) / (1.4826 * mad)
    bads = [raw.ch_names[i] for i, zi in zip(eeg_picks, z) if abs(zi) > 5]
    if bads:
        raw.info["bads"] = sorted(set(raw.info["bads"] + bads))
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    steps["bad_channels"] = bads

    # 8) ICA — remove ECG / EOG residuals
    ica = mne.preprocessing.ICA(
        n_components=cfg.ICA_N_COMPONENTS,
        method=cfg.ICA_METHOD,
        fit_params=cfg.ICA_FIT_PARAMS,
        random_state=cfg.ICA_RANDOM_STATE,
        max_iter="auto",
        verbose="ERROR",
    )
    ica.fit(raw, picks="eeg", verbose="ERROR")
    bads_eog, _ = ica.find_bads_eog(raw, ch_name=cfg.EOG_CH,
                                    verbose="ERROR") \
        if cfg.EOG_CH in raw.ch_names else ([], None)
    bads_ecg, _ = ica.find_bads_ecg(raw, ch_name=cfg.ECG_CH,
                                    method="correlation",
                                    verbose="ERROR") \
        if cfg.ECG_CH in raw.ch_names else ([], None)
    ica.exclude = sorted(set(bads_eog) | set(bads_ecg))
    steps["ica_excluded"] = {"eog": bads_eog, "ecg": bads_ecg}
    ica.apply(raw, verbose="ERROR")

    _save_raw(raw, out_clean)
    _provenance(run, steps)

    return {"status": "ok",
            "aas": str(out_aas),
            "clean": str(out_clean),
            "steps": steps}
