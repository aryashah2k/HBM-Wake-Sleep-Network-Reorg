"""Sleep-stage handling: TSV -> per-run stage timeline and MNE epochs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import mne

from eeg_analysis import config as cfg
from eeg_analysis.src import io_eeg


def load_stage_tsv(subject: str) -> pd.DataFrame:
    num = int(subject.split("-")[1])
    tsv = cfg.SOURCEDATA_DIR / f"sub-{num:02d}-sleep-stage.tsv"
    if not tsv.is_file():
        raise FileNotFoundError(f"Sleep-stage TSV missing: {tsv}")
    df = pd.read_csv(tsv, sep="\t", dtype={"30-sec_epoch_sleep_stage": str})
    df.columns = [c.strip() for c in df.columns]
    # `subject` column is optional (schema varies across files); we
    # already know the subject from the filename.
    required = {"session", "epoch_start_time_sec",
                "30-sec_epoch_sleep_stage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"TSV {tsv} missing columns: {missing}")
    if "subject" not in df.columns:
        df["subject"] = num
    df = df.rename(columns={"30-sec_epoch_sleep_stage": "stage_raw",
                            "epoch_start_time_sec": "onset_s"})
    # Normalize labels like "W (uncertain)" -> first token "W" before
    # mapping. Keep the full raw label for downstream QC.
    df["stage_token"] = df["stage_raw"].astype(str).str.strip() \
        .str.split(r"\s+", n=1).str[0]
    df["stage_uncertain"] = df["stage_raw"].astype(str) \
        .str.contains("uncertain", case=False, na=False)
    df["stage"] = df["stage_token"].map(lambda s: cfg.STAGE_MAP.get(s))
    return df


def run_stage_table(run: io_eeg.RunInfo) -> pd.DataFrame:
    """Return the sub-table for this specific run (session)."""
    df = load_stage_tsv(run.subject)
    sub = df[df["session"] == run.session_label].copy()
    sub = sub.dropna(subset=["stage"]).reset_index(drop=True)
    return sub


def build_annotations(run: io_eeg.RunInfo,
                      raw: mne.io.BaseRaw) -> mne.Annotations:
    tbl = run_stage_table(run)
    dur = cfg.EPOCH_DURATION_S
    onsets, durations, descs = [], [], []
    for _, row in tbl.iterrows():
        onset = float(row["onset_s"])
        if onset + dur > raw.times[-1]:
            break
        onsets.append(onset)
        durations.append(dur)
        descs.append(row["stage"])
    return mne.Annotations(onset=onsets, duration=durations,
                           description=descs,
                           orig_time=raw.info["meas_date"])


def make_stage_epochs(run: io_eeg.RunInfo,
                      raw: mne.io.BaseRaw) -> mne.Epochs:
    """Create 30-s, non-overlapping stage epochs."""
    ann = build_annotations(run, raw)
    raw = raw.copy().set_annotations(raw.annotations + ann)
    events, event_id = mne.events_from_annotations(
        raw, event_id={"Wake": 1, "N1": 2, "N2": 3},
        regexp=None, verbose="ERROR")
    if len(events) == 0:
        raise RuntimeError(f"No stage events for {run.tag}")
    # Build an Epochs object with the PTP threshold applied only to
    # reliable scalp channels (mastoids excluded — they frequently show
    # residual BCG or electrode-pop artifact that would kill otherwise
    # clean epochs). We do this by temporarily marking excluded channels
    # as "bad" so MNE's reject dict ignores them, then restoring.
    prev_bads = list(raw.info["bads"])
    exclude = [c for c in cfg.EPOCH_REJECT_EXCLUDE if c in raw.ch_names
               and c not in prev_bads]
    raw.info["bads"] = sorted(set(prev_bads + exclude))
    epochs = mne.Epochs(
        raw, events, event_id=event_id, tmin=0.0,
        tmax=cfg.EPOCH_DURATION_S - 1 / raw.info["sfreq"],
        baseline=None, preload=True, reject_by_annotation=True,
        reject={"eeg": cfg.EPOCH_PTP_UV},
        flat=None, verbose="ERROR",
    )
    # Restore bad-channel list on both raw and epochs
    raw.info["bads"] = prev_bads
    epochs.info["bads"] = prev_bads
    return epochs


def stage_counts(run: io_eeg.RunInfo) -> Dict[str, int]:
    tbl = run_stage_table(run)
    return tbl["stage"].value_counts().to_dict()
