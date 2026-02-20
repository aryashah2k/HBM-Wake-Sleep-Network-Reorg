#!/usr/bin/env python3
"""
02_extract_timeseries.py — ROI Time-Series Extraction

Extracts mean BOLD time-series from:
    - Schaefer 200-parcel cortical atlas (Schaefer et al., 2018)
    - Bilateral thalamus ROIs (FreeSurfer aseg labels 10, 49)

Also assigns per-TR sleep-stage labels and applies confound regression
with bandpass filtering (0.01–0.1 Hz).

Outputs:
    results/timeseries/sub-XX_task-YY_run-ZZ_timeseries.npz
    Each contains:
        - timeseries: (n_rois, n_trs) array
        - roi_labels: list of ROI names
        - stage_labels: per-TR stage labels
        - censor_mask: boolean mask (True = keep)
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, image, maskers, signal
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

warnings.filterwarnings("ignore")


def get_schaefer_atlas():
    """
    Fetch the Schaefer 200-parcel 7-network atlas.

    Reference: Schaefer et al., "Local-Global Parcellation of the Human
    Cerebral Cortex from Intrinsic Functional Connectivity MRI",
    Cerebral Cortex, 2018.
    """
    atlas = datasets.fetch_atlas_schaefer_2018(
        n_rois=config.N_PARCELS,
        yeo_networks=config.N_NETWORKS,
        resolution_mm=config.ATLAS_RESOLUTION_MM,
    )
    return atlas


def create_thalamus_mask(aparc_path):
    """
    Create bilateral thalamus mask from FreeSurfer aparcaseg segmentation.

    Uses aseg labels:
        Left Thalamus = 10
        Right Thalamus = 49

    Returns:
        left_mask_img, right_mask_img: NIfTI images of thalamic ROIs
    """
    aparc_img = nib.load(str(aparc_path))
    aparc_data = aparc_img.get_fdata()

    left_mask = (aparc_data == config.THALAMUS_LABELS["Left-Thalamus"]).astype(np.float32)
    right_mask = (aparc_data == config.THALAMUS_LABELS["Right-Thalamus"]).astype(np.float32)

    left_img = nib.Nifti1Image(left_mask, aparc_img.affine, aparc_img.header)
    right_img = nib.Nifti1Image(right_mask, aparc_img.affine, aparc_img.header)

    return left_img, right_img


def _normalize_stage_label(raw_label):
    """Strip annotations like '(uncertain)' from stage labels."""
    raw = str(raw_label).strip()
    if "(" in raw:
        raw = raw[: raw.index("(")].strip()
    return raw


def parse_sleep_stages_for_run(subject, task, run):
    """
    Parse sleep-stage TSV and return stages for a specific run.

    Robust to malformed rows (missing tab delimiters) and stage
    labels with annotations (e.g. 'W (uncertain)').
    """
    stage_path = config.get_sleep_stage_path(subject)
    if not stage_path.exists():
        return None

    with open(stage_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    if not lines:
        return None

    header = lines[0].split("\t")
    n_expected_cols = len(header)

    # Identify column indices
    session_idx, epoch_idx, stage_idx = 0, 1, 2
    for i, col in enumerate(header):
        col_lower = col.strip().lower()
        if col_lower == "session":
            session_idx = i
        elif "epoch_start" in col_lower:
            epoch_idx = i
        elif "sleep_stage" in col_lower:
            stage_idx = i

    target_session = f"{task}_{run}"
    result = []

    for line_num, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if not line:
            continue

        fields = line.split("\t")

        if len(fields) >= n_expected_cols:
            session_str = fields[session_idx].strip()
            try:
                epoch_val = float(fields[epoch_idx].strip())
            except ValueError:
                continue
            stage_raw = fields[stage_idx].strip()
        else:
            # Malformed row — recover by splitting on whitespace
            all_tokens = line.split()
            session_str = None
            epoch_val = None
            stage_raw = None
            for token in all_tokens:
                if "task-" in token:
                    session_str = token
                elif epoch_val is None and session_str is not None:
                    try:
                        epoch_val = float(token)
                    except ValueError:
                        continue
                elif epoch_val is not None and stage_raw is None:
                    stage_raw = token

            if session_str is None or epoch_val is None or stage_raw is None:
                continue

        if session_str == target_session:
            result.append((epoch_val, _normalize_stage_label(stage_raw)))

    return result if result else None


def map_stages_to_trs(epochs, n_trs):
    """Map 30-sec epoch labels to individual TRs."""
    tr_labels = np.full(n_trs, "Unknown", dtype="U20")

    for epoch_start, stage in epochs:
        first_tr = int(np.floor(epoch_start / config.TR))
        last_tr = int(np.floor((epoch_start + config.EPOCH_DURATION) / config.TR))
        last_tr = min(last_tr, n_trs)

        for tr_idx in range(first_tr, last_tr):
            tr_labels[tr_idx] = stage

    return tr_labels


def extract_timeseries_for_run(subject, task, run, schaefer_atlas):
    """
    Extract ROI time-series for a single run.

    Steps:
        1. Load MNI-space BOLD
        2. Extract Schaefer 200-parcel cortical time-series
        3. Extract bilateral thalamus time-series
        4. Apply confound regression + bandpass filtering
        5. Assign per-TR sleep-stage labels
        6. Create censoring mask

    Returns:
        dict with timeseries, roi_labels, stage_labels, censor_mask
        or None if extraction fails
    """
    bold_path = config.get_bold_path(subject, task, run)
    confounds_path = config.get_confounds_path(subject, task, run)
    aparc_path = config.get_aparc_path(subject, task, run)

    if not all(p.exists() for p in [bold_path, confounds_path, aparc_path]):
        return None

    # Load BOLD
    bold_img = nib.load(str(bold_path))
    n_trs = bold_img.shape[-1]

    # --- Extract cortical time-series (Schaefer 200) ---
    cortical_masker = maskers.NiftiLabelsMasker(
        labels_img=schaefer_atlas.maps,
        labels=schaefer_atlas.labels,
        standardize="zscore_sample",
        detrend=True,
        low_pass=config.LOW_PASS,
        high_pass=config.HIGH_PASS,
        t_r=config.TR,
        memory="nilearn_cache",
        memory_level=1,
    )

    # Load confounds
    confounds_df = pd.read_csv(confounds_path, sep="\t")
    available_confounds = [c for c in config.ALL_CONFOUNDS if c in confounds_df.columns]
    confounds_selected = confounds_df[available_confounds].fillna(0)

    try:
        cortical_ts = cortical_masker.fit_transform(
            str(bold_path),
            confounds=confounds_selected,
        )
    except Exception as e:
        print(f"      Error extracting cortical time-series: {e}")
        return None

    # --- Extract thalamic time-series ---
    left_thal_mask, right_thal_mask = create_thalamus_mask(aparc_path)

    thalamic_ts_list = []
    thalamic_labels = []
    for thal_name, thal_mask in [("Left-Thalamus", left_thal_mask),
                                  ("Right-Thalamus", right_thal_mask)]:
        # Check if thalamus mask has any voxels
        if np.sum(thal_mask.get_fdata()) < 5:
            print(f"      Warning: {thal_name} mask has <5 voxels, skipping")
            continue

        thal_masker = maskers.NiftiMasker(
            mask_img=thal_mask,
            standardize="zscore_sample",
            detrend=True,
            low_pass=config.LOW_PASS,
            high_pass=config.HIGH_PASS,
            t_r=config.TR,
        )

        try:
            thal_ts = thal_masker.fit_transform(
                str(bold_path),
                confounds=confounds_selected,
            )
            # Mean across voxels within thalamus
            thal_mean = thal_ts.mean(axis=1, keepdims=True)
            thalamic_ts_list.append(thal_mean)
            thalamic_labels.append(thal_name)
        except Exception as e:
            print(f"      Error extracting {thal_name} time-series: {e}")

    # Combine cortical + thalamic
    all_ts = [cortical_ts]
    all_labels = list(schaefer_atlas.labels)

    for tts in thalamic_ts_list:
        all_ts.append(tts)
    all_labels.extend(thalamic_labels)

    combined_ts = np.hstack(all_ts)  # (n_trs, n_rois)

    # --- Stage labels ---
    stage_data = parse_sleep_stages_for_run(subject, task, run)
    if stage_data is None:
        stage_labels = np.full(n_trs, "Unknown", dtype="U20")
    else:
        stage_labels = map_stages_to_trs(stage_data, n_trs)

    # --- Censoring mask ---
    if "framewise_displacement" in confounds_df.columns:
        fd = confounds_df["framewise_displacement"].values.copy()
        fd = np.nan_to_num(fd, nan=0.0)
        censor_mask = fd <= config.FD_THRESHOLD
    else:
        censor_mask = np.ones(n_trs, dtype=bool)

    return {
        "timeseries": combined_ts.T,  # (n_rois, n_trs)
        "roi_labels": all_labels,
        "stage_labels": stage_labels,
        "censor_mask": censor_mask,
        "n_cortical_rois": cortical_ts.shape[1],
        "n_thalamic_rois": len(thalamic_labels),
        "thalamic_roi_names": thalamic_labels,
    }


def main():
    print("=" * 70)
    print("SLEEP NETWORK REORGANIZATION — TIME-SERIES EXTRACTION")
    print("=" * 70)

    config.setup_directories()

    # Load QC report
    qc_path = config.QC_DIR / "data_validation_report.json"
    if not qc_path.exists():
        print("ERROR: QC report not found. Run 00_validate_data.py first.")
        sys.exit(1)

    with open(qc_path) as f:
        qc_report = json.load(f)

    included_subjects = qc_report["summary"]["included_subject_ids"]

    # Fetch atlas
    print("\nFetching Schaefer 200-parcel atlas...")
    schaefer_atlas = get_schaefer_atlas()
    print(f"  Atlas loaded: {len(schaefer_atlas.labels)} parcels")

    # Process each subject
    extraction_log = {"subjects": {}}
    total_runs_processed = 0

    for subject in tqdm(included_subjects, desc="Extracting time-series"):
        print(f"\n  Processing {subject}...")

        sub_qc = None
        for s in qc_report["subjects"]:
            if s["subject"] == subject:
                sub_qc = s
                break

        if sub_qc is None:
            continue

        sub_runs_processed = 0
        for run_info in sub_qc.get("runs", []):
            if not run_info["included"]:
                continue

            task = run_info["task"]
            run = run_info["run"]
            print(f"    {task}_{run}...")

            result = extract_timeseries_for_run(subject, task, run, schaefer_atlas)

            if result is None:
                print(f"      Skipped")
                continue

            # Save
            out_path = config.TIMESERIES_DIR / f"{subject}_{task}_{run}_timeseries.npz"
            np.savez_compressed(
                str(out_path),
                timeseries=result["timeseries"],
                roi_labels=np.array(result["roi_labels"], dtype="U100"),
                stage_labels=result["stage_labels"],
                censor_mask=result["censor_mask"],
                n_cortical_rois=result["n_cortical_rois"],
                n_thalamic_rois=result["n_thalamic_rois"],
                thalamic_roi_names=np.array(result.get("thalamic_roi_names", []), dtype="U50"),
            )

            n_rois, n_trs = result["timeseries"].shape
            print(f"      Saved: {n_rois} ROIs × {n_trs} TRs")
            sub_runs_processed += 1
            total_runs_processed += 1

        extraction_log["subjects"][subject] = {"runs_processed": sub_runs_processed}

    # Save log
    extraction_log["total_runs"] = total_runs_processed
    log_path = config.TIMESERIES_DIR / "extraction_log.json"
    with open(log_path, "w") as f:
        json.dump(extraction_log, f, indent=2)

    print(f"\n✓ Time-series extraction complete. {total_runs_processed} runs processed.")
    print(f"  Output: {config.TIMESERIES_DIR}")


if __name__ == "__main__":
    main()
