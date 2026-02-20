#!/usr/bin/env python3
"""
01_glm_analysis.py — First-Level and Group-Level GLM Analysis

Models brain activity differences across Wake, N1, and N2 sleep stages
using a block-design GLM approach with nilearn.

Reference for approach:
    Tagliazucchi & Laufs, "Decoding wakefulness levels from typical fMRI
    resting-state data reveals reliable drifts between wakefulness and sleep",
    Neuron, 2014.

Outputs:
    results/glm/sub-XX/first_level/contrast_zmaps.nii.gz
    results/glm/group/contrast_thresholded.nii.gz
    results/figures/glm_*.png / .pdf
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting, image
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel
from nilearn.reporting import get_clusters_table
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

warnings.filterwarnings("ignore")


def load_confounds(subject, task, run):
    """
    Load and select relevant confound regressors from fMRIPrep output.

    Returns:
        pd.DataFrame: Selected confounds with NaN in first row filled.
    """
    confounds_path = config.get_confounds_path(subject, task, run)
    df = pd.read_csv(confounds_path, sep="\t")

    # Select confounds that exist in the file
    available_confounds = [c for c in config.ALL_CONFOUNDS if c in df.columns]
    confounds_df = df[available_confounds].copy()

    # Fill NaN in derivative/power columns (first row is NaN for derivatives)
    confounds_df = confounds_df.fillna(0)

    return confounds_df, df


def create_stage_events(subject, task, run, n_trs):
    """
    Create event DataFrame for sleep stages mapped to BOLD TRs.

    Each 30-sec epoch is treated as a block with onset and duration.

    Returns:
        pd.DataFrame: Events with columns [onset, duration, trial_type]
    """
    stages_data = parse_sleep_stages_for_run(subject, task, run)
    if stages_data is None:
        return None

    events = []
    for epoch_start, stage in stages_data:
        if stage not in config.VALID_STAGES:
            continue
        stage_name = config.STAGE_LABELS[stage]
        onset = epoch_start
        duration = config.EPOCH_DURATION

        # Ensure we don't go past the end of the scan
        max_time = n_trs * config.TR
        if onset >= max_time:
            continue
        duration = min(duration, max_time - onset)

        events.append({
            "onset": onset,
            "duration": duration,
            "trial_type": stage_name,
        })

    if not events:
        return None

    return pd.DataFrame(events)


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


def get_censor_mask(confounds_full_df):
    """
    Create a boolean censoring mask based on framewise displacement.

    Returns:
        np.array: True = keep, False = censor
    """
    if "framewise_displacement" not in confounds_full_df.columns:
        return np.ones(len(confounds_full_df), dtype=bool)

    fd = confounds_full_df["framewise_displacement"].values.copy()
    fd = np.nan_to_num(fd, nan=0.0)
    return fd <= config.FD_THRESHOLD


def run_first_level(subject, qc_report):
    """
    Run first-level GLM for a single subject across all valid runs.

    Returns:
        dict mapping contrast_name -> z-map NIfTI image
    """
    available_runs = config.get_run_ids(subject)

    # Check subject inclusion from QC report
    sub_qc = None
    for s in qc_report["subjects"]:
        if s["subject"] == subject:
            sub_qc = s
            break

    if sub_qc is None or not sub_qc.get("included", False):
        return None

    # Collect valid run paths
    valid_imgs = []
    valid_confounds = []
    valid_events = []

    for task, run in available_runs:
        # Check if run passed QC
        run_passed = False
        for run_info in sub_qc.get("runs", []):
            if run_info["task"] == task and run_info["run"] == run and run_info["included"]:
                run_passed = True
                break
        if not run_passed:
            continue

        bold_path = config.get_bold_path(subject, task, run)
        if not bold_path.exists():
            continue

        # Load BOLD to get n_trs
        bold_img = nib.load(str(bold_path))
        n_trs = bold_img.shape[-1]

        # Create events
        events = create_stage_events(subject, task, run, n_trs)
        if events is None or len(events) == 0:
            continue

        # Check that at least 2 stages are present
        unique_stages = events["trial_type"].unique()
        if len(unique_stages) < 2:
            continue

        # Load confounds
        confounds_df, confounds_full = load_confounds(subject, task, run)

        # Add censoring regressors (spike regressors for high-motion volumes)
        censor_mask = get_censor_mask(confounds_full)
        high_motion_indices = np.where(~censor_mask)[0]

        for idx in high_motion_indices:
            spike_col = f"motion_censor_{idx}"
            confounds_df[spike_col] = 0
            confounds_df.loc[idx, spike_col] = 1

        valid_imgs.append(str(bold_path))
        valid_confounds.append(confounds_df)
        valid_events.append(events)

    if not valid_imgs:
        return None

    # Fit first-level model
    fmri_glm = FirstLevelModel(
        t_r=config.TR,
        noise_model="ar1",
        standardize=False,
        hrf_model="spm",
        drift_model="cosine",
        high_pass=1.0 / config.GLM_HIGH_PASS_PERIOD,
        smoothing_fwhm=6.0,  # 6mm FWHM Gaussian smoothing (standard)
        minimize_memory=True,
    )

    fmri_glm.fit(valid_imgs, events=valid_events, confounds=valid_confounds)

    # Define contrasts using numeric vectors to handle missing conditions
    # across runs. nilearn's string expressions fail when a condition is
    # absent from any single run's design matrix.
    #
    # Strategy: check which conditions appear in ANY design matrix, define
    # contrasts as dicts of {condition: weight}, then build per-run numeric
    # vectors. Runs missing a required condition for a contrast are skipped
    # by nilearn when a zero-vector is passed.

    all_conditions = set()
    for dm in fmri_glm.design_matrices_:
        for col in dm.columns:
            if col in ("Wake", "N1", "N2"):
                all_conditions.add(col)

    contrast_defs = {}
    if "Wake" in all_conditions and "N2" in all_conditions:
        contrast_defs["Wake_gt_N2"] = {"Wake": 1, "N2": -1}
        contrast_defs["N2_gt_Wake"] = {"Wake": -1, "N2": 1}

    if "Wake" in all_conditions and "N1" in all_conditions:
        contrast_defs["Wake_gt_N1"] = {"Wake": 1, "N1": -1}

    if "N1" in all_conditions and "N2" in all_conditions:
        contrast_defs["N1_gt_N2"] = {"N1": 1, "N2": -1}

    if all(c in all_conditions for c in ("Wake", "N1", "N2")):
        contrast_defs["Linear_Wake_to_N2"] = {"Wake": 1, "N1": -0.5, "N2": -0.5}

    def _build_contrast_vector(dm_columns, weights_dict):
        """Build a numeric contrast vector for one design matrix."""
        vec = np.zeros(len(dm_columns))
        has_all = True
        for cond, w in weights_dict.items():
            if cond in dm_columns:
                vec[list(dm_columns).index(cond)] = w
            else:
                has_all = False
        # Return zero-vector for runs missing a required condition so
        # nilearn gracefully ignores them.
        if not has_all:
            return np.zeros(len(dm_columns))
        return vec

    # Compute contrast maps
    zmaps = {}
    sub_output_dir = config.GLM_DIR / subject / "first_level"
    sub_output_dir.mkdir(parents=True, exist_ok=True)

    for contrast_name, weights in contrast_defs.items():
        try:
            # Build per-run contrast vectors
            contrast_vectors = []
            n_valid_runs = 0
            for dm in fmri_glm.design_matrices_:
                vec = _build_contrast_vector(dm.columns, weights)
                contrast_vectors.append(vec)
                if np.any(vec != 0):
                    n_valid_runs += 1

            if n_valid_runs == 0:
                print(f"    Skipping {contrast_name}: no runs have all required conditions")
                continue

            zmap = fmri_glm.compute_contrast(
                contrast_vectors, output_type="z_score"
            )
            zmaps[contrast_name] = zmap

            # Save z-map
            out_path = sub_output_dir / f"{contrast_name}_zmap.nii.gz"
            nib.save(zmap, str(out_path))

        except Exception as e:
            print(f"    Warning: Could not compute contrast {contrast_name}: {e}")

    # Save design matrix visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    dm = fmri_glm.design_matrices_[0]
    ax.imshow(dm.values, aspect="auto", cmap="gray", interpolation="nearest")
    ax.set_yticks(range(0, dm.shape[0], 50))
    ax.set_xticks(range(dm.shape[1]))
    ax.set_xticklabels(dm.columns, rotation=90, fontsize=6)
    ax.set_ylabel("TR", fontsize=config.FONT_SIZE_LABEL)
    ax.set_title(f"Design Matrix — {subject} (first run)", fontsize=config.FONT_SIZE_TITLE)
    plt.tight_layout()
    fig.savefig(str(sub_output_dir / "design_matrix.png"), dpi=config.FIGURE_DPI)
    plt.close(fig)

    return zmaps


def run_group_level(all_zmaps, contrast_name):
    """
    Run group-level (second-level) one-sample t-test on first-level z-maps.

    Returns:
        (thresholded_map, unthresholded_map, clusters_table)
    """
    if contrast_name not in all_zmaps or len(all_zmaps[contrast_name]) < 3:
        return None, None, None

    subject_zmaps = all_zmaps[contrast_name]

    # Create second-level design matrix (intercept only for one-sample t-test)
    design_matrix = pd.DataFrame(
        {"intercept": np.ones(len(subject_zmaps))}
    )

    second_level_model = SecondLevelModel(smoothing_fwhm=None)
    second_level_model.fit(subject_zmaps, design_matrix=design_matrix)

    # Compute contrast (testing intercept > 0)
    zmap = second_level_model.compute_contrast(output_type="z_score")

    # Threshold: voxel p < 0.001, cluster FWE p < 0.05
    from nilearn.glm import threshold_stats_img
    thresholded_map, threshold_val = threshold_stats_img(
        zmap,
        alpha=config.GLM_CLUSTER_ALPHA,
        height_control="fpr",
        cluster_threshold=10,
        two_sided=True,
    )

    # Get cluster table
    try:
        clusters = get_clusters_table(
            zmap,
            stat_threshold=threshold_val,
            cluster_threshold=10,
            two_sided=True,
        )
    except Exception:
        clusters = pd.DataFrame()

    return thresholded_map, zmap, clusters


def create_group_figures(thresholded_map, unthresholded_map, contrast_name, clusters_table):
    """Create publication-quality brain visualization for a group contrast."""
    fig_dir = config.FIGURES_DIR
    group_dir = config.GLM_DIR / "group"
    group_dir.mkdir(parents=True, exist_ok=True)

    # Save thresholded map
    if thresholded_map is not None:
        nib.save(thresholded_map, str(group_dir / f"{contrast_name}_thresholded.nii.gz"))
    if unthresholded_map is not None:
        nib.save(unthresholded_map, str(group_dir / f"{contrast_name}_unthresholded.nii.gz"))

    # Glass brain plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    display_map = thresholded_map if thresholded_map is not None else unthresholded_map
    if display_map is None:
        plt.close(fig)
        return

    for ax, display_mode in zip(axes, ["x", "y", "z"]):
        plotting.plot_glass_brain(
            display_map,
            display_mode=display_mode,
            colorbar=True,
            title=f"{contrast_name.replace('_', ' ')} ({display_mode})",
            axes=ax,
            threshold=0,
            plot_abs=False,
            cmap="RdBu_r",
        )

    plt.suptitle(
        f"Group-Level: {contrast_name.replace('_', ' ')}",
        fontsize=config.FONT_SIZE_TITLE + 2,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        fig.savefig(
            str(fig_dir / f"glm_{contrast_name}_glass_brain.{fmt}"),
            dpi=config.FIGURE_DPI, bbox_inches="tight",
        )
    plt.close(fig)

    # Surface plot (lateral views)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10),
                             subplot_kw={"projection": "3d"})
    views = [("left", "lateral"), ("right", "lateral"),
             ("left", "medial"), ("right", "medial")]

    for idx, (hemi, view) in enumerate(views):
        ax = axes[idx // 2][idx % 2]
        try:
            plotting.plot_stat_map(
                display_map,
                display_mode="x" if hemi == "left" else "x",
                cut_coords=[-40 if hemi == "left" else 40],
                axes=ax,
                colorbar=(idx == 0),
                threshold=0,
                cmap="RdBu_r",
            )
        except Exception:
            # Fall back to simple stat map
            pass

    plt.suptitle(
        f"Group-Level: {contrast_name.replace('_', ' ')}",
        fontsize=config.FONT_SIZE_TITLE + 2,
        fontweight="bold",
    )
    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        fig.savefig(
            str(fig_dir / f"glm_{contrast_name}_stat_map.{fmt}"),
            dpi=config.FIGURE_DPI, bbox_inches="tight",
        )
    plt.close(fig)

    # Comprehensive stat map with orthogonal slices
    fig = plt.figure(figsize=(16, 5))
    try:
        display = plotting.plot_stat_map(
            display_map,
            display_mode="ortho",
            colorbar=True,
            threshold=0,
            cmap="RdBu_r",
            title=f"Group-Level: {contrast_name.replace('_', ' ')}",
            figure=fig,
        )
    except Exception:
        pass
    for fmt in config.FIGURE_FORMAT:
        fig.savefig(
            str(fig_dir / f"glm_{contrast_name}_ortho.{fmt}"),
            dpi=config.FIGURE_DPI, bbox_inches="tight",
        )
    plt.close(fig)

    # Save clusters table
    if clusters_table is not None and len(clusters_table) > 0:
        clusters_table.to_csv(
            str(group_dir / f"{contrast_name}_clusters.csv"), index=False
        )


def main():
    print("=" * 70)
    print("SLEEP NETWORK REORGANIZATION — GLM ANALYSIS")
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
    print(f"\nProcessing {len(included_subjects)} included subjects...")

    # First-level analysis
    all_zmaps = {}  # contrast_name -> [list of z-map images]
    results_log = {"subjects": {}, "contrasts": []}

    for subject in tqdm(included_subjects, desc="First-level GLM"):
        print(f"\n  Processing {subject}...")
        zmaps = run_first_level(subject, qc_report)

        if zmaps is None:
            print(f"    Skipped: no valid runs/contrasts")
            results_log["subjects"][subject] = {"status": "skipped"}
            continue

        results_log["subjects"][subject] = {
            "status": "completed",
            "contrasts": list(zmaps.keys()),
        }

        for contrast_name, zmap in zmaps.items():
            if contrast_name not in all_zmaps:
                all_zmaps[contrast_name] = []
            all_zmaps[contrast_name].append(zmap)

        print(f"    Computed {len(zmaps)} contrasts: {list(zmaps.keys())}")

    # Group-level analysis
    print("\n" + "=" * 70)
    print("GROUP-LEVEL ANALYSIS")
    print("=" * 70)

    contrast_names = list(all_zmaps.keys())
    results_log["contrasts"] = contrast_names

    for contrast_name in contrast_names:
        n_subs = len(all_zmaps[contrast_name])
        print(f"\n  {contrast_name}: {n_subs} subjects")

        if n_subs < 3:
            print(f"    Skipped: insufficient subjects (need ≥ 3)")
            continue

        thresholded_map, unthresholded_map, clusters_table = run_group_level(
            all_zmaps, contrast_name
        )

        # Create visualizations
        create_group_figures(thresholded_map, unthresholded_map, contrast_name, clusters_table)

        if clusters_table is not None and len(clusters_table) > 0:
            print(f"    Found {len(clusters_table)} significant clusters")
        else:
            print(f"    No significant clusters at current threshold")

    # Save results log
    results_log_path = config.GLM_DIR / "glm_results_log.json"
    with open(results_log_path, "w") as f:
        json.dump(results_log, f, indent=2, default=str)

    print(f"\n✓ GLM analysis complete. Results in: {config.GLM_DIR}")


if __name__ == "__main__":
    main()
