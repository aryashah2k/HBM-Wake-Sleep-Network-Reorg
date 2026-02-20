#!/usr/bin/env python3
"""
00_validate_data.py — Data Validation and Quality Control

Validates all preprocessed fMRIPrep data and sleep-staging files for the
Sleep Network Reorganization Analysis Pipeline.

Outputs:
    results/qc/data_validation_report.json
    results/qc/stage_distribution.csv
    results/qc/motion_summary.csv
    results/figures/qc_stage_distribution.png
    results/figures/qc_motion_summary.png
"""

import json
import sys
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

warnings.filterwarnings("ignore", category=FutureWarning)


def _normalize_stage_label(raw_label):
    """
    Normalize a sleep-stage label by stripping annotations.

    Handles: 'W', 'W (uncertain)', '1', '1 (uncertain)', '2',
             'Unscorable', etc.

    Returns the core label: 'W', '1', '2', or the original string
    for unrecognised labels (e.g. 'Unscorable').
    """
    raw = str(raw_label).strip()
    # Remove (uncertain) or similar parenthetical annotations
    if "(" in raw:
        raw = raw[: raw.index("(")].strip()
    return raw


def parse_sleep_stages(subject):
    """
    Parse sleep-stage TSV for a subject.

    Robust to:
        - Malformed rows where a tab delimiter is missing (e.g. '810 W')
        - Stage labels with annotations (e.g. 'W (uncertain)')

    Returns:
        dict: {(task, run): [(epoch_start_sec, stage), ...]}
    """
    stage_path = config.get_sleep_stage_path(subject)
    if not stage_path.exists():
        return None

    # --- Read raw lines to handle malformed rows gracefully ---
    with open(stage_path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    if not lines:
        return None

    # Detect header
    header = lines[0].split("\t")
    n_expected_cols = len(header)

    # Identify column indices by name
    session_idx = None
    epoch_idx = None
    stage_idx = None
    for i, col in enumerate(header):
        col_lower = col.strip().lower()
        if col_lower in ("session",):
            session_idx = i
        elif "epoch_start" in col_lower:
            epoch_idx = i
        elif "sleep_stage" in col_lower:
            stage_idx = i

    # Fallback: first column is session, second is epoch, third is stage
    if session_idx is None:
        session_idx = 0
    if epoch_idx is None:
        epoch_idx = 1
    if stage_idx is None:
        stage_idx = 2

    runs = {}

    for line_num, line in enumerate(lines[1:], start=2):
        line = line.strip()
        if not line:
            continue

        fields = line.split("\t")

        if len(fields) >= n_expected_cols:
            # Normal row
            session_str = fields[session_idx].strip()
            try:
                epoch_start = float(fields[epoch_idx].strip())
            except ValueError:
                print(f"    WARNING ({subject}): skipping line {line_num}, "
                      f"cannot parse epoch_start: '{fields[epoch_idx]}'")
                continue
            stage_raw = fields[stage_idx].strip()
        else:
            # Malformed row — try to recover by splitting on whitespace.
            # Common case: '810 W' where a tab between epoch and stage is
            # replaced by a space.
            all_tokens = line.split()
            # First token(s) should be the session, then a number, then stage
            # Heuristic: find the session field (contains 'task-')
            session_str = None
            epoch_start = None
            stage_raw = None

            for t_idx, token in enumerate(all_tokens):
                if "task-" in token:
                    session_str = token
                elif epoch_start is None and session_str is not None:
                    try:
                        epoch_start = float(token)
                    except ValueError:
                        continue
                elif epoch_start is not None and stage_raw is None:
                    stage_raw = token

            if session_str is None or epoch_start is None or stage_raw is None:
                print(f"    WARNING ({subject}): skipping malformed line {line_num}: "
                      f"'{line}'")
                continue

            print(f"    INFO ({subject}): recovered malformed line {line_num}: "
                  f"session={session_str}, epoch={epoch_start}, stage={stage_raw}")

        # Parse task and run from session string (e.g., "task-rest_run-1")
        parts = session_str.split("_")
        task = parts[0]  # e.g., "task-rest"
        run = parts[1] if len(parts) > 1 else "run-1"  # e.g., "run-1"

        key = (task, run)
        if key not in runs:
            runs[key] = []

        stage = _normalize_stage_label(stage_raw)
        runs[key].append((epoch_start, stage))

    return runs


def map_stages_to_trs(epochs, n_trs):
    """
    Map 30-sec epoch labels to individual TRs.

    Args:
        epochs: list of (epoch_start_sec, stage_label)
        n_trs: total number of TRs in the run

    Returns:
        np.array of stage labels per TR (string array)
    """
    tr_labels = np.full(n_trs, "Unknown", dtype="U20")

    for epoch_start, stage in epochs:
        # First TR index in this epoch
        first_tr = int(np.floor(epoch_start / config.TR))
        # Last TR index in this epoch (exclusive)
        last_tr = int(np.floor((epoch_start + config.EPOCH_DURATION) / config.TR))
        last_tr = min(last_tr, n_trs)

        for tr_idx in range(first_tr, last_tr):
            tr_labels[tr_idx] = stage

    return tr_labels


def validate_subject(subject):
    """
    Validate a single subject's data.

    Returns:
        dict with validation results
    """
    result = {
        "subject": subject,
        "runs": [],
        "included": True,
        "exclusion_reasons": [],
    }

    # Check sleep stages
    stages_data = parse_sleep_stages(subject)
    if stages_data is None:
        result["included"] = False
        result["exclusion_reasons"].append("Missing sleep-stage TSV")
        return result

    # Check available BOLD runs
    available_runs = config.get_run_ids(subject)
    if not available_runs:
        result["included"] = False
        result["exclusion_reasons"].append("No preprocessed BOLD files found")
        return result

    total_valid_trs = {"W": 0, "1": 0, "2": 0}

    for task, run in available_runs:
        run_info = {
            "task": task,
            "run": run,
            "included": True,
            "exclusion_reasons": [],
        }

        # Check BOLD file
        bold_path = config.get_bold_path(subject, task, run)
        if not bold_path.exists():
            run_info["included"] = False
            run_info["exclusion_reasons"].append("BOLD file missing")
            result["runs"].append(run_info)
            continue

        # Check confounds
        confounds_path = config.get_confounds_path(subject, task, run)
        if not confounds_path.exists():
            run_info["included"] = False
            run_info["exclusion_reasons"].append("Confounds file missing")
            result["runs"].append(run_info)
            continue

        # Check brain mask
        mask_path = config.get_brain_mask_path(subject, task, run)
        if not mask_path.exists():
            run_info["included"] = False
            run_info["exclusion_reasons"].append("Brain mask missing")
            result["runs"].append(run_info)
            continue

        # Load confounds to check motion
        confounds_df = pd.read_csv(confounds_path, sep="\t")
        n_trs = len(confounds_df)
        run_info["n_trs"] = n_trs

        # Check FD for motion scrubbing
        if "framewise_displacement" in confounds_df.columns:
            fd = confounds_df["framewise_displacement"].values
            fd[0] = 0  # First volume has no FD (NaN in fMRIPrep)
            fd = np.nan_to_num(fd, nan=0.0)

            n_censored = np.sum(fd > config.FD_THRESHOLD)
            censored_fraction = n_censored / n_trs
            run_info["mean_fd"] = float(np.mean(fd))
            run_info["max_fd"] = float(np.max(fd))
            run_info["n_censored"] = int(n_censored)
            run_info["censored_fraction"] = float(censored_fraction)

            if censored_fraction > config.MAX_CENSORED_FRACTION:
                run_info["included"] = False
                run_info["exclusion_reasons"].append(
                    f"Excessive motion: {censored_fraction:.1%} volumes censored "
                    f"(threshold: {config.MAX_CENSORED_FRACTION:.0%})"
                )

            # Binary censoring mask: True = keep, False = censor
            censor_mask = fd <= config.FD_THRESHOLD
        else:
            run_info["included"] = False
            run_info["exclusion_reasons"].append("FD column missing in confounds")
            result["runs"].append(run_info)
            continue

        # Check required confound columns exist
        missing_confounds = [
            c for c in config.ALL_CONFOUNDS if c not in confounds_df.columns
        ]
        if missing_confounds:
            run_info["missing_confounds"] = missing_confounds
            # Not a hard exclusion — we can proceed with available confounds
            run_info["exclusion_reasons"].append(
                f"Missing confounds: {missing_confounds[:5]}..."
            )

        # Check sleep staging for this run
        key = (task, run)
        if key not in stages_data:
            run_info["included"] = False
            run_info["exclusion_reasons"].append("No sleep-stage data for this run")
            result["runs"].append(run_info)
            continue

        # Map stages to TRs
        tr_labels = map_stages_to_trs(stages_data[key], n_trs)
        stage_counts = {}
        stage_clean_counts = {}
        for stage in config.VALID_STAGES:
            stage_mask = tr_labels == stage
            stage_counts[stage] = int(np.sum(stage_mask))

            # Count only low-motion TRs per stage
            clean_mask = stage_mask & censor_mask
            stage_clean_counts[stage] = int(np.sum(clean_mask))

        run_info["stage_tr_counts"] = stage_counts
        run_info["stage_clean_tr_counts"] = stage_clean_counts

        # Check minimum TRs per stage (only for valid/included stages)
        insufficient_stages = [
            s for s, c in stage_clean_counts.items()
            if c < config.MIN_TRS_PER_STAGE and c > 0
        ]
        if insufficient_stages:
            run_info["insufficient_stages"] = insufficient_stages

        # Aggregate valid TRs
        if run_info["included"]:
            for s in config.VALID_STAGES:
                total_valid_trs[s] += stage_clean_counts.get(s, 0)

        result["runs"].append(run_info)

    # Subject-level exclusion: check if any stage has zero valid TRs
    included_runs = [r for r in result["runs"] if r["included"]]
    if not included_runs:
        result["included"] = False
        result["exclusion_reasons"].append("No valid runs after QC")
    else:
        result["n_included_runs"] = len(included_runs)
        result["total_valid_trs"] = total_valid_trs

        # Check minimum representation across stages
        stages_with_data = [s for s, c in total_valid_trs.items() if c >= config.MIN_TRS_PER_STAGE]
        result["stages_with_sufficient_data"] = stages_with_data

        if len(stages_with_data) < 2:
            result["included"] = False
            result["exclusion_reasons"].append(
                f"Insufficient stage coverage: only {stages_with_data} have enough data"
            )

    return result


def create_stage_distribution_plot(report, output_path):
    """Create a publication-quality plot of stage distribution across subjects."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Collect data
    subjects = []
    wake_counts = []
    n1_counts = []
    n2_counts = []

    for sub_result in report["subjects"]:
        if not sub_result["included"]:
            continue
        sub = sub_result["subject"]
        subjects.append(sub)
        tvt = sub_result.get("total_valid_trs", {})
        wake_counts.append(tvt.get("W", 0))
        n1_counts.append(tvt.get("1", 0))
        n2_counts.append(tvt.get("2", 0))

    if not subjects:
        plt.close(fig)
        return

    x = np.arange(len(subjects))
    width = 0.6

    # Stacked bar
    ax = axes[0]
    ax.bar(x, wake_counts, width, label="Wake", color=config.STAGE_COLORS["Wake"])
    ax.bar(x, n1_counts, width, bottom=wake_counts, label="N1",
           color=config.STAGE_COLORS["N1"])
    bottoms = [w + n for w, n in zip(wake_counts, n1_counts)]
    ax.bar(x, n2_counts, width, bottom=bottoms, label="N2",
           color=config.STAGE_COLORS["N2"])
    ax.set_xlabel("Subject", fontsize=config.FONT_SIZE_LABEL)
    ax.set_ylabel("Number of valid TRs", fontsize=config.FONT_SIZE_LABEL)
    ax.set_title("Sleep Stage Distribution per Subject", fontsize=config.FONT_SIZE_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("sub-", "") for s in subjects],
                       rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=config.FONT_SIZE_LEGEND)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Proportional bar
    ax = axes[1]
    totals = [w + n1 + n2 for w, n1, n2 in zip(wake_counts, n1_counts, n2_counts)]
    totals = [max(t, 1) for t in totals]  # avoid division by zero
    wake_frac = [w / t for w, t in zip(wake_counts, totals)]
    n1_frac = [n / t for n, t in zip(n1_counts, totals)]
    n2_frac = [n / t for n, t in zip(n2_counts, totals)]

    ax.bar(x, wake_frac, width, label="Wake", color=config.STAGE_COLORS["Wake"])
    ax.bar(x, n1_frac, width, bottom=wake_frac, label="N1",
           color=config.STAGE_COLORS["N1"])
    bottoms_frac = [w + n for w, n in zip(wake_frac, n1_frac)]
    ax.bar(x, n2_frac, width, bottom=bottoms_frac, label="N2",
           color=config.STAGE_COLORS["N2"])
    ax.set_xlabel("Subject", fontsize=config.FONT_SIZE_LABEL)
    ax.set_ylabel("Proportion of TRs", fontsize=config.FONT_SIZE_LABEL)
    ax.set_title("Stage Proportions per Subject", fontsize=config.FONT_SIZE_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("sub-", "") for s in subjects],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=config.FONT_SIZE_LEGEND)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        fig.savefig(str(output_path).replace(".png", f".{fmt}"), dpi=config.FIGURE_DPI)
    plt.close(fig)


def create_motion_summary_plot(report, output_path):
    """Create a publication-quality plot of motion statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Collect data
    mean_fds = []
    censored_fracs = []
    subjects_list = []
    included_list = []

    for sub_result in report["subjects"]:
        sub = sub_result["subject"]
        for run_info in sub_result["runs"]:
            if "mean_fd" in run_info:
                subjects_list.append(sub)
                mean_fds.append(run_info["mean_fd"])
                censored_fracs.append(run_info.get("censored_fraction", 0))
                included_list.append(run_info["included"])

    if not subjects_list:
        plt.close(fig)
        return

    df_motion = pd.DataFrame({
        "Subject": subjects_list,
        "Mean FD (mm)": mean_fds,
        "Censored Fraction": censored_fracs,
        "Included": included_list,
    })

    # Mean FD distribution
    ax = axes[0]
    colors = ["#4CAF50" if inc else "#F44336" for inc in df_motion["Included"]]
    ax.scatter(range(len(df_motion)), df_motion["Mean FD (mm)"], c=colors, alpha=0.6, s=20)
    ax.axhline(y=config.FD_THRESHOLD, color="red", linestyle="--", linewidth=1,
               label=f"FD threshold = {config.FD_THRESHOLD} mm")
    ax.set_xlabel("Run index", fontsize=config.FONT_SIZE_LABEL)
    ax.set_ylabel("Mean Framewise Displacement (mm)", fontsize=config.FONT_SIZE_LABEL)
    ax.set_title("Motion Quality: Mean FD per Run", fontsize=config.FONT_SIZE_TITLE)
    ax.legend(fontsize=config.FONT_SIZE_LEGEND)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Censored fraction distribution
    ax = axes[1]
    ax.hist(df_motion["Censored Fraction"], bins=20, color="#2196F3", alpha=0.7,
            edgecolor="white")
    ax.axvline(x=config.MAX_CENSORED_FRACTION, color="red", linestyle="--", linewidth=1,
               label=f"Exclusion threshold = {config.MAX_CENSORED_FRACTION:.0%}")
    ax.set_xlabel("Fraction of Censored Volumes", fontsize=config.FONT_SIZE_LABEL)
    ax.set_ylabel("Number of Runs", fontsize=config.FONT_SIZE_LABEL)
    ax.set_title("Volume Censoring Distribution", fontsize=config.FONT_SIZE_TITLE)
    ax.legend(fontsize=config.FONT_SIZE_LEGEND)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        fig.savefig(str(output_path).replace(".png", f".{fmt}"), dpi=config.FIGURE_DPI)
    plt.close(fig)


def main():
    print("=" * 70)
    print("SLEEP NETWORK REORGANIZATION — DATA VALIDATION & QC")
    print("=" * 70)

    config.setup_directories()

    report = {
        "subjects": [],
        "summary": {},
    }

    included_subjects = []
    excluded_subjects = []

    for subject in config.SUBJECTS:
        print(f"\nValidating {subject}...")
        sub_result = validate_subject(subject)
        report["subjects"].append(sub_result)

        if sub_result["included"]:
            included_subjects.append(subject)
            n_runs = sub_result.get("n_included_runs", 0)
            tvt = sub_result.get("total_valid_trs", {})
            print(f"  ✓ INCLUDED — {n_runs} valid runs | "
                  f"W={tvt.get('W', 0)} N1={tvt.get('1', 0)} N2={tvt.get('2', 0)} TRs")
        else:
            excluded_subjects.append(subject)
            reasons = sub_result.get("exclusion_reasons", ["Unknown"])
            print(f"  ✗ EXCLUDED — {'; '.join(reasons)}")

    # Summary
    report["summary"] = {
        "total_subjects": len(config.SUBJECTS),
        "included_subjects": len(included_subjects),
        "excluded_subjects": len(excluded_subjects),
        "included_subject_ids": included_subjects,
        "excluded_subject_ids": excluded_subjects,
    }

    # Aggregate stage counts
    total_w, total_n1, total_n2 = 0, 0, 0
    for sub_result in report["subjects"]:
        if sub_result["included"]:
            tvt = sub_result.get("total_valid_trs", {})
            total_w += tvt.get("W", 0)
            total_n1 += tvt.get("1", 0)
            total_n2 += tvt.get("2", 0)

    report["summary"]["total_valid_trs"] = {
        "Wake": total_w,
        "N1": total_n1,
        "N2": total_n2,
        "Total": total_w + total_n1 + total_n2,
    }

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Included: {len(included_subjects)}/{len(config.SUBJECTS)} subjects")
    print(f"Excluded: {len(excluded_subjects)} subjects — {excluded_subjects}")
    print(f"\nTotal valid TRs across included subjects:")
    print(f"  Wake: {total_w} ({total_w * config.TR / 60:.1f} min)")
    print(f"  N1:   {total_n1} ({total_n1 * config.TR / 60:.1f} min)")
    print(f"  N2:   {total_n2} ({total_n2 * config.TR / 60:.1f} min)")
    print(f"  Total: {total_w + total_n1 + total_n2} "
          f"({(total_w + total_n1 + total_n2) * config.TR / 60:.1f} min)")

    # Save report
    report_path = config.QC_DIR / "data_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to: {report_path}")

    # Create visualizations
    stage_plot_path = config.FIGURES_DIR / "qc_stage_distribution.png"
    create_stage_distribution_plot(report, stage_plot_path)
    print(f"Stage distribution plot saved to: {stage_plot_path}")

    motion_plot_path = config.FIGURES_DIR / "qc_motion_summary.png"
    create_motion_summary_plot(report, motion_plot_path)
    print(f"Motion summary plot saved to: {motion_plot_path}")

    # Save stage distribution CSV
    stage_rows = []
    for sub_result in report["subjects"]:
        if sub_result["included"]:
            tvt = sub_result.get("total_valid_trs", {})
            stage_rows.append({
                "subject": sub_result["subject"],
                "Wake_TRs": tvt.get("W", 0),
                "N1_TRs": tvt.get("1", 0),
                "N2_TRs": tvt.get("2", 0),
            })
    if stage_rows:
        pd.DataFrame(stage_rows).to_csv(
            config.QC_DIR / "stage_distribution.csv", index=False
        )

    print("\n✓ Data validation complete.")
    return report


if __name__ == "__main__":
    main()
