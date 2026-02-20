#!/usr/bin/env python3
"""
06_reliability.py — Test-Retest Reliability Analysis

Computes ICC (Intraclass Correlation Coefficient) and split-half reliability
for key functional connectivity metrics across repeated runs.

References:
    Shrout & Fleiss, "Intraclass correlations: Uses in assessing rater
    reliability", Psychological Bulletin, 1979.

    Noble et al., "A decade of test-retest reliability of functional
    connectivity: A systematic review and meta-analysis", NeuroImage, 2019.

Outputs:
    results/reliability/icc_results.csv
    results/reliability/split_half_results.json
    results/figures/reliability_*.png / .pdf
"""

import json
import sys
import warnings
from pathlib import Path
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

warnings.filterwarnings("ignore")


def compute_within_run_fc(ts_file, n_cortical_only=True):
    """
    Compute static FC matrix for a single run.

    Returns:
        fc_matrix: (n_rois, n_rois) correlation matrix
        subject: subject ID
        task: task name
        run: run name
    """
    data = np.load(str(ts_file), allow_pickle=True)
    timeseries = data["timeseries"]
    censor_mask = data["censor_mask"]
    n_cortical = int(data["n_cortical_rois"])

    parts = ts_file.stem.split("_")
    subject = parts[0]
    task = parts[1] if len(parts) > 1 else ""
    run = parts[2] if len(parts) > 2 else ""

    # Use cortical ROIs only
    if n_cortical_only:
        ts = timeseries[:n_cortical, :]
    else:
        ts = timeseries

    # Censor high-motion TRs
    ts_clean = ts[:, censor_mask.astype(bool)]

    if ts_clean.shape[1] < 30:
        return None

    # Compute FC
    fc = np.corrcoef(ts_clean)
    fc = np.nan_to_num(fc, nan=0.0)
    np.fill_diagonal(fc, 0)

    return {
        "fc": fc,
        "subject": subject,
        "task": task,
        "run": run,
    }


def compute_graph_metrics_from_fc(fc, density=0.15):
    """Compute global efficiency and modularity from FC matrix."""
    try:
        import bct
    except ImportError:
        return {"global_efficiency": np.nan, "modularity": np.nan}

    n = fc.shape[0]
    fc_abs = np.abs(fc)
    np.fill_diagonal(fc_abs, 0)

    threshold_val = np.percentile(
        fc_abs[np.triu_indices(n, k=1)],
        (1 - density) * 100
    )
    thresholded = fc_abs.copy()
    thresholded[thresholded < threshold_val] = 0

    binary_adj = (thresholded > 0).astype(float)

    # Global efficiency
    dist = bct.distance_bin(binary_adj)
    inv_dist = np.zeros_like(dist)
    mask = dist > 0
    inv_dist[mask] = 1.0 / dist[mask]
    np.fill_diagonal(inv_dist, 0)
    ge = inv_dist.sum() / (n * (n - 1))

    # Modularity
    _, q = bct.community_louvain(thresholded, gamma=config.MODULARITY_GAMMA)

    return {
        "global_efficiency": float(ge),
        "modularity": float(q),
    }


def compute_icc(data_df, targets_col, raters_col, values_col):
    """
    Compute ICC(3,1) using pingouin.

    Args:
        data_df: DataFrame with columns for targets, raters, values
        targets_col: column name for subjects (targets)
        raters_col: column name for runs (raters/measurements)
        values_col: column name for the metric value

    Returns:
        dict with ICC value, CI, p-value
    """
    try:
        icc = pg.intraclass_corr(
            data=data_df,
            targets=targets_col,
            raters=raters_col,
            ratings=values_col,
        )

        # Extract ICC3 (two-way mixed, consistency, single measures)
        icc3_row = icc[icc["Type"] == config.ICC_TYPE]
        if len(icc3_row) > 0:
            row = icc3_row.iloc[0]
            return {
                "icc_type": config.ICC_TYPE,
                "icc_value": float(row["ICC"]),
                "ci95_lower": float(row["CI95%"][0]),
                "ci95_upper": float(row["CI95%"][1]),
                "p_value": float(row["pval"]),
                "interpretation": interpret_icc(float(row["ICC"])),
            }
    except Exception as e:
        return {"error": str(e)}

    return {"error": "ICC type not found"}


def interpret_icc(icc_val):
    """
    Interpret ICC value per Cicchetti (1994) guidelines.

    Reference: Cicchetti, "Guidelines, criteria, and rules of thumb for
    evaluating normed and standardized assessment instruments in psychology",
    Psychological Assessment, 1994.
    """
    if icc_val < 0.40:
        return "Poor"
    elif icc_val < 0.60:
        return "Fair"
    elif icc_val < 0.75:
        return "Good"
    else:
        return "Excellent"


def run_icc_analysis(ts_files):
    """
    Compute ICC for global efficiency and modularity across repeated runs.

    Uses runs within the same task as repeated measurements for each subject.

    Returns:
        icc_results: dict
        metrics_df: DataFrame with per-subject per-run metrics
    """
    # Compute FC and graph metrics per run
    all_metrics = []

    for ts_file in tqdm(ts_files, desc="Computing FC per run"):
        result = compute_within_run_fc(ts_file)
        if result is None:
            continue

        graph_metrics = compute_graph_metrics_from_fc(
            result["fc"], config.GRAPH_DENSITY_THRESHOLD
        )

        # Also compute mean FC strength (average correlation)
        upper_tri = result["fc"][np.triu_indices(result["fc"].shape[0], k=1)]
        mean_fc = float(np.mean(upper_tri))

        all_metrics.append({
            "subject": result["subject"],
            "task": result["task"],
            "run": result["run"],
            "global_efficiency": graph_metrics["global_efficiency"],
            "modularity": graph_metrics["modularity"],
            "mean_fc": mean_fc,
        })

    metrics_df = pd.DataFrame(all_metrics)

    # Only subjects with ≥2 runs of the same task
    icc_results = {}
    for task in metrics_df["task"].unique():
        task_data = metrics_df[metrics_df["task"] == task]

        # Count runs per subject
        runs_per_subj = task_data.groupby("subject")["run"].nunique()
        multi_run_subjects = runs_per_subj[runs_per_subj >= 2].index.tolist()

        if len(multi_run_subjects) < 3:
            continue

        task_multi = task_data[task_data["subject"].isin(multi_run_subjects)].copy()

        # Map runs to numerical rater IDs
        task_multi["rater"] = task_multi.groupby("subject").cumcount()

        for metric in ["global_efficiency", "modularity", "mean_fc"]:
            valid = task_multi.dropna(subset=[metric])
            if len(valid) < 6:
                continue

            icc_res = compute_icc(valid, "subject", "rater", metric)
            icc_results[f"{task}_{metric}"] = icc_res

    return icc_results, metrics_df


def run_split_half_reliability(ts_files):
    """
    Compute split-half reliability by splitting time-series within runs.

    Splits each run into odd/even TRs, computes FC for each half,
    then correlates the upper triangles.

    Returns:
        split_half_results: dict with Spearman-Brown corrected correlations
    """
    correlations = []

    for ts_file in tqdm(ts_files, desc="Split-half reliability"):
        data = np.load(str(ts_file), allow_pickle=True)
        timeseries = data["timeseries"]
        censor_mask = data["censor_mask"]
        n_cortical = int(data["n_cortical_rois"])

        ts = timeseries[:n_cortical, :]
        ts_clean = ts[:, censor_mask.astype(bool)]

        if ts_clean.shape[1] < 60:
            continue

        # Split into odd/even TRs
        odd_ts = ts_clean[:, 0::2]
        even_ts = ts_clean[:, 1::2]

        fc_odd = np.corrcoef(odd_ts)
        fc_even = np.corrcoef(even_ts)

        fc_odd = np.nan_to_num(fc_odd, nan=0.0)
        fc_even = np.nan_to_num(fc_even, nan=0.0)

        # Upper triangle correlation
        idx = np.triu_indices(n_cortical, k=1)
        r, _ = stats.pearsonr(fc_odd[idx], fc_even[idx])

        # Spearman-Brown prophecy formula
        sb_corrected = (2 * r) / (1 + r)
        correlations.append(sb_corrected)

    if not correlations:
        return {"error": "No valid data"}

    return {
        "mean_split_half_r": float(np.mean(correlations)),
        "std_split_half_r": float(np.std(correlations)),
        "median_split_half_r": float(np.median(correlations)),
        "n_runs": len(correlations),
        "ci95_lower": float(np.percentile(correlations, 2.5)),
        "ci95_upper": float(np.percentile(correlations, 97.5)),
        "interpretation": interpret_icc(float(np.mean(correlations))),
    }


def create_reliability_figures(icc_results, split_half_results, metrics_df):
    """Generate reliability analysis figures."""
    fig_dir = config.FIGURES_DIR

    # --- Figure 1: ICC values ---
    if icc_results:
        fig, ax = plt.subplots(figsize=(10, 6))

        metrics_with_icc = []
        icc_values = []
        ci_lower = []
        ci_upper = []
        colors = []

        for key, res in icc_results.items():
            if "icc_value" in res:
                metrics_with_icc.append(key.replace("_", "\n"))
                icc_values.append(res["icc_value"])
                ci_lower.append(res["icc_value"] - res["ci95_lower"])
                ci_upper.append(res["ci95_upper"] - res["icc_value"])
                interp = res["interpretation"]
                color_map = {
                    "Excellent": "#4CAF50", "Good": "#8BC34A",
                    "Fair": "#FF9800", "Poor": "#F44336",
                }
                colors.append(color_map.get(interp, "#9E9E9E"))

        x = np.arange(len(metrics_with_icc))
        bars = ax.bar(x, icc_values, color=colors, alpha=0.8,
                      yerr=[ci_lower, ci_upper], capsize=5,
                      edgecolor="white", linewidth=0.5)

        # Reference lines
        ax.axhline(y=0.75, color="green", linestyle="--", linewidth=0.8,
                   alpha=0.5, label="Excellent (≥0.75)")
        ax.axhline(y=0.60, color="orange", linestyle="--", linewidth=0.8,
                   alpha=0.5, label="Good (≥0.60)")
        ax.axhline(y=0.40, color="red", linestyle="--", linewidth=0.8,
                   alpha=0.5, label="Fair (≥0.40)")

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_with_icc, fontsize=config.FONT_SIZE_TICK)
        ax.set_ylabel("ICC(3,1)", fontsize=config.FONT_SIZE_LABEL)
        ax.set_title("Test-Retest Reliability (ICC)",
                    fontsize=config.FONT_SIZE_TITLE, fontweight="bold")
        ax.legend(fontsize=config.FONT_SIZE_LEGEND - 1, loc="upper right")
        ax.set_ylim(-0.1, 1.1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        for fmt in config.FIGURE_FORMAT:
            fig.savefig(str(fig_dir / f"reliability_icc.{fmt}"),
                       dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)

    # --- Figure 2: Split-half scatter ---
    if split_half_results and "mean_split_half_r" in split_half_results:
        fig, ax = plt.subplots(figsize=(8, 6))

        # For visualization, show individual split-half values if we stored them
        mean_r = split_half_results["mean_split_half_r"]
        std_r = split_half_results["std_split_half_r"]
        n = split_half_results["n_runs"]

        # Histogram of split-half correlations
        # (Re-run quickly to get individual values)
        ax.axvline(x=mean_r, color="#2196F3", linewidth=2,
                   label=f"Mean r={mean_r:.3f}")
        ax.axvspan(split_half_results["ci95_lower"],
                   split_half_results["ci95_upper"],
                   alpha=0.2, color="#2196F3", label="95% CI")

        ax.set_xlabel("Spearman-Brown Corrected Correlation",
                      fontsize=config.FONT_SIZE_LABEL)
        ax.set_ylabel("Density", fontsize=config.FONT_SIZE_LABEL)
        ax.set_title("Split-Half Reliability of Functional Connectivity",
                    fontsize=config.FONT_SIZE_TITLE, fontweight="bold")
        ax.legend(fontsize=config.FONT_SIZE_LEGEND)
        ax.set_xlim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add text with interpretation
        ax.text(0.02, 0.95, f"Interpretation: {split_half_results['interpretation']}",
                transform=ax.transAxes, fontsize=config.FONT_SIZE_LABEL,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout()
        for fmt in config.FIGURE_FORMAT:
            fig.savefig(str(fig_dir / f"reliability_split_half.{fmt}"),
                       dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)


def main():
    print("=" * 70)
    print("SLEEP NETWORK REORGANIZATION — TEST-RETEST RELIABILITY")
    print("=" * 70)

    config.setup_directories()

    ts_files = sorted(config.TIMESERIES_DIR.glob("*_timeseries.npz"))
    if not ts_files:
        print("ERROR: No time-series files found. Run 02_extract_timeseries.py first.")
        sys.exit(1)

    print(f"\nFound {len(ts_files)} time-series files")

    # ICC analysis
    print("\nComputing ICC for graph metrics...")
    icc_results, metrics_df = run_icc_analysis(ts_files)

    for key, res in icc_results.items():
        if "icc_value" in res:
            print(f"  {key}: ICC={res['icc_value']:.3f} "
                  f"[{res['ci95_lower']:.3f}, {res['ci95_upper']:.3f}] "
                  f"({res['interpretation']})")

    # Split-half reliability
    print("\nComputing split-half reliability...")
    split_half_results = run_split_half_reliability(ts_files)

    if "mean_split_half_r" in split_half_results:
        print(f"  Mean Spearman-Brown r = {split_half_results['mean_split_half_r']:.3f} "
              f"({split_half_results['interpretation']})")

    # Save results
    icc_df = pd.DataFrame([
        {"metric": k, **v} for k, v in icc_results.items()
    ])
    icc_df.to_csv(str(config.RELIABILITY_DIR / "icc_results.csv"), index=False)

    with open(config.RELIABILITY_DIR / "split_half_results.json", "w") as f:
        json.dump(split_half_results, f, indent=2)

    with open(config.RELIABILITY_DIR / "icc_full_results.json", "w") as f:
        json.dump(icc_results, f, indent=2, default=str)

    # Figures
    print("\nGenerating figures...")
    create_reliability_figures(icc_results, split_half_results, metrics_df)

    print(f"\n✓ Reliability analysis complete. Results: {config.RELIABILITY_DIR}")


if __name__ == "__main__":
    main()
