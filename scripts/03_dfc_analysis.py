#!/usr/bin/env python3
"""
03_dfc_analysis.py — Dynamic Functional Connectivity Analysis

Computes sliding-window functional connectivity matrices and derives
graph-theoretic metrics (Global Efficiency, Modularity) across
Wake, N1, and N2 sleep stages.

Features:
    - Per-subject checkpointing: each subject's results saved immediately
    - Resume capability: skips already-processed subjects on re-run
    - Memory-efficient: processes one subject at a time, streams FC stats
    - Intermediate saves: metrics CSV updated after each subject

References:
    Leonardi & Van De Ville, "On spurious and real fluctuations of dynamic
    functional connectivity during rest", NeuroImage, 2015.

    Rubinov & Sporns, "Complex network measures of brain connectivity:
    Uses and interpretations", NeuroImage, 2010.

Outputs:
    results/dfc/per_subject/sub-XX_run-YY_metrics.csv   (per-run checkpoint)
    results/dfc/graph_metrics_per_window.csv             (combined)
    results/dfc/mean_fc_per_stage.npz
    results/dfc/stage_comparison_stats.json
    results/figures/dfc_*.png / .pdf
"""

import gc
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

warnings.filterwarnings("ignore")

# Import graph theory tools
try:
    import bct
except ImportError:
    print("WARNING: bctpy not installed. Install with: pip install bctpy")
    sys.exit(1)

# Checkpoint directory
CHECKPOINT_DIR = config.DFC_DIR / "per_subject"


def compute_sliding_window_fc(timeseries, window_size, step_size):
    """
    Compute sliding-window Pearson correlation matrices.

    Args:
        timeseries: (n_rois, n_trs) array
        window_size: number of TRs per window
        step_size: stride in TRs

    Returns:
        fc_matrices: (n_windows, n_rois, n_rois) array
        window_centers: center TR index for each window
    """
    n_rois, n_trs = timeseries.shape
    n_windows = (n_trs - window_size) // step_size + 1

    if n_windows <= 0:
        return None, None

    fc_matrices = np.zeros((n_windows, n_rois, n_rois), dtype=np.float32)
    window_centers = []

    for w in range(n_windows):
        start = w * step_size
        end = start + window_size
        window_data = timeseries[:, start:end]

        fc = np.corrcoef(window_data).astype(np.float32)
        fc = np.nan_to_num(fc, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(fc, 0)

        fc_matrices[w] = fc
        window_centers.append(start + window_size // 2)

    return fc_matrices, np.array(window_centers)


def assign_window_stages(stage_labels, censor_mask, window_centers, window_size):
    """
    Assign a stage label to each window via majority vote of valid TRs.

    Returns:
        window_stages: list of stage labels (or 'Mixed'/'Censored')
    """
    n_trs = len(stage_labels)
    window_stages = []

    for center in window_centers:
        start = max(0, center - window_size // 2)
        end = min(n_trs, start + window_size)

        window_labels = stage_labels[start:end]
        window_mask = censor_mask[start:end]

        valid = window_mask & np.isin(window_labels, list(config.VALID_STAGES))
        valid_labels = window_labels[valid]

        if len(valid_labels) == 0:
            window_stages.append("Censored")
            continue

        unique, counts = np.unique(valid_labels, return_counts=True)
        max_count = counts.max()
        majority_fraction = max_count / len(valid_labels)

        if majority_fraction >= 0.8:
            majority_stage = unique[counts.argmax()]
            window_stages.append(config.STAGE_LABELS.get(majority_stage, majority_stage))
        else:
            window_stages.append("Mixed")

    return window_stages


def compute_graph_metrics(fc_matrix, density_threshold, n_louvain_iter=10):
    """
    Compute Global Efficiency and Modularity for a single FC matrix.

    Uses reduced Louvain iterations (default 10 instead of 100) for speed.
    This is standard practice — consensus modularity with 10 restarts
    provides a good balance of accuracy and speed.

    Reference:
        Rubinov & Sporns, NeuroImage, 2010.
    """
    n_rois = fc_matrix.shape[0]

    fc_abs = np.abs(fc_matrix)
    np.fill_diagonal(fc_abs, 0)

    # Proportional threshold: keep top density_threshold fraction of edges
    threshold_value = np.percentile(
        fc_abs[np.triu_indices(n_rois, k=1)],
        (1 - density_threshold) * 100
    )
    thresholded = fc_abs.copy()
    thresholded[thresholded < threshold_value] = 0

    binary_adj = (thresholded > 0).astype(float)

    # Global Efficiency
    distance = bct.distance_bin(binary_adj)
    inv_dist = np.zeros_like(distance)
    mask = distance > 0
    inv_dist[mask] = 1.0 / distance[mask]
    np.fill_diagonal(inv_dist, 0)
    global_efficiency = inv_dist.sum() / (n_rois * (n_rois - 1))

    # Modularity (Louvain, reduced iterations)
    best_q = -np.inf
    for _ in range(n_louvain_iter):
        ci, q = bct.community_louvain(
            thresholded, gamma=config.MODULARITY_GAMMA
        )
        if q > best_q:
            best_q = q

    return {
        "global_efficiency": float(global_efficiency),
        "modularity": float(best_q),
    }


def process_single_run(ts_file, checkpoint_dir):
    """
    Process a single time-series file: compute dFC and graph metrics.

    Returns:
        metrics_rows: list of dicts (one per valid window)
        fc_sums: dict {stage: (sum_matrix, count)} for streaming mean
        checkpoint_path: path where results were saved

    If already checkpointed, loads from disk and returns the saved results.
    """
    stem = ts_file.stem  # e.g. sub-01_task-rest_run-1_timeseries
    checkpoint_path = checkpoint_dir / f"{stem}_metrics.csv"

    parts = stem.split("_")
    subject = parts[0]

    # --- Check if already processed ---
    if checkpoint_path.exists():
        try:
            cached = pd.read_csv(checkpoint_path)
            if len(cached) > 0:
                print(f"    ✓ {stem}: loaded {len(cached)} cached windows")
                return cached.to_dict("records"), None, checkpoint_path
        except Exception:
            pass  # Re-process if cache is corrupted

    # --- Load data ---
    data = np.load(str(ts_file), allow_pickle=True)
    timeseries = data["timeseries"]
    stage_labels = data["stage_labels"]
    censor_mask = data["censor_mask"]
    n_cortical = int(data["n_cortical_rois"])

    cortical_ts = timeseries[:n_cortical, :]

    # --- Compute sliding-window FC ---
    fc_matrices, window_centers = compute_sliding_window_fc(
        cortical_ts, config.DFC_WINDOW_SIZE, config.DFC_STEP_SIZE
    )

    if fc_matrices is None:
        print(f"    ⊘ {stem}: too few TRs for windowing")
        # Save empty checkpoint
        pd.DataFrame().to_csv(checkpoint_path, index=False)
        return [], None, checkpoint_path

    # --- Assign window stages ---
    window_stages = assign_window_stages(
        stage_labels, censor_mask, window_centers, config.DFC_WINDOW_SIZE
    )

    # --- Compute graph metrics per window ---
    metrics_rows = []
    fc_sums = {}  # stage -> (running_sum_matrix, count)

    n_valid = 0
    for w_idx in range(len(fc_matrices)):
        stage = window_stages[w_idx]
        if stage in ("Censored", "Mixed"):
            continue

        metrics = compute_graph_metrics(
            fc_matrices[w_idx], config.GRAPH_DENSITY_THRESHOLD,
            n_louvain_iter=10
        )

        metrics_rows.append({
            "subject": subject,
            "file": stem,
            "window_idx": w_idx,
            "window_center_tr": int(window_centers[w_idx]),
            "stage": stage,
            **metrics,
        })

        # Streaming mean: accumulate FC sum per stage
        if stage not in fc_sums:
            fc_sums[stage] = (np.zeros_like(fc_matrices[w_idx], dtype=np.float64), 0)
        running_sum, count = fc_sums[stage]
        running_sum += fc_matrices[w_idx].astype(np.float64)
        fc_sums[stage] = (running_sum, count + 1)

        n_valid += 1

    # --- Save checkpoint ---
    if metrics_rows:
        pd.DataFrame(metrics_rows).to_csv(checkpoint_path, index=False)

    print(f"    ✓ {stem}: {n_valid} valid windows "
          f"(of {len(fc_matrices)} total)")

    # Free FC matrices immediately to save memory
    del fc_matrices, timeseries, cortical_ts
    gc.collect()

    return metrics_rows, fc_sums, checkpoint_path


def run_statistical_comparison(metrics_df):
    """
    Run linear mixed-effects model and post-hoc pairwise comparisons.

    Model: metric ~ stage + (1|subject)

    Returns:
        dict with ANOVA results and pairwise comparisons
    """
    import statsmodels.formula.api as smf
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    results = {}

    for metric in ["global_efficiency", "modularity"]:
        metric_data = metrics_df[metrics_df[metric].notna()].copy()

        if len(metric_data) < 10:
            continue

        # Mixed-effects model: metric ~ stage + (1|subject)
        try:
            model = smf.mixedlm(
                f"{metric} ~ C(stage, Treatment(reference='Wake'))",
                data=metric_data,
                groups=metric_data["subject"],
            )
            fit = model.fit(reml=True)

            results[metric] = {
                "lmm_summary": str(fit.summary()),
                "coefficients": {},
                "pvalues": {},
            }

            for param in fit.params.index:
                results[metric]["coefficients"][param] = float(fit.params[param])
                results[metric]["pvalues"][param] = float(fit.pvalues[param])

        except Exception as e:
            print(f"    LMM failed for {metric}: {e}")
            results[metric] = {"error": str(e)}
            continue

        # Post-hoc pairwise comparisons (Tukey HSD)
        try:
            tukey = pairwise_tukeyhsd(
                metric_data[metric],
                metric_data["stage"],
                alpha=config.ALPHA,
            )
            results[metric]["tukey_summary"] = str(tukey.summary())

            # Pairwise effect sizes (Cohen's d)
            stages = ["Wake", "N1", "N2"]
            pairwise_effects = {}
            for i, s1 in enumerate(stages):
                for s2 in stages[i+1:]:
                    d1 = metric_data.loc[metric_data["stage"] == s1, metric].values
                    d2 = metric_data.loc[metric_data["stage"] == s2, metric].values
                    if len(d1) > 1 and len(d2) > 1:
                        pooled_std = np.sqrt(
                            ((len(d1) - 1) * np.var(d1, ddof=1) +
                             (len(d2) - 1) * np.var(d2, ddof=1)) /
                            (len(d1) + len(d2) - 2)
                        )
                        if pooled_std > 0:
                            cohens_d = (np.mean(d1) - np.mean(d2)) / pooled_std
                        else:
                            cohens_d = 0.0
                        pairwise_effects[f"{s1}_vs_{s2}"] = {
                            "cohens_d": float(cohens_d),
                            "mean_diff": float(np.mean(d1) - np.mean(d2)),
                        }
            results[metric]["effect_sizes"] = pairwise_effects

        except Exception as e:
            results[metric]["tukey_error"] = str(e)

    return results


def create_dfc_figures(metrics_df, mean_fc_per_stage, stats_results):
    """Generate publication-quality dFC figures."""
    fig_dir = config.FIGURES_DIR

    # --- Figure 1: Violin plots of graph metrics by stage ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    stage_order = ["Wake", "N1", "N2"]

    for ax, metric, ylabel in zip(
        axes,
        ["global_efficiency", "modularity"],
        ["Global Efficiency", "Modularity (Q)"],
    ):
        plot_data = metrics_df[metrics_df["stage"].isin(stage_order)].copy()
        if len(plot_data) == 0:
            continue

        # Subject-level means
        subject_means = plot_data.groupby(["subject", "stage"])[metric].mean().reset_index()

        palette = [config.STAGE_COLORS[s] for s in stage_order]
        present_stages = [s for s in stage_order if s in subject_means["stage"].values]
        parts = ax.violinplot(
            [subject_means.loc[subject_means["stage"] == s, metric].values
             for s in present_stages],
            positions=range(len(present_stages)),
            showmeans=True,
            showextrema=True,
        )

        for i, pc in enumerate(parts["bodies"]):
            if i < len(palette):
                pc.set_facecolor(palette[i])
                pc.set_alpha(0.6)

        for i, stage in enumerate(present_stages):
            vals = subject_means.loc[subject_means["stage"] == stage, metric].values
            jitter = np.random.normal(0, 0.04, size=len(vals))
            ax.scatter(i + jitter, vals, alpha=0.5, s=15,
                      color=config.STAGE_COLORS.get(stage, "gray"),
                      edgecolors="white", linewidth=0.5, zorder=3)

        ax.set_xticks(range(len(present_stages)))
        ax.set_xticklabels(present_stages, fontsize=config.FONT_SIZE_LABEL)
        ax.set_ylabel(ylabel, fontsize=config.FONT_SIZE_LABEL)
        ax.set_title(f"{ylabel} Across Sleep Stages", fontsize=config.FONT_SIZE_TITLE)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        fig.savefig(str(fig_dir / f"dfc_graph_metrics.{fmt}"),
                   dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Mean FC matrices per stage ---
    if mean_fc_per_stage:
        n_stages = len(mean_fc_per_stage)
        fig, axes = plt.subplots(1, n_stages + 1, figsize=(6 * (n_stages + 1), 5))
        if n_stages + 1 == 1:
            axes = [axes]

        all_fc_vals = []
        for stage_name, fc in mean_fc_per_stage.items():
            all_fc_vals.extend(fc[np.triu_indices(fc.shape[0], k=1)].tolist())
        vmax = np.percentile(np.abs(all_fc_vals), 98) if all_fc_vals else 1

        for idx, (stage_name, fc) in enumerate(mean_fc_per_stage.items()):
            ax = axes[idx]
            im = ax.imshow(fc, cmap=config.CONNECTIVITY_CMAP, vmin=-vmax, vmax=vmax,
                          aspect="equal")
            ax.set_title(f"{stage_name}", fontsize=config.FONT_SIZE_TITLE,
                        fontweight="bold",
                        color=config.STAGE_COLORS.get(stage_name, "black"))
            ax.set_xlabel("ROI", fontsize=config.FONT_SIZE_LABEL)
            ax.set_ylabel("ROI", fontsize=config.FONT_SIZE_LABEL)

        # Difference matrix (Wake - N2)
        if "Wake" in mean_fc_per_stage and "N2" in mean_fc_per_stage:
            ax = axes[-1]
            diff = mean_fc_per_stage["Wake"] - mean_fc_per_stage["N2"]
            im = ax.imshow(diff, cmap=config.CONNECTIVITY_CMAP,
                          vmin=-vmax/2, vmax=vmax/2, aspect="equal")
            ax.set_title("Wake − N2", fontsize=config.FONT_SIZE_TITLE, fontweight="bold")
            ax.set_xlabel("ROI", fontsize=config.FONT_SIZE_LABEL)
            ax.set_ylabel("ROI", fontsize=config.FONT_SIZE_LABEL)

        plt.colorbar(im, ax=axes, shrink=0.6, label="Correlation (r)")
        plt.suptitle("Mean Functional Connectivity Matrices",
                    fontsize=config.FONT_SIZE_TITLE + 2, fontweight="bold", y=1.02)
        plt.tight_layout()
        for fmt in config.FIGURE_FORMAT:
            fig.savefig(str(fig_dir / f"dfc_fc_matrices.{fmt}"),
                       dpi=config.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)


def main():
    print("=" * 70)
    print("SLEEP NETWORK REORGANIZATION — DYNAMIC FUNCTIONAL CONNECTIVITY")
    print("=" * 70)

    config.setup_directories()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all extracted time-series files
    ts_files = sorted(config.TIMESERIES_DIR.glob("*_timeseries.npz"))
    if not ts_files:
        print("ERROR: No time-series files found. Run 02_extract_timeseries.py first.")
        sys.exit(1)

    print(f"\nFound {len(ts_files)} time-series files")
    print(f"Window size: {config.DFC_WINDOW_SIZE} TRs ({config.DFC_WINDOW_SIZE * config.TR:.1f}s)")
    print(f"Step size: {config.DFC_STEP_SIZE} TR")
    print(f"Graph density threshold: {config.GRAPH_DENSITY_THRESHOLD}")

    # Check how many are already checkpointed
    existing = set(f.stem.replace("_metrics", "") for f in CHECKPOINT_DIR.glob("*_metrics.csv"))
    done = sum(1 for f in ts_files if f.stem in existing)
    print(f"Checkpointed: {done}/{len(ts_files)} runs already processed\n")

    # --- Process each run with checkpointing ---
    all_metrics = []
    # Streaming mean FC: {stage: (running_sum, count)}
    global_fc_sums = {}

    for ts_file in tqdm(ts_files, desc="Computing dFC"):
        metrics_rows, fc_sums, checkpoint_path = process_single_run(
            ts_file, CHECKPOINT_DIR
        )

        all_metrics.extend(metrics_rows)

        # Merge per-run FC sums into global accumulator
        if fc_sums is not None:
            for stage, (fc_sum, count) in fc_sums.items():
                if stage not in global_fc_sums:
                    global_fc_sums[stage] = (np.zeros_like(fc_sum), 0)
                g_sum, g_count = global_fc_sums[stage]
                g_sum += fc_sum
                global_fc_sums[stage] = (g_sum, g_count + count)

        # Force garbage collection after each run
        gc.collect()

    if not all_metrics:
        print("ERROR: No valid windows computed.")
        sys.exit(1)

    # --- Combine all metrics ---
    print("\nCombining results from all runs...")
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(str(config.DFC_DIR / "graph_metrics_per_window.csv"), index=False)

    print(f"Total: {len(metrics_df)} valid windows")
    print("Stage distribution:")
    for stage in ["Wake", "N1", "N2"]:
        n = len(metrics_df[metrics_df["stage"] == stage])
        n_subs = metrics_df.loc[metrics_df["stage"] == stage, "subject"].nunique()
        print(f"  {stage}: {n} windows from {n_subs} subjects")

    # --- Compute mean FC per stage (from streaming sums) ---
    mean_fc_per_stage = {}

    # First try to use streaming sums (from freshly processed runs)
    if global_fc_sums:
        for stage, (fc_sum, count) in global_fc_sums.items():
            if count > 0:
                mean_fc_per_stage[stage] = fc_sum / count
    else:
        # If all runs were cached, recompute from checkpoints
        # (the FC sums aren't saved per-run, only graph metrics are)
        print("  Re-computing mean FC from cached checkpoints...")
        for ts_file in tqdm(ts_files, desc="  Mean FC"):
            data = np.load(str(ts_file), allow_pickle=True)
            timeseries = data["timeseries"]
            stage_labels = data["stage_labels"]
            censor_mask = data["censor_mask"]
            n_cortical = int(data["n_cortical_rois"])

            cortical_ts = timeseries[:n_cortical, :]
            fc_matrices, window_centers = compute_sliding_window_fc(
                cortical_ts, config.DFC_WINDOW_SIZE, config.DFC_STEP_SIZE
            )

            if fc_matrices is None:
                del data, timeseries, cortical_ts
                gc.collect()
                continue

            window_stages = assign_window_stages(
                stage_labels, censor_mask, window_centers, config.DFC_WINDOW_SIZE
            )

            for w_idx in range(len(fc_matrices)):
                stage = window_stages[w_idx]
                if stage in ("Censored", "Mixed"):
                    continue
                if stage not in mean_fc_per_stage:
                    mean_fc_per_stage[stage] = {
                        "sum": np.zeros_like(fc_matrices[w_idx], dtype=np.float64),
                        "count": 0
                    }
                mean_fc_per_stage[stage]["sum"] += fc_matrices[w_idx].astype(np.float64)
                mean_fc_per_stage[stage]["count"] += 1

            del fc_matrices, timeseries, cortical_ts, data
            gc.collect()

        # Finalize means
        for stage in list(mean_fc_per_stage.keys()):
            if isinstance(mean_fc_per_stage[stage], dict):
                s = mean_fc_per_stage[stage]
                if s["count"] > 0:
                    mean_fc_per_stage[stage] = s["sum"] / s["count"]
                else:
                    del mean_fc_per_stage[stage]

    # Save mean FC
    if mean_fc_per_stage:
        np.savez(
            str(config.DFC_DIR / "mean_fc_per_stage.npz"),
            **{f"fc_{stage}": fc for stage, fc in mean_fc_per_stage.items()},
            stages=list(mean_fc_per_stage.keys()),
        )

    # --- Statistical comparison ---
    print("\nRunning statistical analyses...")
    stats_results = run_statistical_comparison(metrics_df)

    for metric in ["global_efficiency", "modularity"]:
        if metric in stats_results and "effect_sizes" in stats_results[metric]:
            print(f"\n  {metric}:")
            for pair, eff in stats_results[metric]["effect_sizes"].items():
                print(f"    {pair}: Cohen's d = {eff['cohens_d']:.3f}, "
                      f"mean_diff = {eff['mean_diff']:.4f}")

    stats_path = config.DFC_DIR / "stage_comparison_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats_results, f, indent=2, default=str)

    # --- Figures ---
    print("\nGenerating figures...")
    create_dfc_figures(metrics_df, mean_fc_per_stage, stats_results)

    print(f"\n✓ dFC analysis complete. Results in: {config.DFC_DIR}")
    print(f"  Per-subject checkpoints: {CHECKPOINT_DIR}")


if __name__ == "__main__":
    main()
