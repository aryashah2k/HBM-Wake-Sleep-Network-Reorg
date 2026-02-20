#!/usr/bin/env python3
"""
04_thalamocortical.py — Thalamocortical Coupling Analysis

Examines the functional connectivity between thalamus and cortical networks
(Yeo 7-network parcellation) across Wake, N1, and N2 sleep stages.

Reference:
    Tagliazucchi & Laufs, "Decoding wakefulness levels from typical fMRI
    resting-state data reveals reliable drifts between wakefulness and sleep",
    Neuron, 2014.

    Spoormaker et al., "Development of a large-scale functional brain network
    during human non-rapid eye movement sleep", J. Neuroscience, 2010.

Outputs:
    results/thalamocortical/coupling_values.csv
    results/thalamocortical/anova_results.json
    results/figures/thalamocortical_*.png / .pdf
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import pingouin as pg

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

warnings.filterwarnings("ignore")


def get_network_assignment(roi_labels, n_cortical):
    """
    Map Schaefer 200 parcel labels to Yeo 7-network assignments.

    Schaefer labels follow the format: '7Networks_LH_Vis_1' etc.

    Returns:
        dict: {network_name: list of parcel indices}
    """
    network_map = {net: [] for net in config.YEO7_NETWORKS}

    for idx in range(n_cortical):
        label = str(roi_labels[idx])

        # Parse network from Schaefer label
        assigned = False
        for net in config.YEO7_NETWORKS:
            if net in label:
                network_map[net].append(idx)
                assigned = True
                break

        if not assigned:
            # Try alternative naming conventions
            label_lower = label.lower()
            for net in config.YEO7_NETWORKS:
                if net.lower() in label_lower:
                    network_map[net].append(idx)
                    assigned = True
                    break

    return network_map


def find_contiguous_blocks(stage_labels, censor_mask, target_stage, min_length):
    """
    Find contiguous blocks of TRs with the same stage label.

    Args:
        stage_labels: per-TR stage labels
        censor_mask: boolean mask (True = valid)
        target_stage: stage to find blocks of
        min_length: minimum block length in TRs

    Returns:
        list of (start_idx, end_idx) tuples
    """
    n_trs = len(stage_labels)
    blocks = []
    current_start = None

    for i in range(n_trs):
        is_target = (stage_labels[i] == target_stage) and censor_mask[i]

        if is_target:
            if current_start is None:
                current_start = i
        else:
            if current_start is not None:
                length = i - current_start
                if length >= min_length:
                    blocks.append((current_start, i))
                current_start = None

    # Handle block at end
    if current_start is not None:
        length = n_trs - current_start
        if length >= min_length:
            blocks.append((current_start, n_trs))

    return blocks


def compute_thalamocortical_coupling(timeseries, thalamic_indices, network_map, blocks):
    """
    Compute correlation between thalamus mean signal and each cortical network
    mean signal for given time blocks.

    Returns:
        dict: {network_name: list of correlation values}
    """
    if not blocks:
        return None

    coupling = {net: [] for net in config.YEO7_NETWORKS}

    for start, end in blocks:
        # Mean thalamic signal (bilateral average)
        thal_ts = timeseries[thalamic_indices, start:end].mean(axis=0)

        for net_name, parcel_indices in network_map.items():
            if not parcel_indices:
                continue

            # Mean network signal
            net_ts = timeseries[parcel_indices, start:end].mean(axis=0)

            # Pearson correlation
            if np.std(thal_ts) > 0 and np.std(net_ts) > 0:
                r, _ = stats.pearsonr(thal_ts, net_ts)
                # Fisher z-transform for averaging
                z = np.arctanh(np.clip(r, -0.999, 0.999))
                coupling[net_name].append(z)

    return coupling


def run_statistical_analysis(coupling_df):
    """
    Run repeated-measures ANOVA (stage × network) and post-hoc tests.

    Returns:
        dict with ANOVA table, post-hoc comparisons
    """
    results = {}

    # 1. Overall repeated-measures ANOVA: coupling ~ stage * network
    # Using pingouin for cleaner repeated-measures ANOVA
    try:
        # Average coupling per subject × stage × network
        avg_df = coupling_df.groupby(
            ["subject", "stage", "network"]
        )["coupling_z"].mean().reset_index()

        # Two-way RM-ANOVA
        aov = pg.rm_anova(
            data=avg_df,
            dv="coupling_z",
            within=["stage", "network"],
            subject="subject",
        )
        results["rm_anova"] = aov.to_dict()
        results["rm_anova_text"] = str(aov)
    except Exception as e:
        print(f"    RM-ANOVA failed: {e}")
        results["rm_anova_error"] = str(e)

    # 2. Per-network analysis: coupling ~ stage (within-subject)
    network_results = {}
    for network in config.YEO7_NETWORKS:
        net_data = coupling_df[coupling_df["network"] == network].copy()
        if len(net_data) < 10:
            continue

        net_avg = net_data.groupby(["subject", "stage"])["coupling_z"].mean().reset_index()

        # Check we have at least 2 stages with data
        stages_present = net_avg["stage"].unique()
        if len(stages_present) < 2:
            continue

        # One-way RM-ANOVA within this network
        try:
            aov_net = pg.rm_anova(
                data=net_avg,
                dv="coupling_z",
                within="stage",
                subject="subject",
            )
            network_results[network] = {
                "anova": aov_net.to_dict(),
            }

            # Post-hoc pairwise t-tests (FDR corrected)
            posthoc = pg.pairwise_tests(
                data=net_avg,
                dv="coupling_z",
                within="stage",
                subject="subject",
                padjust="fdr_bh",
            )
            network_results[network]["posthoc"] = posthoc.to_dict()
            network_results[network]["posthoc_text"] = str(posthoc)

            # Effect sizes
            stage_means = net_avg.groupby("stage")["coupling_z"].agg(["mean", "std"])
            network_results[network]["stage_means"] = stage_means.to_dict()

        except Exception as e:
            network_results[network] = {"error": str(e)}

    results["per_network"] = network_results

    return results


def create_thalamocortical_figures(coupling_df, stats_results):
    """Generate publication-quality thalamocortical coupling figures."""
    fig_dir = config.FIGURES_DIR

    # --- Figure 1: Radar/Spider plot of thalamocortical coupling per stage ---
    avg_df = coupling_df.groupby(
        ["stage", "network"]
    )["coupling_z"].mean().reset_index()

    networks = config.YEO7_NETWORKS
    stages = ["Wake", "N1", "N2"]

    # Radar plot
    n_networks = len(networks)
    angles = np.linspace(0, 2 * np.pi, n_networks, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for stage in stages:
        stage_data = avg_df[avg_df["stage"] == stage]
        values = []
        for net in networks:
            net_vals = stage_data[stage_data["network"] == net]["coupling_z"].values
            values.append(float(net_vals[0]) if len(net_vals) > 0 else 0)
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2, label=stage,
                color=config.STAGE_COLORS[stage])
        ax.fill(angles, values, alpha=0.1, color=config.STAGE_COLORS[stage])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [config.YEO7_NETWORK_FULL_NAMES.get(n, n) for n in networks],
        fontsize=config.FONT_SIZE_TICK,
    )
    ax.set_title("Thalamocortical Coupling Across Sleep Stages",
                fontsize=config.FONT_SIZE_TITLE, pad=20, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
             fontsize=config.FONT_SIZE_LEGEND)
    ax.set_ylabel("Coupling (Fisher z)", fontsize=config.FONT_SIZE_LABEL, labelpad=30)

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        fig.savefig(str(fig_dir / f"thalamocortical_radar.{fmt}"),
                   dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Bar plot per network with error bars ---
    fig, ax = plt.subplots(figsize=(14, 6))

    # Subject-level means for error bars
    subj_avg = coupling_df.groupby(
        ["subject", "stage", "network"]
    )["coupling_z"].mean().reset_index()

    x = np.arange(len(networks))
    width = 0.25
    offsets = {"Wake": -width, "N1": 0, "N2": width}

    for stage in stages:
        stage_data = subj_avg[subj_avg["stage"] == stage]
        means = []
        sems = []
        for net in networks:
            vals = stage_data[stage_data["network"] == net]["coupling_z"].values
            means.append(np.mean(vals) if len(vals) > 0 else 0)
            sems.append(stats.sem(vals) if len(vals) > 1 else 0)

        ax.bar(x + offsets[stage], means, width, yerr=sems,
               label=stage, color=config.STAGE_COLORS[stage],
               capsize=3, alpha=0.8, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Cortical Network", fontsize=config.FONT_SIZE_LABEL)
    ax.set_ylabel("Thalamocortical Coupling (Fisher z)", fontsize=config.FONT_SIZE_LABEL)
    ax.set_title("Thalamocortical Coupling by Network and Sleep Stage",
                fontsize=config.FONT_SIZE_TITLE, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [config.YEO7_NETWORK_FULL_NAMES.get(n, n) for n in networks],
        rotation=30, ha="right", fontsize=config.FONT_SIZE_TICK,
    )
    ax.legend(fontsize=config.FONT_SIZE_LEGEND)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add significance markers from per-network post-hoc
    if "per_network" in stats_results:
        y_max = ax.get_ylim()[1]
        for net_idx, net in enumerate(networks):
            if net in stats_results["per_network"]:
                net_res = stats_results["per_network"][net]
                if "posthoc" in net_res:
                    posthoc = pd.DataFrame(net_res["posthoc"])
                    for _, row in posthoc.iterrows():
                        p_val = row.get("p-corr", row.get("p-unc", 1.0))
                        if isinstance(p_val, dict):
                            p_val = list(p_val.values())[0] if p_val else 1.0
                        if p_val < 0.05:
                            ax.text(net_idx, y_max * 0.95, "*",
                                   ha="center", fontsize=14, fontweight="bold")

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        fig.savefig(str(fig_dir / f"thalamocortical_barplot.{fmt}"),
                   dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: Heatmap of coupling change (N2 - Wake) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Stage × Network coupling heatmap
    pivot_data = avg_df.pivot(index="network", columns="stage", values="coupling_z")
    pivot_data = pivot_data.reindex(index=networks, columns=stages)

    ax = axes[0]
    sns.heatmap(pivot_data, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                ax=ax, cbar_kws={"label": "Coupling (Fisher z)"},
                yticklabels=[config.YEO7_NETWORK_FULL_NAMES.get(n, n) for n in networks])
    ax.set_title("Thalamocortical Coupling", fontsize=config.FONT_SIZE_TITLE,
                fontweight="bold")

    # Change heatmap
    if "Wake" in pivot_data.columns and "N2" in pivot_data.columns:
        ax = axes[1]
        change = pd.DataFrame(index=networks)
        change["Wake→N1"] = pivot_data["N1"] - pivot_data["Wake"]
        change["N1→N2"] = pivot_data["N2"] - pivot_data["N1"]
        change["Wake→N2"] = pivot_data["N2"] - pivot_data["Wake"]

        sns.heatmap(change, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                    ax=ax, cbar_kws={"label": "Δ Coupling (Fisher z)"},
                    yticklabels=[config.YEO7_NETWORK_FULL_NAMES.get(n, n) for n in networks])
        ax.set_title("Coupling Change Across Stages",
                    fontsize=config.FONT_SIZE_TITLE, fontweight="bold")

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        fig.savefig(str(fig_dir / f"thalamocortical_heatmap.{fmt}"),
                   dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=" * 70)
    print("SLEEP NETWORK REORGANIZATION — THALAMOCORTICAL COUPLING")
    print("=" * 70)

    config.setup_directories()

    # Load time-series files
    ts_files = sorted(config.TIMESERIES_DIR.glob("*_timeseries.npz"))
    if not ts_files:
        print("ERROR: No time-series files found. Run 02_extract_timeseries.py first.")
        sys.exit(1)

    print(f"\nProcessing {len(ts_files)} time-series files...")

    all_coupling_data = []

    for ts_file in tqdm(ts_files, desc="Thalamocortical coupling"):
        data = np.load(str(ts_file), allow_pickle=True)
        timeseries = data["timeseries"]
        stage_labels = data["stage_labels"]
        censor_mask = data["censor_mask"]
        roi_labels = data["roi_labels"]
        n_cortical = int(data["n_cortical_rois"])
        n_thalamic = int(data["n_thalamic_rois"])

        subject = ts_file.stem.split("_")[0]

        if n_thalamic == 0:
            continue

        # Get thalamic indices
        thalamic_indices = list(range(n_cortical, n_cortical + n_thalamic))

        # Get network assignment
        network_map = get_network_assignment(roi_labels, n_cortical)

        # Process each stage
        for stage_code, stage_name in config.STAGE_LABELS.items():
            blocks = find_contiguous_blocks(
                stage_labels, censor_mask, stage_code, config.MIN_BLOCK_LENGTH_TRS
            )

            if not blocks:
                continue

            coupling = compute_thalamocortical_coupling(
                timeseries, thalamic_indices, network_map, blocks
            )

            if coupling is None:
                continue

            for net_name, z_values in coupling.items():
                for z_val in z_values:
                    all_coupling_data.append({
                        "subject": subject,
                        "stage": stage_name,
                        "network": net_name,
                        "coupling_z": z_val,
                    })

    if not all_coupling_data:
        print("ERROR: No coupling data computed.")
        sys.exit(1)

    coupling_df = pd.DataFrame(all_coupling_data)
    coupling_df.to_csv(str(config.THALAMOCORTICAL_DIR / "coupling_values.csv"), index=False)

    print(f"\nComputed {len(coupling_df)} coupling measurements")
    print(f"Subjects: {coupling_df['subject'].nunique()}")
    print(f"Stage distribution:")
    for stage in ["Wake", "N1", "N2"]:
        n = len(coupling_df[coupling_df["stage"] == stage])
        print(f"  {stage}: {n} measurements")

    # Statistical analysis
    print("\nRunning statistical analysis...")
    stats_results = run_statistical_analysis(coupling_df)

    # Print key results
    if "rm_anova_text" in stats_results:
        print(f"\nRM-ANOVA results:\n{stats_results['rm_anova_text']}")

    stats_path = config.THALAMOCORTICAL_DIR / "anova_results.json"
    with open(stats_path, "w") as f:
        json.dump(stats_results, f, indent=2, default=str)

    # Create figures
    print("\nGenerating figures...")
    create_thalamocortical_figures(coupling_df, stats_results)

    print(f"\n✓ Thalamocortical coupling analysis complete. Results: {config.THALAMOCORTICAL_DIR}")


if __name__ == "__main__":
    main()
