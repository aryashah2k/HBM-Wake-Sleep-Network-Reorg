#!/usr/bin/env python3
"""
07_visualize.py — Publication-Quality Summary Figures

Generates integrated summary figures combining results from all
analysis steps: GLM, dFC, and thalamocortical coupling.

Outputs:
    results/figures/fig1_study_overview.{png,pdf}
    results/figures/fig7_summary_trajectory.{png,pdf}
"""


import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

warnings.filterwarnings("ignore")


def create_study_overview_figure():
    """
    Figure 1: Study overview — sleep staging distribution, pipeline schematic.
    """
    fig_dir = config.FIGURES_DIR

    # Load stage distribution
    stage_csv = config.QC_DIR / "stage_distribution.csv"
    if not stage_csv.exists():
        print("  Skipping study overview: no stage distribution data")
        return

    stage_df = pd.read_csv(stage_csv)

    fig = plt.figure(figsize=(18, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])

    # Panel A: Stage distribution stacked bar
    ax_a = fig.add_subplot(gs[0])
    x = np.arange(len(stage_df))
    width = 0.6

    wake = stage_df["Wake_TRs"].values
    n1 = stage_df["N1_TRs"].values
    n2 = stage_df["N2_TRs"].values

    ax_a.bar(x, wake, width, label="Wake", color=config.STAGE_COLORS["Wake"])
    ax_a.bar(x, n1, width, bottom=wake, label="N1", color=config.STAGE_COLORS["N1"])
    ax_a.bar(x, n2, width, bottom=wake + n1, label="N2", color=config.STAGE_COLORS["N2"])

    ax_a.set_xlabel("Subject", fontsize=config.FONT_SIZE_LABEL)
    ax_a.set_ylabel("Number of Valid TRs", fontsize=config.FONT_SIZE_LABEL)
    ax_a.set_title("A  Sleep Stage Distribution", fontsize=config.FONT_SIZE_TITLE,
                   fontweight="bold", loc="left")
    ax_a.set_xticks(x[::3])
    ax_a.set_xticklabels(stage_df["subject"].values[::3],
                         rotation=45, ha="right", fontsize=8)
    ax_a.legend(fontsize=config.FONT_SIZE_LEGEND, loc="upper right")
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # Panel B: Pipeline flowchart (text-based schematic)
    ax_b = fig.add_subplot(gs[1])
    ax_b.axis("off")

    pipeline_steps = [
        ("1. Data Validation", "QC, motion censoring\nFD < 0.3 mm", "#E3F2FD"),
        ("2. GLM Analysis", "Wake vs N1 vs N2\nblock-design GLM", "#E8F5E9"),
        ("3. Time-Series\n    Extraction", "Schaefer 200 parcels\n+ bilateral thalamus", "#FFF3E0"),
        ("4. Dynamic FC", "Sliding window (44 TRs)\nModularity, Global Eff.", "#F3E5F5"),
        ("5. Thalamocortical\n    Coupling", "Thalamus ↔ 7 networks\nRM-ANOVA", "#FFEBEE"),

        ("7. Reliability", "ICC, split-half\nSpearman-Brown", "#FFF9C4"),
    ]

    y_positions = np.linspace(0.95, 0.05, len(pipeline_steps))

    for i, (title, desc, color) in enumerate(pipeline_steps):
        y = y_positions[i]
        bbox = dict(boxstyle="round,pad=0.3", facecolor=color,
                   edgecolor="gray", alpha=0.8)
        ax_b.text(0.5, y, f"{title}\n{desc}",
                 transform=ax_b.transAxes,
                 fontsize=9, ha="center", va="center",
                 bbox=bbox, fontfamily="monospace")

        if i < len(pipeline_steps) - 1:
            ax_b.annotate("", xy=(0.5, y_positions[i + 1] + 0.04),
                         xytext=(0.5, y - 0.04),
                         arrowprops=dict(arrowstyle="->", color="gray",
                                        lw=1.5))

    ax_b.set_title("B  Analysis Pipeline", fontsize=config.FONT_SIZE_TITLE,
                   fontweight="bold", loc="left")

    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        fig.savefig(str(fig_dir / f"fig1_study_overview.{fmt}"),
                   dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Figure 1: Study overview")


def create_summary_trajectory_figure():
    """
    Figure 7: Integrated summary of network reorganization trajectory
    (Wake → N1 → N2) combining dFC and thalamocortical results.
    """
    fig_dir = config.FIGURES_DIR

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.35)

    # Panel A: dFC metrics trajectory
    dfc_csv = config.DFC_DIR / "graph_metrics_per_window.csv"
    if dfc_csv.exists():
        ax_a = fig.add_subplot(gs[0, 0])
        dfc_df = pd.read_csv(dfc_csv)

        stage_order = ["Wake", "N1", "N2"]
        dfc_stage = dfc_df[dfc_df["stage"].isin(stage_order)]
        subj_means = dfc_stage.groupby(
            ["subject", "stage"]
        )[["global_efficiency", "modularity"]].mean().reset_index()

        # Global efficiency trajectory
        for subject in subj_means["subject"].unique():
            s_data = subj_means[subj_means["subject"] == subject]
            s_data = s_data.set_index("stage").reindex(stage_order)
            ax_a.plot(range(3), s_data["global_efficiency"].values,
                     "-", color="gray", alpha=0.15, linewidth=0.5)

        group_means = subj_means.groupby("stage")["global_efficiency"].agg(["mean", "sem"])
        group_means = group_means.reindex(stage_order)

        ax_a.errorbar(range(3), group_means["mean"], yerr=group_means["sem"],
                     fmt="o-", color=sns.color_palette("deep")[0], linewidth=2.5,
                     markersize=10, capsize=5, zorder=5)

        ax_a.set_xticks(range(3))
        ax_a.set_xticklabels(stage_order, fontsize=config.FONT_SIZE_LABEL)
        ax_a.set_ylabel("Global Efficiency", fontsize=config.FONT_SIZE_LABEL)
        ax_a.set_title("A  Integration Trajectory", fontsize=config.FONT_SIZE_TITLE,
                       fontweight="bold", loc="left")
        ax_a.spines["top"].set_visible(False)
        ax_a.spines["right"].set_visible(False)

    # Panel B: Modularity trajectory
    if dfc_csv.exists():
        ax_b = fig.add_subplot(gs[0, 1])

        for subject in subj_means["subject"].unique():
            s_data = subj_means[subj_means["subject"] == subject]
            s_data = s_data.set_index("stage").reindex(stage_order)
            ax_b.plot(range(3), s_data["modularity"].values,
                     "-", color="gray", alpha=0.15, linewidth=0.5)

        group_mod = subj_means.groupby("stage")["modularity"].agg(["mean", "sem"])
        group_mod = group_mod.reindex(stage_order)

        ax_b.errorbar(range(3), group_mod["mean"], yerr=group_mod["sem"],
                     fmt="s-", color=sns.color_palette("deep")[1], linewidth=2.5,
                     markersize=10, capsize=5, zorder=5)

        ax_b.set_xticks(range(3))
        ax_b.set_xticklabels(stage_order, fontsize=config.FONT_SIZE_LABEL)
        ax_b.set_ylabel("Modularity (Q)", fontsize=config.FONT_SIZE_LABEL)
        ax_b.set_title("B  Segregation Trajectory", fontsize=config.FONT_SIZE_TITLE,
                       fontweight="bold", loc="left")
        ax_b.spines["top"].set_visible(False)
        ax_b.spines["right"].set_visible(False)

    # Panel C: Thalamocortical coupling summary
    thc_csv = config.THALAMOCORTICAL_DIR / "coupling_values.csv"
    if thc_csv.exists():
        ax_c = fig.add_subplot(gs[1, 0])
        thc_df = pd.read_csv(thc_csv)

        # Mean coupling across all networks per stage
        stage_coupling = thc_df.groupby("stage")["coupling_z"].agg(["mean", "sem"]).reset_index()
        stage_coupling = stage_coupling.set_index("stage").reindex(stage_order)

        colors = [config.STAGE_COLORS[s] for s in stage_order]
        ax_c.bar(range(3), stage_coupling["mean"], yerr=stage_coupling["sem"],
                color=colors, capsize=5, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax_c.set_xticks(range(3))
        ax_c.set_xticklabels(stage_order, fontsize=config.FONT_SIZE_LABEL)
        ax_c.set_ylabel("Mean Coupling (Fisher z)", fontsize=config.FONT_SIZE_LABEL)
        ax_c.set_title("C  Thalamocortical Coupling", fontsize=config.FONT_SIZE_TITLE,
                       fontweight="bold", loc="left")
        ax_c.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
        ax_c.spines["top"].set_visible(False)
        ax_c.spines["right"].set_visible(False)



    # Panel D: Reliability summary
    icc_csv = config.RELIABILITY_DIR / "icc_results.csv"
    if icc_csv.exists():
        ax_f = fig.add_subplot(gs[1, 1])
        icc_df = pd.read_csv(icc_csv)

        if "icc_value" in icc_df.columns and len(icc_df) > 0:
            valid_icc = icc_df.dropna(subset=["icc_value"])
            colors_rel = []
            for _, row in valid_icc.iterrows():
                interp = row.get("interpretation", "Poor")
                color_map = {
                    "Excellent": "#4CAF50", "Good": "#8BC34A",
                    "Fair": "#FF9800", "Poor": "#F44336",
                }
                colors_rel.append(color_map.get(interp, "#9E9E9E"))

            ax_f.barh(range(len(valid_icc)), valid_icc["icc_value"],
                     color=colors_rel, alpha=0.8, edgecolor="white", linewidth=0.5)
            ax_f.set_yticks(range(len(valid_icc)))
            ax_f.set_yticklabels(valid_icc["metric"].apply(
                lambda x: x.replace("task-", "").replace("_", "\n")
            ), fontsize=8)
            ax_f.axvline(x=0.75, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
            ax_f.axvline(x=0.60, color="orange", linestyle="--", linewidth=0.8, alpha=0.5)
            ax_f.set_xlabel("ICC(3,1)", fontsize=config.FONT_SIZE_LABEL)
            ax_f.set_title("D  Test-Retest Reliability", fontsize=config.FONT_SIZE_TITLE,
                           fontweight="bold", loc="left")
            ax_f.set_xlim(0, 1)
            ax_f.spines["top"].set_visible(False)
            ax_f.spines["right"].set_visible(False)

    plt.suptitle("Network Reorganization Across Sleep Transitions: Summary",
                fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    for fmt in config.FIGURE_FORMAT:
        fig.savefig(str(fig_dir / f"fig7_summary_trajectory.{fmt}"),
                   dpi=config.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ Figure 7: Summary trajectory")


def main():
    print("=" * 70)
    print("SLEEP NETWORK REORGANIZATION — SUMMARY FIGURES")
    print("=" * 70)

    config.setup_directories()

    print("\nGenerating summary figures...")
    create_study_overview_figure()
    create_summary_trajectory_figure()

    # List all generated figures
    fig_files = sorted(config.FIGURES_DIR.glob("*"))
    print(f"\nAll figures in {config.FIGURES_DIR}:")
    for f in fig_files:
        size_kb = f.stat().st_size / 1024 if f.exists() else 0
        print(f"  {f.name} ({size_kb:.1f} KB)")

    print(f"\n✓ Summary visualization complete.")


if __name__ == "__main__":
    main()
