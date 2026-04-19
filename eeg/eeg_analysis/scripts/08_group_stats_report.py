"""Build the consolidated EEG results narrative: tables + summary figure."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from _common import arg_parser
from eeg_analysis import config as cfg
from eeg_analysis.src.plotting import default_style, save_fig


def _read_csv(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.exists() else None


def main() -> None:
    arg_parser(__doc__).parse_args()
    default_style()

    alpha = _read_csv(cfg.TABLES_DIR / "alpha_wake_vs_n2_stats.csv")
    fooof_occ = _read_csv(cfg.TABLES_DIR / "fooof_occipital_summary.csv")
    bycyc = _read_csv(cfg.TABLES_DIR / "bycycle_wake_vs_n2_stats.csv")
    mt_summ = _read_csv(cfg.TABLES_DIR / "multitaper_band_power_summary.csv")
    group_so = cfg.TABLES_DIR / "group_so_histograms.npz"

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.5, wspace=0.35)

    # (a) Occipital alpha Wake vs N2 effect sizes
    ax = fig.add_subplot(gs[0, 0])
    if alpha is not None:
        sub = alpha.dropna(subset=["hedges_g"])
        sns.barplot(data=sub, x="channel", y="hedges_g", ax=ax, color="#1f77b4")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title("(a) Alpha Wake−N2  (Hedges' g)")
        ax.set_ylabel("g"); ax.set_xlabel("Channel")
    else:
        ax.set_axis_off()

    # (b) FOOOF exponent per stage, occipital
    ax = fig.add_subplot(gs[0, 1])
    if fooof_occ is not None:
        sns.barplot(data=fooof_occ, x="stage", y="exponent_mean",
                    hue="channel", order=cfg.STAGES, ax=ax)
        ax.set_title("(b) Aperiodic exponent (occipital)")
        ax.set_ylabel("exponent"); ax.legend(frameon=False, fontsize=8)
    else:
        ax.set_axis_off()

    # (c) Bycycle rdsym at Oz
    ax = fig.add_subplot(gs[0, 2])
    if bycyc is not None:
        sub = bycyc[bycyc["feature"] == "time_rdsym"]
        xp = np.arange(len(sub))
        ax.bar(xp, sub["Wake_median"], width=0.4, label="Wake", color="#1f77b4")
        ax.bar(xp + 0.4, sub["N2_median"], width=0.4, label="N2", color="#9467bd")
        ax.set_xticks(xp + 0.2); ax.set_xticklabels(sub["channel"])
        ax.set_title("(c) bycycle rise–decay symmetry")
        ax.set_ylabel("rdsym"); ax.legend(frameon=False, fontsize=8)
    else:
        ax.set_axis_off()

    # (d) Multitaper stage-mean band power heatmap (Cz)
    ax = fig.add_subplot(gs[1, 0])
    if mt_summ is not None:
        piv = mt_summ[mt_summ["channel"] == "Cz"].pivot_table(
            index="band", columns="stage", values="mean")
        piv = piv.reindex(index=list(cfg.BANDS.keys()),
                          columns=cfg.STAGES)
        sns.heatmap(10 * np.log10(piv + 1e-24), ax=ax, cmap="magma",
                    cbar_kws={"label": "dB"})
        ax.set_title("(d) Multitaper band power  (Cz)")
    else:
        ax.set_axis_off()

    # (e) Group SO-power histogram at Cz
    ax = fig.add_subplot(gs[1, 1])
    if group_so.exists():
        z = np.load(group_so)
        key = "power_Cz"
        if key in z.files:
            H = z[key]
            pe = z["power_edges"]; fe = z["freq_edges"]
            ax.pcolormesh(fe, pe, H, cmap="viridis", shading="auto")
            ax.set_title("(e) Group SO-power × freq (Cz, N2)")
            ax.set_xlabel("Freq (Hz)"); ax.set_ylabel("SO power %")
    else:
        ax.set_axis_off()

    # (f) Group SO-phase histogram at Cz
    ax = fig.add_subplot(gs[1, 2])
    if group_so.exists():
        z = np.load(group_so)
        key = "phase_Cz"
        if key in z.files:
            H = z[key]
            phe = z["phase_edges"]; fe = z["freq_edges"]
            ax.pcolormesh(fe, phe, H, cmap="twilight", shading="auto")
            ax.set_title("(f) Group SO-phase × freq (Cz, N2)")
            ax.set_xlabel("Freq (Hz)"); ax.set_ylabel("SO phase (rad)")
    else:
        ax.set_axis_off()

    save_fig(fig, "fig_eeg_summary", subdir="group")

    # Markdown narrative
    md_lines = ["# EEG Phase Results — ds003768",
                "",
                "Auto-generated from `results/tables/` + `results/figures/`.",
                ""]
    if alpha is not None:
        md_lines += ["## Wake vs N2 occipital alpha", "",
                     alpha.to_markdown(index=False), ""]
    if fooof_occ is not None:
        md_lines += ["## FOOOF occipital summary", "",
                     fooof_occ.to_markdown(index=False), ""]
    if bycyc is not None:
        md_lines += ["## Bycycle Wake vs N2", "",
                     bycyc.to_markdown(index=False), ""]
    if mt_summ is not None:
        md_lines += ["## Multitaper band-power summary", "",
                     mt_summ.head(40).to_markdown(index=False), ""]
    md_lines += ["## Summary figure",
                 "",
                 "![summary](figures/png/group/fig_eeg_summary.png)",
                 ""]
    out = cfg.RESULTS_DIR / "EEG_RESULTS.md"
    out.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
