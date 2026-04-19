"""Plot helpers: always save PNG + PDF to `results/figures/{png,pdf}/<subdir>`."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

from eeg_analysis import config as cfg


def save_fig(fig: plt.Figure, name: str, subdir: str = "group",
             dpi: int = cfg.FIG_DPI, close: bool = True) -> tuple[Path, Path]:
    png = cfg.png_path(name, subdir=subdir)
    pdf = cfg.pdf_path(name, subdir=subdir)
    fig.savefig(png, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    if close:
        plt.close(fig)
    return png, pdf


def default_style() -> None:
    plt.rcParams.update({
        "figure.dpi": cfg.FIG_DPI,
        "savefig.dpi": cfg.FIG_DPI,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
