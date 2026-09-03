"""Shared figure styling for the M_QTC paper."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Repository root (two levels up from Figures/_style.py)
REPO_ROOT = Path(__file__).resolve().parent.parent

# Figure dimensions (IEEE single-column width)
FIG_W, FIG_H = 3.5, 1.45

# Font sizes
FS_TITLE = 7.0
FS_LABEL = 7.0
FS_TICK = 7.0
FS_LEG = 7.0

# Spine, tick, line geometry
SPINE_W = 1.4
TICK_W = 1.2
TICK_LEN = 3.5
LINE_W = 1.8
MARK_S = 5.5
MARK_EW = 0.8

# DPI settings
DPI_PNG = 300
DPI_PDF = 600

# Color palette
CIT = {
    "blue": "#002e99",
    "red": "#d32f2f",
    "tan": "#f5bc7a",
    "green": "#2e7d32",
    "gray": "#333333",
    "lblue": "#7e8ec1",
    "brown": "#d97e3d",
}

# Model-specific colors and labels
MODEL_COLORS = {
    "M1_AWGN": CIT["tan"],
    "M2_Hybrid": "#4450b1",
    "M3_Copula": CIT["brown"],
    "M_QTC": CIT["green"],
}

MODEL_LABELS = {
    "M1_AWGN": "M1 (AWGN)",
    "M3_Hybrid": "M2 (Hybrid)",
    "M6_Copula": "M3 (Copula)",
    "M_QTC": r"M$_\mathrm{QTC}$",
}

MODEL_ORDER = ["M1_AWGN", "M3_Hybrid", "M6_Copula", "M_QTC"]

# rcParams for paper figures
CIT_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FS_TICK,
    "axes.titlesize": FS_TITLE,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
    "legend.fontsize": FS_LEG,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": SPINE_W,
    "axes.grid": False,
    "xtick.major.width": TICK_W,
    "ytick.major.width": TICK_W,
    "lines.linewidth": LINE_W,
    "lines.markersize": MARK_S,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
}


def setup_style():
    """Apply the paper rcParams globally."""
    plt.rcParams.update(CIT_RC)


def cit_ax(ax):
    """Bold bottom+left spines, no top/right."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_linewidth(SPINE_W)
    ax.tick_params(
        axis="both", labelsize=FS_TICK, length=TICK_LEN, width=TICK_W
    )


def cit_legend(fig, handles, labels, ncol):
    """Top-centre framed legend."""
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=ncol,
        fontsize=FS_LEG,
        frameon=True,
        fancybox=True,
        edgecolor="black",
        facecolor="white",
        framealpha=0.95,
        handlelength=1.8,
        columnspacing=1.1,
        handletextpad=0.5,
        borderpad=0.4,
    )
    leg.get_frame().set_linewidth(1.0)
    return leg


def save_figure(fig, name, output_dir=None, bbox_inches="tight"):
    """Save figure as PNG + PDF, then close."""
    if output_dir is None:
        out = REPO_ROOT / "Figures"
    else:
        out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        out / f"{name}.png",
        format="png",
        dpi=DPI_PNG,
        bbox_inches=bbox_inches,
        facecolor="white",
    )
    fig.savefig(
        out / f"{name}.pdf",
        format="pdf",
        dpi=DPI_PDF,
        bbox_inches=bbox_inches,
    )
    plt.close(fig)
    print(f"  Saved {name}.pdf ({DPI_PDF} DPI) + {name}.png ({DPI_PNG} DPI)")
