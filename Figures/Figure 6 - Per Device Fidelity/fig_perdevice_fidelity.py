"""Per-device fidelity comparison (Figure 6).
Data: pre-computed per-device JSON files from within-device evaluation.
"""

import glob
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FIGURES_DIR = SCRIPT_DIR.parent
REPO_ROOT = FIGURES_DIR.parent

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FIGURES_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from _style import (
    CIT, CIT_RC, FIG_W, FIG_H, FS_LABEL, FS_TICK,
    MARK_EW, cit_ax, cit_legend,
)

# Data paths
RESULTS_DIR = REPO_ROOT / "results" / "per_device_20dB"


def main():
    devices, mqtc_vals, awgn_vals = [], [], []
    for f in sorted(glob.glob(str(RESULTS_DIR / "*.json"))):
        d = json.load(open(f))
        fname = Path(f).stem
        devices.append(fname.split("node-")[1].split("-")[0])
        mqtc_vals.append(d["M_QTC"]["mean"]["aggregate"])
        awgn_vals.append(d["M1_AWGN"]["mean"]["aggregate"])

    # Plot
    plt.rcParams.update(CIT_RC)
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    x = np.arange(len(devices))

    for xi, (mq, aw) in enumerate(zip(mqtc_vals, awgn_vals)):
        ax.plot([xi - 0.08, xi + 0.08], [mq, aw], color="#bbbbbb",
                lw=1.0, zorder=2)

    ax.scatter(x - 0.08, mqtc_vals, s=46, color=CIT["blue"], marker="o",
               edgecolors="white", linewidths=MARK_EW, zorder=3)
    ax.scatter(x + 0.08, awgn_vals, s=46, color=CIT["red"], marker="s",
               edgecolors="white", linewidths=MARK_EW, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"RX{i}" for i in range(1, len(devices) + 1)],
                       fontsize=FS_TICK)
    ax.set_xlabel("Receiver", fontsize=FS_LABEL, labelpad=2)
    ax.set_ylabel("Aggregate fidelity", fontsize=FS_LABEL)
    ax.set_ylim(0, 6.3)
    cit_ax(ax)

    handles = [
        Line2D([], [], color=CIT["red"], marker="s", ls="none", ms=6,
               markeredgecolor="white", markeredgewidth=MARK_EW,
               label=r"M$_1$ (AWGN)"),
        Line2D([], [], color=CIT["blue"], marker="o", ls="none", ms=6,
               markeredgecolor="white", markeredgewidth=MARK_EW,
               label=r"M$_{\mathrm{QTC}}$"),
    ]

    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.24, top=0.82)
    cit_legend(fig, handles, [h.get_label() for h in handles], ncol=2)

    out = SCRIPT_DIR / "fig_perdevice_detection.png"
    fig.savefig(out, format="png", dpi=300, bbox_inches=None, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
