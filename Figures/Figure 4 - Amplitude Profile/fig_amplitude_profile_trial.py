"""Mean amplitude profile across 52 subcarriers (Figure 4).
Data: controlled room, constant Gaussian jamming at 20 dB.
"""

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

from mqtc.data.loaders import load_npz_csi
from mqtc.models import M1AWGN, MQTC
from _style import (
    CIT, CIT_RC, FIG_W, FIG_H, FS_LABEL,
    LINE_W, MARK_S, MARK_EW,
    cit_ax, cit_legend,
)

# Data paths
DATA_DIR = REPO_ROOT / "data" / "controlled" / "constant_gaussian_20dB"
BASELINE_PATH = str(DATA_DIR / "baseline.npz")
JAMMED_PATH = str(DATA_DIR / "jammed.npz")

MARKER_EVERY = 4


def mean_amplitude_profile(csi_raw: np.ndarray) -> np.ndarray:
    """Mean amplitude per subcarrier from raw [N, 32, 52, 2] data."""
    csi_complex = csi_raw[..., 0] + 1j * csi_raw[..., 1]
    return np.abs(csi_complex).mean(axis=(0, 1))


def main():
    clean = load_npz_csi(BASELINE_PATH)
    jammed = load_npz_csi(JAMMED_PATH)

    n_min = min(clean.shape[0], jammed.shape[0])
    clean, jammed = clean[:n_min], jammed[:n_min]
    print(f"Windows: {n_min}")

    m1 = M1AWGN(seed=42)
    m1.calibrate(clean, jammed)
    sim_m1 = m1.simulate(clean)

    qtc = MQTC(seed=42)
    qtc.calibrate(clean, jammed)
    sim_qtc = qtc.simulate(clean)

    profile_real = mean_amplitude_profile(jammed)
    profile_m1 = mean_amplitude_profile(sim_m1)
    profile_qtc = mean_amplitude_profile(sim_qtc)
    subcarriers = np.arange(52)

    # Plot
    plt.rcParams.update(CIT_RC)
    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, FIG_H))

    ax.plot(
        subcarriers, profile_real,
        color=CIT["red"], ls="--", dashes=(5, 2.5), lw=LINE_W,
        marker="o", markersize=MARK_S * 0.65, markeredgewidth=MARK_EW,
        markerfacecolor=CIT["red"], markeredgecolor=CIT["red"],
        markevery=MARKER_EVERY, label="Real jammed", zorder=3,
    )
    ax.plot(
        subcarriers, profile_m1,
        color=CIT["tan"], ls="--", dashes=(5, 2.5), lw=LINE_W,
        marker="s", markersize=MARK_S * 0.6, markeredgewidth=MARK_EW,
        markerfacecolor=CIT["tan"], markeredgecolor=CIT["tan"],
        markevery=MARKER_EVERY, label="M1 (AWGN)", zorder=2,
    )
    ax.plot(
        subcarriers, profile_qtc,
        color=CIT["green"], ls="--", dashes=(5, 2.5), lw=LINE_W,
        marker="^", markersize=MARK_S * 0.65, markeredgewidth=MARK_EW,
        markerfacecolor=CIT["green"], markeredgecolor=CIT["green"],
        markevery=MARKER_EVERY, label=r"M$_{\mathrm{QTC}}$", zorder=4,
    )

    ax.set_xlabel("Subcarrier", fontsize=FS_LABEL, labelpad=2)
    ax.set_ylabel("Amplitude", fontsize=FS_LABEL, labelpad=2)
    ax.set_xlim(-1, 52)
    cit_ax(ax)

    handles, labels = ax.get_legend_handles_labels()
    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.24, top=0.78)
    cit_legend(fig, handles, labels, ncol=3)

    out_path = SCRIPT_DIR / "fig_amplitude_profile_trial.png"
    fig.savefig(out_path, format="png", dpi=300, bbox_inches=None,
                facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
