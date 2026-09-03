"""Amplitude density comparison on subcarrier 26 (Figure 5).
Data: controlled room, constant Gaussian jamming at 20 dB, bandwidth 0.3.
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
from scipy.stats import gaussian_kde

from mqtc.data.loaders import load_npz_csi
from mqtc.models import M1AWGN, MQTC
from _style import (
    CIT_RC, FIG_W, FIG_H, FS_TITLE, FS_LABEL,
    LINE_W, cit_ax, cit_legend,
)

# Colors
RED = "#c62828"
GREEN = "#2e7d32"
BLUE = "#1565c0"

DATA_DIR = REPO_ROOT / "data" / "controlled" / "constant_gaussian_20dB"
SC = 26


def main():
    clean = load_npz_csi(str(DATA_DIR / "baseline.npz"))
    jammed = load_npz_csi(str(DATA_DIR / "jammed.npz"))

    n_min = min(clean.shape[0], jammed.shape[0])
    clean, jammed = clean[:n_min], jammed[:n_min]

    m1 = M1AWGN(seed=42)
    m1.calibrate(clean, jammed)
    sim_m1 = m1.simulate(clean)

    qtc = MQTC(seed=42)
    qtc.calibrate(clean, jammed)
    sim_qtc = qtc.simulate(clean)

    # Amplitude for subcarrier SC
    jammed_amp = np.abs(jammed[..., 0] + 1j * jammed[..., 1]).reshape(-1, 52)[:, SC]
    m1_amp = np.abs(sim_m1[..., 0] + 1j * sim_m1[..., 1]).reshape(-1, 52)[:, SC]
    qtc_amp = np.abs(sim_qtc[..., 0] + 1j * sim_qtc[..., 1]).reshape(-1, 52)[:, SC]

    # KDE
    xmin = min(jammed_amp.min(), m1_amp.min(), qtc_amp.min()) - 2
    xmax = max(jammed_amp.max(), m1_amp.max(), qtc_amp.max()) + 2
    xs = np.linspace(xmin, xmax, 500)
    bw = 0.3

    jammed_kde = gaussian_kde(jammed_amp, bw_method=bw)(xs)
    m1_kde = gaussian_kde(m1_amp, bw_method=bw)(xs)
    qtc_kde = gaussian_kde(qtc_amp, bw_method=bw)(xs)

    step = max(1, len(xs) // 20)
    mx, m_jam = xs[::step], jammed_kde[::step]
    m_m1, m_qtc = m1_kde[::step], qtc_kde[::step]

    # Plot
    plt.rcParams.update(CIT_RC)
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), sharey=True)

    # Panel (a): Real jammed + M1
    ax = axes[0]
    ax.plot(xs, jammed_kde, color=RED, ls="--", lw=LINE_W, dashes=(4, 2))
    ax.plot(mx, m_jam, color=RED, ls="none", marker="o", ms=4.5,
            markeredgecolor="white", markeredgewidth=0.4, label="Real jammed")
    ax.plot(xs, m1_kde, color=GREEN, ls="--", lw=LINE_W, dashes=(4, 2))
    ax.plot(mx, m_m1, color=GREEN, ls="none", marker="s", ms=4.5,
            markeredgecolor="white", markeredgewidth=0.4, label=r"M$_1$ (AWGN)")
    ax.set_title("(a) AWGN", fontsize=FS_TITLE, fontweight="bold", pad=3)
    ax.set_xlabel("Amplitude", fontsize=FS_LABEL, labelpad=2)
    ax.set_ylabel("Density", fontsize=FS_LABEL)
    ax.set_yticks([0.0, 0.1])
    cit_ax(ax)

    # Panel (b): Real jammed + M_QTC
    ax = axes[1]
    ax.plot(xs, jammed_kde, color=RED, ls="--", lw=LINE_W, dashes=(4, 2))
    ax.plot(mx, m_jam, color=RED, ls="none", marker="o", ms=4.5,
            markeredgecolor="white", markeredgewidth=0.4)
    ax.plot(xs, qtc_kde, color=BLUE, ls="--", lw=LINE_W, dashes=(4, 2))
    ax.plot(mx, m_qtc, color=BLUE, ls="none", marker="^", ms=4.5,
            markeredgecolor="white", markeredgewidth=0.4, label=r"M$_{\mathrm{QTC}}$")
    ax.set_title(r"(b) M$_{\mathrm{QTC}}$", fontsize=FS_TITLE,
                 fontweight="bold", pad=3)
    ax.set_xlabel("Amplitude", fontsize=FS_LABEL, labelpad=2)
    cit_ax(ax)

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    handles += h2
    labels += l2

    fig.subplots_adjust(left=0.135, right=0.985, bottom=0.24, top=0.73,
                        wspace=0.09)
    cit_legend(fig, handles, labels, ncol=3)

    out = SCRIPT_DIR / "fig_amplitude_kde.png"
    fig.savefig(out, format="png", dpi=300, bbox_inches=None, facecolor="white")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
