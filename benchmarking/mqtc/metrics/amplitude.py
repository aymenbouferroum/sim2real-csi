"""Per-subcarrier amplitude distance metrics (Wasserstein, KL, KS)."""

import numpy as np
from scipy.stats import entropy, ks_2samp, wasserstein_distance


def kl_divergence_per_subcarrier(
    sim_mag: np.ndarray,
    real_mag: np.ndarray,
    bins: str = "fd",
    epsilon: float = 1e-10,
) -> np.ndarray:
    """KL divergence per subcarrier between magnitude distributions."""
    n_subcarriers = sim_mag.shape[1]
    kl_vals = np.empty(n_subcarriers)

    for sc in range(n_subcarriers):
        # pool both distributions so they share the same bin edges
        combined = np.concatenate([sim_mag[:, sc], real_mag[:, sc]])
        bin_edges = np.histogram_bin_edges(combined, bins=bins)

        p, _ = np.histogram(real_mag[:, sc], bins=bin_edges, density=True)
        q, _ = np.histogram(sim_mag[:, sc], bins=bin_edges, density=True)

        # epsilon-smooth then renormalize to avoid log(0) in KL
        p = (p + epsilon) / (p + epsilon).sum()
        q = (q + epsilon) / (q + epsilon).sum()

        kl_vals[sc] = entropy(p, q)

    return kl_vals


def wasserstein_per_subcarrier(
    sim_mag: np.ndarray,
    real_mag: np.ndarray,
) -> np.ndarray:
    """Wasserstein distance per subcarrier, stable under small samples unlike KL."""
    n_subcarriers = sim_mag.shape[1]
    w_vals = np.empty(n_subcarriers)

    for sc in range(n_subcarriers):
        w_vals[sc] = wasserstein_distance(sim_mag[:, sc], real_mag[:, sc])

    return w_vals


def ks_test_per_subcarrier(
    sim_mag: np.ndarray,
    real_mag: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Two-sample KS test per subcarrier, returns (statistics, pvalues)."""
    n_subcarriers = sim_mag.shape[1]
    stat_vals = np.empty(n_subcarriers)
    pval_vals = np.empty(n_subcarriers)

    for sc in range(n_subcarriers):
        stat, pval = ks_2samp(sim_mag[:, sc], real_mag[:, sc])
        stat_vals[sc] = stat
        pval_vals[sc] = pval

    return stat_vals, pval_vals
