"""Per-subcarrier ACF difference metric for temporal fidelity."""

import numpy as np
from statsmodels.tsa.stattools import acf


def acf_difference_per_subcarrier(
    sim_series: np.ndarray,
    real_series: np.ndarray,
    nlags: int = 50,
) -> np.ndarray:
    """L2 norm of ACF difference per subcarrier between two time series."""
    n_subcarriers = sim_series.shape[1]
    diffs = np.empty(n_subcarriers)

    for sc in range(n_subcarriers):
        acf_sim = acf(sim_series[:, sc], nlags=nlags, fft=True)
        acf_real = acf(real_series[:, sc], nlags=nlags, fft=True)
        diffs[sc] = np.linalg.norm(acf_sim - acf_real)

    return diffs
