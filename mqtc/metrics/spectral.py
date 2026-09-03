"""Cross-subcarrier correlation matrix distance (Frobenius norm)."""

import numpy as np


def correlation_matrix_distance(
    sim_mag: np.ndarray,
    real_mag: np.ndarray,
) -> float:
    """Frobenius distance between 52x52 cross-subcarrier correlation matrices."""
    # each row of .T is one subcarrier's time series -> corrcoef gives 52x52
    c_sim = np.corrcoef(sim_mag.T)
    c_real = np.corrcoef(real_mag.T)
    return float(np.linalg.norm(c_real - c_sim, "fro"))
