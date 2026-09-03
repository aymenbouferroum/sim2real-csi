"""Weighted composite of amplitude, phase, temporal, and spectral fidelity.

Lower aggregate score means better sim-to-real match.
"""

import numpy as np

from mqtc.metrics.amplitude import wasserstein_per_subcarrier
from mqtc.metrics.phase import circular_variance_per_subcarrier
from mqtc.metrics.spectral import correlation_matrix_distance
from mqtc.metrics.temporal import acf_difference_per_subcarrier

_DEFAULT_WEIGHTS = {
    "amplitude": 0.25,
    "phase": 0.25,
    "temporal": 0.25,
    "spectral": 0.25,
}


def aggregate_fidelity_score(
    sim_csi: np.ndarray,
    real_csi: np.ndarray,
    weights: dict | None = None,
) -> dict:
    """Weighted fidelity score across all four metric dimensions."""
    if weights is None:
        weights = _DEFAULT_WEIGHTS.copy()

    # 4D [N,32,52,2] -> extract mag and phase; 2D -> mag only
    if sim_csi.ndim == 4:
        sim_complex = sim_csi[..., 0] + 1j * sim_csi[..., 1]
        real_complex = real_csi[..., 0] + 1j * real_csi[..., 1]
        sim_mag = np.abs(sim_complex).reshape(-1, sim_csi.shape[2])
        real_mag = np.abs(real_complex).reshape(-1, real_csi.shape[2])
        sim_phase = np.angle(sim_complex).reshape(-1, sim_csi.shape[2])
        real_phase = np.angle(real_complex).reshape(-1, real_csi.shape[2])
        has_phase = True
    else:
        sim_mag = sim_csi
        real_mag = real_csi
        has_phase = False

    amplitude_score = float(np.mean(wasserstein_per_subcarrier(sim_mag, real_mag)))

    if has_phase:
        cv_sim = circular_variance_per_subcarrier(sim_phase)
        cv_real = circular_variance_per_subcarrier(real_phase)
        phase_score = float(np.mean(np.abs(cv_sim - cv_real)))
    else:
        phase_score = 0.0

    t_min = min(sim_mag.shape[0], real_mag.shape[0])
    nlags = min(50, t_min // 2 - 1)  # ACF needs at least 2*nlags samples
    temporal_score = float(
        np.mean(acf_difference_per_subcarrier(sim_mag, real_mag, nlags=nlags))
    )

    spectral_score = float(correlation_matrix_distance(sim_mag, real_mag))

    scores = {
        "amplitude": amplitude_score,
        "phase": phase_score,
        "temporal": temporal_score,
        "spectral": spectral_score,
    }
    aggregate = sum(weights[dim] * scores[dim] for dim in weights)

    return {
        "amplitude": amplitude_score,
        "phase": phase_score,
        "temporal": temporal_score,
        "spectral": spectral_score,
        "aggregate": float(aggregate),
        "weights": weights,
    }
