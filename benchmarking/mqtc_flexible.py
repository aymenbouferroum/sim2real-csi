"""Subcarrier-flexible M_QTC for cross-dataset benchmarking.

Infers K from input so the same algorithm runs on any dataset (ESP32=52, Intel 5300=30, etc.).
"""

import numpy as np
from scipy.signal import lfilter
from statsmodels.tsa.stattools import acf

from mqtc.models.base import SimulationModel

N_QUANTILES = 1000


class MQTCFlexible(SimulationModel):
    """Quantile-Temporal-Copula model with inferred subcarrier count."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._n_sub: int | None = None
        self._clean_quantiles: np.ndarray | None = None
        self._jammed_quantiles: np.ndarray | None = None
        self._phi: float = 0.0
        self._residual_std: np.ndarray | None = None
        self._target_cholesky: np.ndarray | None = None
        self._phase_shift: np.ndarray | None = None

    def calibrate(self, clean_csi: np.ndarray, jammed_csi: np.ndarray) -> None:
        k = clean_csi.shape[-2]
        self._n_sub = k

        clean_c = (clean_csi[..., 0] + 1j * clean_csi[..., 1]).reshape(-1, k)
        jammed_c = (jammed_csi[..., 0] + 1j * jammed_csi[..., 1]).reshape(-1, k)

        clean_amp = np.abs(clean_c)
        jammed_amp = np.abs(jammed_c)
        clean_phase = np.angle(clean_c)
        jammed_phase = np.angle(jammed_c)

        # per-subcarrier quantile mapping tables
        quantile_grid = np.linspace(0, 1, N_QUANTILES)
        self._clean_quantiles = np.empty((k, N_QUANTILES))
        self._jammed_quantiles = np.empty((k, N_QUANTILES))
        for s in range(k):
            self._clean_quantiles[s] = np.quantile(clean_amp[:, s], quantile_grid)
            self._jammed_quantiles[s] = np.quantile(jammed_amp[:, s], quantile_grid)

        # AR(1) coefficient from residuals of quantile-mapped clean vs jammed
        n_min = min(clean_amp.shape[0], jammed_amp.shape[0])
        mapped_clean = self._apply_quantile_map(clean_amp[:n_min])
        residuals = jammed_amp[:n_min] - mapped_clean

        residual_std = np.std(residuals, axis=0)
        self._residual_std = np.maximum(residual_std, 1e-10)

        lag1_acfs = []
        for s in range(k):
            series = residuals[:, s]
            if np.std(series) < 1e-10:
                continue
            acf_vals = acf(series, nlags=1, fft=True)
            lag1_acfs.append(acf_vals[1])
        self._phi = float(np.clip(np.median(lag1_acfs), 0.0, 0.99)) if lag1_acfs else 0.0

        # target correlation matrix from jammed amplitudes (Iman-Conover)
        target_corr = np.corrcoef(jammed_amp.T)
        try:
            self._target_cholesky = np.linalg.cholesky(
                target_corr + 1e-6 * np.eye(k)
            )
        except np.linalg.LinAlgError:
            self._target_cholesky = None

        # mean circular phase shift per subcarrier
        n_phase = min(clean_phase.shape[0], jammed_phase.shape[0])
        phase_diff = jammed_phase[:n_phase] - clean_phase[:n_phase]
        self._phase_shift = np.angle(np.mean(np.exp(1j * phase_diff), axis=0))

    def simulate(self, clean_csi: np.ndarray) -> np.ndarray:
        if (
            self._clean_quantiles is None
            or self._jammed_quantiles is None
            or self._residual_std is None
            or self._phase_shift is None
        ):
            raise RuntimeError("Model not calibrated. Call calibrate() first.")

        k = self._clean_quantiles.shape[0]
        rng = np.random.default_rng(self.seed)
        complex_csi = clean_csi[..., 0] + 1j * clean_csi[..., 1]
        orig_shape = complex_csi.shape

        clean_amp = np.abs(complex_csi)
        clean_phase = np.angle(complex_csi)

        flat_amp = clean_amp.reshape(-1, k)
        n_samples = flat_amp.shape[0]

        # quantile mapping
        mapped_amp = self._apply_quantile_map(flat_amp)

        # AR(1)-filtered, Cholesky-correlated rank template
        phi = self._phi
        innovation_scale = np.sqrt(max(1.0 - phi ** 2, 0.0))

        z_indep = rng.normal(0, 1, (n_samples, k))
        if self._target_cholesky is not None:
            z_corr = z_indep @ self._target_cholesky.T
        else:
            z_corr = z_indep
        z_corr *= innovation_scale

        rank_scores = lfilter([1.0], [1.0, -phi], z_corr, axis=0)

        # Iman-Conover reordering
        result_amp = np.empty_like(mapped_amp)
        for s in range(k):
            target_ranks = np.argsort(np.argsort(rank_scores[:, s]))
            sorted_vals = np.sort(mapped_amp[:, s])
            result_amp[:, s] = sorted_vals[target_ranks]

        sim_phase = clean_phase.reshape(-1, k) + self._phase_shift
        sim_complex = result_amp * np.exp(1j * sim_phase)
        sim_complex = sim_complex.reshape(orig_shape)

        return np.stack([sim_complex.real, sim_complex.imag], axis=-1)

    def get_params(self) -> dict:
        return {
            "model": "M_QTC_flexible",
            "n_sub": self._n_sub,
            "phi": self._phi,
            "n_quantiles": N_QUANTILES,
            "has_cholesky": self._target_cholesky is not None,
            "calibrated": self._clean_quantiles is not None,
            "seed": self.seed,
        }

    def _apply_quantile_map(self, amp: np.ndarray) -> np.ndarray:
        """Map amplitudes through per-subcarrier quantile transfer."""
        assert self._clean_quantiles is not None
        assert self._jammed_quantiles is not None
        k = self._clean_quantiles.shape[0]
        quantile_grid = np.linspace(0, 1, N_QUANTILES)
        mapped = np.empty_like(amp)
        for s in range(k):
            ranks = np.interp(amp[:, s], self._clean_quantiles[s], quantile_grid)
            ranks = np.clip(ranks, 0.0, 1.0)
            mapped[:, s] = np.interp(ranks, quantile_grid, self._jammed_quantiles[s])
        return mapped
