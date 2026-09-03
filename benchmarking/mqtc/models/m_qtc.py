"""M_QTC: Quantile-Temporal-Copula simulation model (Section IV, Algorithm 1)."""

import numpy as np
from scipy.signal import lfilter
from statsmodels.tsa.stattools import acf

from mqtc.models.base import SimulationModel

N_QUANTILES = 1000


class MQTC(SimulationModel):
    """Three-stage pipeline: quantile mapping, AR(1) filtering, Iman-Conover reordering."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._clean_quantiles: np.ndarray | None = None   # (52, N_QUANTILES)
        self._jammed_quantiles: np.ndarray | None = None   # (52, N_QUANTILES)
        self._phi: float = 0.0
        self._residual_std: np.ndarray | None = None       # (52,)
        self._target_cholesky: np.ndarray | None = None     # (52, 52)
        self._phase_shift: np.ndarray | None = None         # (52,)

    def calibrate(self, clean_csi: np.ndarray, jammed_csi: np.ndarray) -> None:
        """Learn quantile tables, AR(1) coefficient, and target correlation matrix."""
        clean_c = (clean_csi[..., 0] + 1j * clean_csi[..., 1]).reshape(-1, 52)
        jammed_c = (jammed_csi[..., 0] + 1j * jammed_csi[..., 1]).reshape(-1, 52)

        clean_amp = np.abs(clean_c)
        jammed_amp = np.abs(jammed_c)
        clean_phase = np.angle(clean_c)
        jammed_phase = np.angle(jammed_c)

        # Stage 1: Per-subcarrier quantile mapping tables
        quantile_grid = np.linspace(0, 1, N_QUANTILES)
        self._clean_quantiles = np.empty((52, N_QUANTILES))
        self._jammed_quantiles = np.empty((52, N_QUANTILES))
        for k in range(52):
            self._clean_quantiles[k] = np.quantile(clean_amp[:, k], quantile_grid)
            self._jammed_quantiles[k] = np.quantile(jammed_amp[:, k], quantile_grid)

        # Stage 2: AR(1) coefficient from quantile-mapped residuals
        n_min = min(clean_amp.shape[0], jammed_amp.shape[0])
        mapped_clean = self._apply_quantile_map(clean_amp[:n_min])
        residuals = jammed_amp[:n_min] - mapped_clean

        residual_std = np.std(residuals, axis=0)
        self._residual_std = np.maximum(residual_std, 1e-10)  # guard against zero-std subcarriers

        lag1_acfs = []
        for k in range(52):
            series = residuals[:, k]
            if np.std(series) < 1e-10:
                continue
            acf_vals = acf(series, nlags=1, fft=True)
            lag1_acfs.append(acf_vals[1])
        # clip to [0, 0.99] for AR(1) stability (phi=1 is a random walk)
        self._phi = float(np.clip(np.median(lag1_acfs), 0.0, 0.99)) if lag1_acfs else 0.0

        # Stage 3: Cholesky factor of jammed correlation for Iman-Conover
        target_corr = np.corrcoef(jammed_amp.T)
        try:
            # small diagonal regularization to ensure positive-definiteness
            self._target_cholesky = np.linalg.cholesky(
                target_corr + 1e-6 * np.eye(52)
            )
        except np.linalg.LinAlgError:
            self._target_cholesky = None

        # Mean circular phase shift per subcarrier
        n_phase = min(clean_phase.shape[0], jammed_phase.shape[0])
        phase_diff = jammed_phase[:n_phase] - clean_phase[:n_phase]
        self._phase_shift = np.angle(np.mean(np.exp(1j * phase_diff), axis=0))

    def simulate(self, clean_csi: np.ndarray) -> np.ndarray:
        """Apply quantile mapping, AR(1) filtering, and Iman-Conover reordering."""
        if (
            self._clean_quantiles is None
            or self._jammed_quantiles is None
            or self._residual_std is None
            or self._phase_shift is None
        ):
            raise RuntimeError("Model not calibrated. Call calibrate() first.")

        rng = np.random.default_rng(self.seed)
        complex_csi = clean_csi[..., 0] + 1j * clean_csi[..., 1]
        orig_shape = complex_csi.shape  # (N, 32, 52)

        clean_amp = np.abs(complex_csi)
        clean_phase = np.angle(complex_csi)

        flat_amp = clean_amp.reshape(-1, 52)
        n_samples = flat_amp.shape[0]

        # Stage 1: Quantile mapping
        mapped_amp = self._apply_quantile_map(flat_amp)

        # Stage 2: AR(1)-filtered, Cholesky-correlated rank template
        phi = self._phi
        innovation_scale = np.sqrt(max(1.0 - phi ** 2, 0.0))  # preserves unit variance under AR(1)

        z_indep = rng.normal(0, 1, (n_samples, 52))
        if self._target_cholesky is not None:
            z_corr = z_indep @ self._target_cholesky.T
        else:
            z_corr = z_indep
        z_corr *= innovation_scale

        # IIR filter implements s[n] = phi*s[n-1] + innovation*z[n]
        rank_scores = lfilter([1.0], [1.0, -phi], z_corr, axis=0)

        # Stage 3: Iman-Conover reordering (marginals preserved exactly)
        result_amp = np.empty_like(mapped_amp)
        for k in range(52):
            target_ranks = np.argsort(np.argsort(rank_scores[:, k]))  # double argsort = rank transform
            sorted_vals = np.sort(mapped_amp[:, k])
            result_amp[:, k] = sorted_vals[target_ranks]

        sim_phase = clean_phase.reshape(-1, 52) + self._phase_shift
        sim_complex = result_amp * np.exp(1j * sim_phase)
        sim_complex = sim_complex.reshape(orig_shape)

        return np.stack([sim_complex.real, sim_complex.imag], axis=-1)

    def get_params(self) -> dict:
        """Return fitted model parameters."""
        return {
            "model": "M_QTC",
            "phi": self._phi,
            "n_quantiles": N_QUANTILES,
            "has_cholesky": self._target_cholesky is not None,
            "calibrated": self._clean_quantiles is not None,
            "seed": self.seed,
        }

    def _apply_quantile_map(self, amp: np.ndarray) -> np.ndarray:
        """Map amplitudes through per-subcarrier quantile transfer tables."""
        assert self._clean_quantiles is not None
        assert self._jammed_quantiles is not None
        quantile_grid = np.linspace(0, 1, N_QUANTILES)
        mapped = np.empty_like(amp)
        for k in range(52):
            # two-step interp: amplitude -> clean quantile rank -> jammed amplitude
            ranks = np.interp(amp[:, k], self._clean_quantiles[k], quantile_grid)
            ranks = np.clip(ranks, 0.0, 1.0)
            mapped[:, k] = np.interp(ranks, quantile_grid, self._jammed_quantiles[k])
        return mapped
