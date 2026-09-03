"""M3: Multiplicative-additive hybrid model (Section VI-C ablation).

H_jam(k) = alpha(k) * H_clean(k) + n(k), where the multiplicative term
captures AGC compression and the additive term captures residual noise.
"""

import numpy as np

from mqtc.models.base import SimulationModel


class M3Hybrid(SimulationModel):
    """Per-subcarrier gain scaling plus correlated residual noise."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._alpha: np.ndarray | None = None  # complex, shape (52,)
        self._noise_power_per_sc: np.ndarray | None = None  # real, shape (52,)
        self._noise_cholesky: np.ndarray | None = None  # (52, 52) lower triangular

    def calibrate(self, clean_csi: np.ndarray, jammed_csi: np.ndarray) -> None:
        """Estimate per-subcarrier alpha and residual noise from power ratios."""
        clean_c = (clean_csi[..., 0] + 1j * clean_csi[..., 1]).reshape(-1, 52)
        jammed_c = (jammed_csi[..., 0] + 1j * jammed_csi[..., 1]).reshape(-1, 52)

        clean_pwr = np.mean(np.abs(clean_c) ** 2, axis=0)    # (52,)
        jammed_pwr = np.mean(np.abs(jammed_c) ** 2, axis=0)  # (52,)

        # Real-valued alpha: complex phase rotation unestimable from sequential data
        pwr_ratio = jammed_pwr / np.maximum(clean_pwr, 1e-10)
        # real-valued alpha stored as complex for multiplication with complex CSI
        self._alpha = np.sqrt(pwr_ratio).astype(np.complex128)

        # Residual noise: excess beyond what scaling explains
        clean_amp = np.abs(clean_c)
        jammed_amp = np.abs(jammed_c)
        scaled_clean_amp = clean_amp * np.sqrt(pwr_ratio)
        N_min = min(clean_amp.shape[0], jammed_amp.shape[0])
        residual_amp = jammed_amp[:N_min] - scaled_clean_amp[:N_min]

        self._noise_power_per_sc = np.maximum(
            np.var(residual_amp, axis=0), 1e-10
        )

        # Cross-subcarrier covariance: broadband jammer affects adjacent
        # subcarriers similarly, creating correlated residuals
        residual_cov = np.cov(residual_amp.T)  # (52, 52)
        residual_cov += 1e-6 * np.eye(52)
        try:
            self._noise_cholesky = np.linalg.cholesky(residual_cov)
        except np.linalg.LinAlgError:
            self._noise_cholesky = None  # fallback to independent noise

    def simulate(self, clean_csi: np.ndarray) -> np.ndarray:
        """Apply multiplicative distortion and correlated additive noise."""
        if self._alpha is None:
            raise RuntimeError("Model not calibrated. Call calibrate() first.")
        rng = np.random.default_rng(self.seed)
        complex_csi = clean_csi[..., 0] + 1j * clean_csi[..., 1]
        orig_shape = complex_csi.shape  # (N, 32, 52)

        scaled = self._alpha * complex_csi

        if self._noise_cholesky is not None:
            # Correlated noise in amplitude domain via Cholesky
            flat_shape = (orig_shape[0] * orig_shape[1], 52)
            white = rng.normal(0, 1, flat_shape)
            corr_noise = white @ self._noise_cholesky.T
            corr_noise = corr_noise.reshape(orig_shape)
            # random phase wraps amplitude-domain noise into complex plane
            jammed = scaled + corr_noise * np.exp(
                1j * rng.uniform(-np.pi, np.pi, orig_shape)
            )
        else:
            noise_std = np.sqrt(self._noise_power_per_sc / 2.0)
            noise = (
                rng.normal(0, 1, orig_shape) * noise_std
                + 1j * rng.normal(0, 1, orig_shape) * noise_std
            )
            jammed = scaled + noise

        return np.stack([jammed.real, jammed.imag], axis=-1)

    def get_params(self) -> dict:
        """Return model parameters."""
        return {
            "model": "M3_Hybrid",
            "alpha_real": (
                self._alpha.real.tolist() if self._alpha is not None else None
            ),
            "alpha_imag": (
                self._alpha.imag.tolist() if self._alpha is not None else None
            ),
            "alpha_magnitude": (
                np.abs(self._alpha).tolist() if self._alpha is not None else None
            ),
            "noise_power_per_sc": (
                self._noise_power_per_sc.tolist()
                if self._noise_power_per_sc is not None
                else None
            ),
            "calibrated": self._alpha is not None,
            "seed": self.seed,
        }
