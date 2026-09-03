"""M2: Per-subcarrier power-scaled additive noise (Section VI-C ablation)."""

import numpy as np

from mqtc.models.base import SimulationModel


class M2PowerScaled(SimulationModel):
    """Calibrates noise power independently for each of 52 subcarriers."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._noise_power_per_sc: np.ndarray | None = None  # shape (52,)

    def calibrate(self, clean_csi: np.ndarray, jammed_csi: np.ndarray) -> None:
        """Estimate per-subcarrier noise power from power differences."""
        clean_c = (clean_csi[..., 0] + 1j * clean_csi[..., 1]).reshape(-1, 52)
        jammed_c = (jammed_csi[..., 0] + 1j * jammed_csi[..., 1]).reshape(-1, 52)
        clean_pwr = np.mean(np.abs(clean_c) ** 2, axis=0)    # (52,)
        jammed_pwr = np.mean(np.abs(jammed_c) ** 2, axis=0)  # (52,)
        # floor at 1e-10 to avoid zero or negative noise power from estimation noise
        self._noise_power_per_sc = np.maximum(jammed_pwr - clean_pwr, 1e-10)

    def simulate(self, clean_csi: np.ndarray) -> np.ndarray:
        """Apply frequency-selective AWGN at calibrated per-subcarrier power."""
        if self._noise_power_per_sc is None:
            raise RuntimeError("Model not calibrated. Call calibrate() first.")
        rng = np.random.default_rng(self.seed)
        complex_csi = clean_csi[..., 0] + 1j * clean_csi[..., 1]
        orig_shape = complex_csi.shape  # (N, 32, 52)

        noise_std = np.sqrt(self._noise_power_per_sc / 2.0)  # /2: power splits between real and imag

        noise_real = rng.normal(0, 1, orig_shape) * noise_std
        noise_imag = rng.normal(0, 1, orig_shape) * noise_std
        jammed = complex_csi + noise_real + 1j * noise_imag

        return np.stack([jammed.real, jammed.imag], axis=-1)

    def get_params(self) -> dict:
        """Return model parameters."""
        return {
            "model": "M2_PowerScaled",
            "noise_power_per_sc": (
                self._noise_power_per_sc.tolist()
                if self._noise_power_per_sc is not None
                else None
            ),
            "calibrated": self._noise_power_per_sc is not None,
            "seed": self.seed,
        }
