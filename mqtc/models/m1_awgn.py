"""M1: Additive White Gaussian Noise baseline (Section III-A, Eq. 3)."""

import numpy as np

from mqtc.models.base import SimulationModel


class M1AWGN(SimulationModel):
    """AWGN baseline that adds complex Gaussian noise at a calibrated SNR."""

    def __init__(self, snr_db: float = 10.0, seed: int = 42):
        self.snr_db = snr_db
        self.seed = seed
        self._calibrated_snr: float | None = None

    def calibrate(self, clean_csi: np.ndarray, jammed_csi: np.ndarray) -> None:
        """Estimate SNR from power difference between clean and jammed recordings."""
        clean_complex = clean_csi[..., 0] + 1j * clean_csi[..., 1]
        jammed_complex = jammed_csi[..., 0] + 1j * jammed_csi[..., 1]
        signal_power = np.mean(np.abs(clean_complex) ** 2)
        jammed_power = np.mean(np.abs(jammed_complex) ** 2)
        noise_power = max(jammed_power - signal_power, 1e-10)
        self._calibrated_snr = float(10.0 * np.log10(signal_power / noise_power))

    def simulate(self, clean_csi: np.ndarray) -> np.ndarray:
        """Apply AWGN at the calibrated or specified SNR."""
        snr = self._calibrated_snr if self._calibrated_snr is not None else self.snr_db
        rng = np.random.default_rng(self.seed)

        complex_csi = clean_csi[..., 0] + 1j * clean_csi[..., 1]
        signal_power = np.mean(np.abs(complex_csi) ** 2)
        noise_power = signal_power / (10.0 ** (snr / 10.0))
        noise_std = np.sqrt(noise_power / 2.0)  # /2 because power splits between real and imag

        noise = (
            rng.normal(0, noise_std, complex_csi.shape)
            + 1j * rng.normal(0, noise_std, complex_csi.shape)
        )
        jammed = complex_csi + noise
        return np.stack([jammed.real, jammed.imag], axis=-1)

    def get_params(self) -> dict:
        """Return model parameters."""
        return {
            "model": "M1_AWGN",
            "snr_db": float(
                self._calibrated_snr
                if self._calibrated_snr is not None
                else self.snr_db
            ),
            "calibrated": self._calibrated_snr is not None,
            "seed": self.seed,
        }
