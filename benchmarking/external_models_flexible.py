"""Subcarrier-flexible versions of calibrated Serbetci and Strohmayer methods."""

import numpy as np

from mqtc.models.base import SimulationModel


def _to_complex_flat(csi: np.ndarray) -> tuple[np.ndarray, int]:
    """Return (flattened complex CSI [M, K], K) from [N, ..., K, 2]."""
    k = csi.shape[-2]
    c = (csi[..., 0] + 1j * csi[..., 1]).reshape(-1, k)
    return c, k


class SerbetciPhaseRotationFlex(SimulationModel):
    """Serbetci et al. -- calibrated global phase rotation."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._phase_shift: float = 0.0

    def calibrate(self, clean_csi: np.ndarray, jammed_csi: np.ndarray) -> None:
        clean_c, _ = _to_complex_flat(clean_csi)
        jammed_c, _ = _to_complex_flat(jammed_csi)
        n = min(clean_c.shape[0], jammed_c.shape[0])
        diff = jammed_c[:n] / (clean_c[:n] + 1e-10)
        self._phase_shift = float(np.angle(np.mean(diff)))

    def simulate(self, clean_csi: np.ndarray) -> np.ndarray:
        complex_csi = clean_csi[..., 0] + 1j * clean_csi[..., 1]
        rotated = complex_csi * np.exp(1j * self._phase_shift)
        return np.stack([rotated.real, rotated.imag], axis=-1)

    def get_params(self) -> dict:
        return {"model": "Serbetci_Phase", "phase_shift_rad": self._phase_shift, "seed": self.seed}


class SerbetciAmplitudeShiftFlex(SimulationModel):
    """Serbetci et al. -- calibrated global dB amplitude shift."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._db_shift: float = 0.0

    def calibrate(self, clean_csi: np.ndarray, jammed_csi: np.ndarray) -> None:
        clean_c, _ = _to_complex_flat(clean_csi)
        jammed_c, _ = _to_complex_flat(jammed_csi)
        clean_power = np.mean(np.abs(clean_c) ** 2)
        jammed_power = np.mean(np.abs(jammed_c) ** 2)
        self._db_shift = float(10 * np.log10(jammed_power / (clean_power + 1e-10)))

    def simulate(self, clean_csi: np.ndarray) -> np.ndarray:
        complex_csi = clean_csi[..., 0] + 1j * clean_csi[..., 1]
        scale = 10 ** (self._db_shift / 20.0)
        scaled = complex_csi * scale
        return np.stack([scaled.real, scaled.imag], axis=-1)

    def get_params(self) -> dict:
        return {"model": "Serbetci_Amplitude", "db_shift": self._db_shift, "seed": self.seed}


class SerbetciCombinedFlex(SimulationModel):
    """Serbetci et al. -- combined phase rotation + amplitude shift."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._phase = SerbetciPhaseRotationFlex(seed)
        self._amp = SerbetciAmplitudeShiftFlex(seed)

    def calibrate(self, clean_csi: np.ndarray, jammed_csi: np.ndarray) -> None:
        self._phase.calibrate(clean_csi, jammed_csi)
        self._amp.calibrate(clean_csi, jammed_csi)

    def simulate(self, clean_csi: np.ndarray) -> np.ndarray:
        complex_csi = clean_csi[..., 0] + 1j * clean_csi[..., 1]
        scale = 10 ** (self._amp._db_shift / 20.0)
        transformed = complex_csi * scale * np.exp(1j * self._phase._phase_shift)
        return np.stack([transformed.real, transformed.imag], axis=-1)

    def get_params(self) -> dict:
        return {
            "model": "Serbetci_Combined",
            "db_shift": self._amp._db_shift,
            "phase_shift_rad": self._phase._phase_shift,
            "seed": self.seed,
        }


class StrohmayerScalingFlex(SimulationModel):
    """Strohmayer & Kampel -- per-subcarrier amplitude scaling."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._scale_factors: np.ndarray | None = None

    def calibrate(self, clean_csi: np.ndarray, jammed_csi: np.ndarray) -> None:
        clean_c, _ = _to_complex_flat(clean_csi)
        jammed_c, _ = _to_complex_flat(jammed_csi)
        clean_amp = np.mean(np.abs(clean_c), axis=0)
        jammed_amp = np.mean(np.abs(jammed_c), axis=0)
        self._scale_factors = jammed_amp / (clean_amp + 1e-10)

    def simulate(self, clean_csi: np.ndarray) -> np.ndarray:
        complex_csi = clean_csi[..., 0] + 1j * clean_csi[..., 1]
        orig_shape = complex_csi.shape
        k = complex_csi.shape[-1]
        flat = complex_csi.reshape(-1, k)
        amp = np.abs(flat)
        phase = np.angle(flat)
        scaled_amp = amp * self._scale_factors
        result = scaled_amp * np.exp(1j * phase)
        result = result.reshape(orig_shape)
        return np.stack([result.real, result.imag], axis=-1)

    def get_params(self) -> dict:
        return {
            "model": "Strohmayer_Scaling",
            "mean_scale": float(np.mean(self._scale_factors)) if self._scale_factors is not None else None,
            "seed": self.seed,
        }
