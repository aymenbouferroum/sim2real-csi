"""Faithful reimplementations of published CSI augmentation methods.

All are subcarrier-flexible ([N, ..., K, 2] real/imag) and seeded.
"""

import numpy as np

from mqtc.models.base import SimulationModel


def _complex(csi: np.ndarray) -> np.ndarray:
    return csi[..., 0] + 1j * csi[..., 1]


def _restack(c: np.ndarray) -> np.ndarray:
    return np.stack([c.real, c.imag], axis=-1)


class AirFiAmplitudeNoise(SimulationModel):
    """AirFi (Wang et al., TMC'22) -- additive Gaussian noise on amplitude."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._sigma: np.ndarray | None = None   # (K,)

    def calibrate(self, clean_csi: np.ndarray, jammed_csi: np.ndarray) -> None:
        k = clean_csi.shape[-2]
        clean_amp = np.abs(_complex(clean_csi)).reshape(-1, k)
        mod_amp = np.abs(_complex(jammed_csi)).reshape(-1, k)
        var_gap = np.var(mod_amp, axis=0) - np.var(clean_amp, axis=0)
        self._sigma = np.sqrt(np.maximum(var_gap, 0.0))

    def simulate(self, clean_csi: np.ndarray) -> np.ndarray:
        assert self._sigma is not None, "Model not calibrated."
        rng = np.random.default_rng(self.seed)
        complex_csi = _complex(clean_csi)
        amp = np.abs(complex_csi)
        phase = np.angle(complex_csi)
        noise = rng.normal(0.0, 1.0, amp.shape) * self._sigma
        new_amp = np.maximum(amp + noise, 0.0)
        return _restack(new_amp * np.exp(1j * phase))

    def get_params(self) -> dict:
        return {"model": "AirFi_AmplNoise",
                "mean_sigma": float(np.mean(self._sigma)) if self._sigma is not None else None,
                "seed": self.seed}


class ProFiNetGaussian(SimulationModel):
    """ProFiNet (2025) -- additive complex Gaussian, std = level * std(signal)."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._level: float = 0.1
        self._sigma_elem: float = 0.0

    def calibrate(self, clean_csi: np.ndarray, jammed_csi: np.ndarray) -> None:
        clean_c = _complex(clean_csi)
        mod_c = _complex(jammed_csi)
        p_clean = np.mean(np.abs(clean_c) ** 2)
        p_mod = np.mean(np.abs(mod_c) ** 2)
        noise_power = max(p_mod - p_clean, 0.0)
        self._sigma_elem = float(np.sqrt(noise_power / 2.0))
        std_signal = float(np.std(clean_c.real))
        self._level = self._sigma_elem / std_signal if std_signal > 0 else 0.0

    def simulate(self, clean_csi: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        complex_csi = _complex(clean_csi)
        noise = (rng.normal(0.0, self._sigma_elem, complex_csi.shape)
                 + 1j * rng.normal(0.0, self._sigma_elem, complex_csi.shape))
        return _restack(complex_csi + noise)

    def get_params(self) -> dict:
        return {"model": "ProFiNet_Gaussian", "level": self._level,
                "sigma_elem": self._sigma_elem, "seed": self.seed}


class MomentMatchGaussian(SimulationModel):
    """Correlation-aware stress-test baseline (not from a single paper)."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._mean: np.ndarray | None = None
        self._chol: np.ndarray | None = None
        self._phase_shift: np.ndarray | None = None

    def calibrate(self, clean_csi: np.ndarray, jammed_csi: np.ndarray) -> None:
        k = clean_csi.shape[-2]
        clean_c = _complex(clean_csi).reshape(-1, k)
        mod_c = _complex(jammed_csi).reshape(-1, k)
        mod_amp = np.abs(mod_c)
        self._mean = mod_amp.mean(axis=0)
        cov = np.cov(mod_amp.T) + 1e-6 * np.eye(k)
        try:
            self._chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            self._chol = np.diag(np.sqrt(np.maximum(np.diag(cov), 1e-12)))
        n = min(clean_c.shape[0], mod_c.shape[0])
        phase_diff = np.angle(mod_c[:n]) - np.angle(clean_c[:n])
        self._phase_shift = np.angle(np.mean(np.exp(1j * phase_diff), axis=0))

    def simulate(self, clean_csi: np.ndarray) -> np.ndarray:
        assert self._mean is not None and self._chol is not None
        assert self._phase_shift is not None
        rng = np.random.default_rng(self.seed)
        complex_csi = _complex(clean_csi)
        orig = complex_csi.shape
        k = complex_csi.shape[-1]
        n = int(np.prod(orig[:-1]))
        z = rng.standard_normal((n, k))
        amp = self._mean + z @ self._chol.T
        amp = np.maximum(amp, 0.0).reshape(orig)
        phase = np.angle(complex_csi) + self._phase_shift
        return _restack(amp * np.exp(1j * phase))

    def get_params(self) -> dict:
        return {"model": "MomentMatch_Gaussian", "seed": self.seed,
                "calibrated": self._mean is not None}


class StrohmayerRandomAmpl(SimulationModel):
    """Strohmayer & Kampel (AIAI'24) -- randomAmplitude augmentation."""

    def __init__(self, seed: int = 42, low: float = 0.75, high: float = 1.25):
        self.seed = seed
        self.low = low
        self.high = high

    def calibrate(self, clean_csi: np.ndarray, jammed_csi: np.ndarray) -> None:
        return None

    def simulate(self, clean_csi: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        complex_csi = _complex(clean_csi)
        n_win = complex_csi.shape[0]
        s = rng.uniform(self.low, self.high, size=(n_win,) + (1,) * (complex_csi.ndim - 1))
        amp = np.abs(complex_csi) * s
        phase = np.angle(complex_csi)
        return _restack(amp * np.exp(1j * phase))

    def get_params(self) -> dict:
        return {"model": "Strohmayer_RandomAmpl", "range": [self.low, self.high], "seed": self.seed}


class SerbetciRandomPhaseAmpl(SimulationModel):
    """Serbetci et al. -- random phase rotation + dB amplitude shift."""

    def __init__(self, seed: int = 42, p_db: float = 1.0):
        self.seed = seed
        self.p_db = p_db

    def calibrate(self, clean_csi: np.ndarray, jammed_csi: np.ndarray) -> None:
        return None

    def simulate(self, clean_csi: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        complex_csi = _complex(clean_csi)
        n_win = complex_csi.shape[0]
        bshape = (n_win,) + (1,) * (complex_csi.ndim - 1)
        phi = rng.uniform(0.0, 2.0 * np.pi, size=bshape)
        alpha_db = rng.uniform(-self.p_db, self.p_db, size=bshape)
        scale = 10.0 ** (alpha_db / 20.0)
        out = complex_csi * scale * np.exp(1j * phi)
        return _restack(out)

    def get_params(self) -> dict:
        return {"model": "Serbetci_RandomPhaseAmpl", "p_db": self.p_db, "seed": self.seed}
