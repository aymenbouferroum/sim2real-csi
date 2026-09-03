"""Abstract base class for CSI simulation models."""

from abc import ABC, abstractmethod

import numpy as np


class SimulationModel(ABC):
    """Calibrate-simulate-evaluate interface for CSI simulation.

    Input/output: np.ndarray of shape [N, 32, 52, 2] (last dim = real/imag).
    """

    @abstractmethod
    def calibrate(self, clean_csi: np.ndarray, jammed_csi: np.ndarray) -> None:
        """Fit model parameters from paired clean/jammed observations."""
        ...

    @abstractmethod
    def simulate(self, clean_csi: np.ndarray) -> np.ndarray:
        """Transform clean CSI into simulated jammed CSI."""
        ...

    @abstractmethod
    def get_params(self) -> dict:
        """Return fitted parameters for serialization."""
        ...
