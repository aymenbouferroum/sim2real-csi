"""NPZ loaders for [N, 32, 52, 2] CSI windows."""

import numpy as np


def load_npz_csi(path: str) -> np.ndarray:
    """Load CSI windows from an NPZ file keyed by 'Xw'."""
    data = np.load(path)
    if "Xw" not in data:
        raise KeyError("NPZ file does not contain 'Xw' key")

    csi = data["Xw"]
    if csi.ndim != 4:
        raise ValueError(
            f"Expected 4-dimensional CSI array, got {csi.ndim} dimensions"
        )
    if csi.shape[-1] != 2:
        raise ValueError(
            f"Expected last dimension to be 2 [real, imag], got {csi.shape[-1]}"
        )

    return csi


def to_complex(csi: np.ndarray) -> np.ndarray:
    """Convert [... , 2] real/imag array to complex."""
    return csi[..., 0] + 1j * csi[..., 1]
