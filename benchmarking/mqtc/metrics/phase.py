"""Per-subcarrier circular statistics for CSI phase comparison."""

import numpy as np

try:
    from pycircstat2 import Circular  # noqa: F401 -- presence check
    from pycircstat2.descriptive import circ_r

    _HAS_PYCIRCSTAT2 = True
except ImportError:
    _HAS_PYCIRCSTAT2 = False


def mean_resultant_length_per_subcarrier(phase: np.ndarray) -> np.ndarray:
    """Mean resultant length R per subcarrier (1 = aligned, 0 = uniform)."""
    n_subcarriers = phase.shape[1]
    mrl = np.empty(n_subcarriers)

    for sc in range(n_subcarriers):
        if _HAS_PYCIRCSTAT2:
            mrl[sc] = float(circ_r(phase[:, sc]))
        else:
            # MRL via complex exponential: R = |mean(e^{j*theta})|
            mrl[sc] = np.abs(np.mean(np.exp(1j * phase[:, sc])))

    return mrl


def circular_variance_per_subcarrier(phase: np.ndarray) -> np.ndarray:
    """Circular variance V = 1 - R per subcarrier (0 = concentrated, 1 = uniform)."""
    return 1.0 - mean_resultant_length_per_subcarrier(phase)
