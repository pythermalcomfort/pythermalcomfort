from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pythermalcomfort.utilities import NumericInput


def adaptive_cooling_effect(
    v: NumericInput,
    to: NumericInput,
) -> NDArray[np.float64]:
    """Return the adaptive model cooling effect for a given air speed and operative temperature.

    The cooling effect is non-zero only when operative temperature is at or
    above 25 °C **and** air speed meets the minimum threshold (0.6 m/s).

    Parameters
    ----------
    v : float or array-like
        Air speed, [m/s].
    to : float or array-like
        Operative temperature, [°C].

    Returns
    -------
    ce : float or ndarray
        Cooling effect magnitude, [°C].
    """
    v = np.asarray(v, dtype=np.float64)
    to = np.asarray(to, dtype=np.float64)
    magnitude = np.where(
        v >= 1.2, 2.2, np.where(v >= 0.9, 1.8, np.where(v >= 0.6, 1.2, 0.0))
    )
    return np.where(to >= 25.0, magnitude, 0.0)
