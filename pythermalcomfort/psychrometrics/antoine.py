from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from pythermalcomfort.utilities import NumericInput


def antoine(tdb: NumericInput) -> NDArray[np.float64]:
    """Calculate saturated vapor pressure using Antoine equation [kPa].

    Parameters
    ----------
    tdb : float or list of floats
        Temperature [°C].

    Returns
    -------
    float or list of floats
        Saturated vapor pressure [kPa].
    """
    tdb = np.asarray(tdb, dtype=np.float64)
    return math.e ** (16.6536 - 4030.183 / (tdb + 235))
