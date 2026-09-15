from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pythermalcomfort.utilities import NumericInput


def p_sat_torr(tdb: NumericInput) -> NDArray[np.float64]:
    """Estimates the saturation vapor pressure in [torr].

    Parameters
    ----------
    tdb : float or list of floats
        dry bulb air temperature, [C]

    Returns
    -------
    p_sat : float
        saturation vapor pressure [torr]
    """
    tdb = np.asarray(tdb, dtype=np.float64)
    return np.exp(18.6686 - 4030.183 / (tdb + 235.0))
