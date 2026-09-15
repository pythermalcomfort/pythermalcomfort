from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pythermalcomfort.utilities import NumericInput


def wet_bulb_tmp(
    tdb: NumericInput,
    rh: NumericInput,
) -> NDArray[np.float64]:
    """Calculate the wet-bulb temperature using the Stull equation [Stull2011]_.

    Parameters
    ----------
    tdb: float or list of floats
        air temperature, [°C]
    rh: float or list of floats
        relative humidity, [%]

    Returns
    -------
    wet_bulb_tmp: float or list of floats
        wet-bulb temperature, [°C]
    """
    tdb = np.asarray(tdb, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)

    return (
        tdb * np.arctan(0.151977 * (rh + 8.313659) ** 0.5)
        + np.arctan(tdb + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )
