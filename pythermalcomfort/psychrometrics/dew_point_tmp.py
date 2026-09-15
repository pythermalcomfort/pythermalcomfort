from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pythermalcomfort.utilities import NumericInput


def dew_point_tmp(
    tdb: NumericInput,
    rh: NumericInput,
) -> NDArray[np.float64]:
    """Calculate the dew point temperature.

    The equation uses the Magnus formula with coefficients from
    the 2024 edition of the Guide to Instruments and Methods of
    Observation. [WMO2024]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C]
    rh : float or list of floats
        Relative humidity, [%]

    Returns
    -------
    dew_point_tmp : ndarray
        Dew point temperature, [°C]

    Raises
    ------
    ValueError
        If relative humidity is outside the range [0, 100]%.

    Examples
    --------
    >>> from pythermalcomfort.psychrometrics import dew_point_tmp
    >>> tdb = 25.0  # dry bulb temperature in °C
    >>> rh = 60.0  # relative humidity in %
    >>> t_d = dew_point_tmp(tdb, rh)
    """
    tdb = np.asarray(tdb, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)

    if np.any(rh < 0) or np.any(rh > 100):
        raise ValueError("Relative humidity must be between 0 and 100%.")

    e_w = 6.112 * np.exp(
        (17.62 * tdb) / (243.12 + tdb)
    )  # saturation vapor pressure in hPa
    e_s = (rh / 100) * e_w  # actual vapor pressure in hPa
    return 243.12 * np.log(e_s / 6.112) / (17.62 - np.log(e_s / 6.112))
