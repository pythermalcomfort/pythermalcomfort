from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pythermalcomfort.utilities import NumericInput

from .p_sat import p_sat


def hr_to_rh(
    hr: NumericInput,
    tdb: NumericInput,
    p_atm: float = 101325,
) -> NDArray[np.float64]:
    """Convert humidity ratio to relative humidity.

    Algebraic inverse of the humidity-ratio formula used in :func:`psy_ta_rh`:

    .. math::

        p_{vap} = \\frac{hr \\cdot p_{atm}}{0.62198 + hr}, \\quad
        rh = \\frac{p_{vap}}{p_{sat}(t_{db})} \\times 100

    Parameters
    ----------
    hr : float or list of floats
        humidity ratio, [kg water / kg dry air]
    tdb : float or list of floats
        air temperature, [°C]
    p_atm : float
        atmospheric pressure, [Pa]

    Returns
    -------
    rh : float or list of floats
        relative humidity, [%]

    """
    hr = np.asarray(hr, dtype=np.float64)
    tdb = np.asarray(tdb, dtype=np.float64)
    p_vap = hr * p_atm / (0.62198 + hr)
    return p_vap / p_sat(tdb) * 100.0
