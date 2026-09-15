from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pythermalcomfort.utilities import NumericInput


def v_relative(
    v: NumericInput,
    met: NumericInput,
) -> NDArray[np.float64]:
    """Estimates the relative air speed which combines the average air speed of the
    space plus the relative air speed caused by the body movement. The same equation is
    used in the ASHRAE 55:2023 and ISO 7730 [7730ISO2005]_ [7730ISO2025]_ standards.

    Parameters
    ----------
    v : float or list of floats
        air speed measured by the sensor, [m/s]
    met : float or list of floats
        metabolic rate, [met]

    Returns
    -------
    vr : float or list of floats
        relative air speed, [m/s]
    """
    v = np.asarray(v, dtype=np.float64)
    met = np.asarray(met, dtype=np.float64)
    return np.where(met > 1, np.around(v + 0.3 * (met - 1), 3), v)
