from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pythermalcomfort.utilities import NumericInput, c_to_k


def p_sat(tdb: NumericInput) -> NDArray[np.float64]:
    """Calculate vapour pressure of water at different temperatures.

    Parameters
    ----------
    tdb: float or list of floats
        air temperature, [°C]

    Returns
    -------
    p_sat: float or list of floats
        saturation vapor pressure, [Pa]
    """
    # pre-calculated constants for p_sat
    c1 = -5674.5359
    c2 = 6.3925247
    c3 = -0.9677843 * 1e-2
    c4 = 0.62215701 * 1e-6
    c5 = 0.20747825 * 1e-8
    c6 = -0.9484024 * 1e-12
    c7 = 4.1635019
    c8 = -5800.2206
    c9 = 1.3914993
    c10 = -0.048640239
    c11 = 0.41764768 * 1e-4
    c12 = -0.14452093 * 1e-7
    c13 = 6.5459673

    tdb = np.asarray(tdb, dtype=np.float64)
    ta_k = tdb + c_to_k
    # pre-calculate the value before passing it to .where
    log_ta_k = np.log(ta_k)
    return np.where(
        ta_k < c_to_k,
        np.exp(
            c1 / ta_k
            + c2
            + ta_k * (c3 + ta_k * (c4 + ta_k * (c5 + c6 * ta_k)))
            + c7 * log_ta_k,
        ),
        np.exp(
            c8 / ta_k + c9 + ta_k * (c10 + ta_k * (c11 + ta_k * c12)) + c13 * log_ta_k,
        ),
    )
