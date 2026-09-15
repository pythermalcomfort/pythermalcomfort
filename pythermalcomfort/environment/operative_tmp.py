from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pythermalcomfort.utilities import NumericInput


def operative_tmp(
    tdb: NumericInput,
    tr: NumericInput,
    v: NumericInput,
    standard: str = "ISO",
) -> NDArray[np.float64]:
    """Calculate the operative temperature in accordance with ISO 7726:1998
    [7726ISO1998]_.

    Parameters
    ----------
    tdb: float or list of floats
        air temperature, [°C]
    tr: float or list of floats
        mean radiant temperature, [°C]
    v: float or list of floats
        air speed, [m/s]
    standard: str (default="ISO")
        either choose between ISO and ASHRAE


    Returns
    -------
    to: float
        operative temperature, [°C]
    """
    tdb = np.asarray(tdb, dtype=np.float64)
    tr = np.asarray(tr, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if standard.lower() == "iso":
        return (tdb * np.sqrt(10 * v) + tr) / (1 + np.sqrt(10 * v))
    if standard.lower() == "ashrae":
        a = np.where(v < 0.6, 0.6, 0.7)
        a = np.where(v < 0.2, 0.5, a)
        return a * tdb + (1 - a) * tr
    error_message = (
        f"Operative temperature can only be calculated in compliance with ISO or ASHRAE standards. "
        f"Received standard: {standard}"
    )
    raise ValueError(error_message)
