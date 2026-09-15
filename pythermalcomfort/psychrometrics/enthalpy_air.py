from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pythermalcomfort.utilities import NumericInput, cp_air, cp_vapour, h_fg


def enthalpy_air(
    tdb: NumericInput,
    hr: NumericInput,
) -> NDArray[np.float64]:
    """Calculate air enthalpy_air.

    Parameters
    ----------
    tdb: float or list of floats
        air temperature, [°C]
    hr: float or list of floats
        humidity ratio, [kg water/kg dry air]

    Returns
    -------
    enthalpy_air: float or list of floats
        enthalpy_air [J/kg dry air]
    """
    tdb = np.asarray(tdb, dtype=np.float64)
    hr = np.asarray(hr, dtype=np.float64)
    h_dry_air = cp_air * tdb
    h_sat_vap = h_fg + cp_vapour * tdb
    return h_dry_air + hr * h_sat_vap
