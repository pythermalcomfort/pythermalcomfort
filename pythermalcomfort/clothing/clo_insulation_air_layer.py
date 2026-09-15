from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pythermalcomfort.utilities import NumericInput


def clo_insulation_air_layer(
    vr: NumericInput,
    v_walk: NumericInput,
    i_a_static: NumericInput,
) -> NDArray[np.float64]:
    """Calculate the insulation of the boundary air layer (`I`:sub:`a,r`).

    The static
    boundary air value is 0.7 clo (0.109 m2K/W) for air velocities around 0.1 m/s to
    0.15 m/s. Thus, for static conditions, the standard recommends using the value of
    0.7 clo (0.109 m2K/W) for the boundary air layer insulation. For walking conditions,
    the boundary air layer insulation is calculated based on the walking speed (v_walk)
    and the relative air speed (vr). This equation is extracted from the ISO 9920:2009
    standard [ISO9920]_ Section 6.

    Parameters
    ----------
    vr: float or list of floats
        relative air speed, [m/s]
    v_walk: float or list of floats
        walking speed, [m/s]
    i_a_static: float or list of floats
        static boundary air layer insulation, [clo]

    Returns
    -------
    i_a_r: float or list of floats
        boundary air layer insulation, [clo]
    """
    vr = np.asarray(vr)
    v_walk = np.asarray(v_walk)
    i_a_static = np.asarray(i_a_static)

    return (
        np.exp(
            -0.533 * (vr - 0.15)
            + 0.069 * (vr - 0.15) ** 2
            - 0.462 * v_walk
            + 0.201 * v_walk**2,
        )
        * i_a_static
    )
