from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pythermalcomfort.utilities import NumericInput

from ._correction_clothing import _correction_normal_clothing, _correction_nude


def clo_correction_factor_environment(
    vr: NumericInput,
    v_walk: NumericInput,
    i_cl: NumericInput,
) -> NDArray[np.float64]:
    """Return the correction factor for the total insulation of the clothing ensemble
    (`I`:sub:`T`) or the basic/intrinsic insulation (`I`:sub:`cl`).

    This correction factor takes into account of the fact that the values of
    (`I`:sub:`T`) and (`I`:sub:`cl`) are estimated in static conditions. In real
    environments the person may be walking, activity may pump air through the clothing,
    etc.

    Parameters
    ----------
    vr: float or list of floats
        relative air speed, [m/s]
    v_walk: float or list of floats
        walking speed, [m/s]
    i_cl: float or list of floats
        intrinsic insulation of the clothing ensemble, this is the thermal insulation
        from the skin surface to the outer clothing surface [clo]

    Returns
    -------
    correction_factor: float or list of floats
        correction factor for the total insulation of the clothing ensemble
        (`I`:sub:`T,r` / (`I`:sub:`T`)) or the basic/intrinsic insulation
        (`I`:sub:`cl,r` / (`I`:sub:`cl`))
    """
    vr = np.asarray(vr)
    v_walk = np.asarray(v_walk)
    i_cl = np.asarray(i_cl)

    def correction_low_clothing(
        _vr: NDArray[np.number[Any]],
        _vw: NDArray[np.number[Any]],
        _i_cl: NDArray[np.number[Any]],
    ) -> NDArray[np.floating[Any]]:
        return (
            (0.6 - _i_cl) * _correction_nude(_vr, _vw)
            + _i_cl * _correction_normal_clothing(_vr, _vw)
        ) / 0.6

    c_f = np.where(
        i_cl <= 0.6,
        correction_low_clothing(_vr=vr, _vw=v_walk, _i_cl=i_cl),
        _correction_normal_clothing(_vr=vr, _vw=v_walk),
    )
    return np.where(i_cl == 0, _correction_nude(_vr=vr, _vw=v_walk), c_f)
