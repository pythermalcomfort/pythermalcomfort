from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from pythermalcomfort.utilities import NumericInput

from ._correction_clothing import _correction_normal_clothing, _correction_nude


def clo_total_insulation(
    i_t: NumericInput,
    vr: NumericInput,
    v_walk: NumericInput,
    i_a_static: NumericInput,
    i_cl: NumericInput,
) -> NDArray[np.float64]:
    """Calculate the total insulation of the clothing ensemble (`I`:sub:`T,r`).

    The clothing ensemble (`I`:sub:`T,r`) which is
    the actual thermal insulation from the body surface to the environment, considering
    all clothing, enclosed air layers, and boundary air layers under given environmental
    conditions and activities. It accounts for the effects of movements and wind. The
    ISO 9920 standard [ISO9920]_ provides different equations to calculate it as a
    function of the total thermal insulation of clothing (`I`:sub:`T`), the insulation
    of the boundary air layer (`I`:sub:`a`), the walking speed (`v`:sub:`walk`), and the
    relative air speed (`v`:sub:`r`). These different equations are used if the person
    is clothed in normal clothing (0.6 clo < (`I`:sub:`cl`) < 1.4 clo or 1.2 clo <
    (`I`:sub:`T`) < 2.0 clo), nude (`I`:sub:`cl` = 0 clo), and if the person is clothed
    in very light clothing (`I`:sub:`cl` < 0.6 clo). Here we have not implemented the
    equation for high clothing (`I`:sub:`T` > 2.0 clo). Hence the applicability of this
    function is limited to 0 clo < (`I`:sub:`T`) < 2.0 clo). You can find all the inputs
    required in this function in the ISO 9920:2009 standard [ISO9920]_ Annex A.

    Parameters
    ----------
    i_t: float or list of floats
        total thermal insulation of clothing under static reference conditions [clo]
    vr: float or list of floats
        relative air speed, [m/s]
    v_walk: float or list of floats
        walking speed, [m/s]
    i_a_static: float or list of floats
        static boundary air layer insulation, [clo]
    i_cl: float or list of floats
        intrinsic insulation of the clothing ensemble, this is the thermal insulation
        from the skin surface to the outer clothing surface [clo]

    Returns
    -------
    i_t_r: float or list of floats
        total insulation of the clothing ensemble, [clo]
    """
    i_t = np.asarray(i_t)
    vr = np.asarray(vr)
    v_walk = np.asarray(v_walk)
    i_a_static = np.asarray(i_a_static)
    i_cl = np.asarray(i_cl)

    def normal_clothing(
        _vr: NDArray[np.number[Any]],
        _vw: NDArray[np.number[Any]],
        _i_t: NDArray[np.number[Any]],
    ) -> NDArray[np.floating[Any]]:
        return _i_t * _correction_normal_clothing(_vw=_vw, _vr=_vr)

    def nude(
        _vr: NDArray[np.number[Any]],
        _vw: NDArray[np.number[Any]],
        _i_a_static: NDArray[np.number[Any]],
    ) -> NDArray[np.floating[Any]]:
        return _i_a_static * _correction_nude(_vr=_vr, _vw=_vw)

    def low_clothing(
        _vr: NDArray[np.number[Any]],
        _vw: NDArray[np.number[Any]],
        _i_a_static: NDArray[np.number[Any]],
        _i_cl: NDArray[np.number[Any]],
        _i_t: NDArray[np.number[Any]],
    ) -> NDArray[np.floating[Any]]:
        return (
            (0.6 - _i_cl) * nude(_vr, _vw, _i_a_static)
            + _i_cl * normal_clothing(_vr, _vw, _i_t)
        ) / 0.6

    i_t_r = np.where(
        i_cl <= 0.6,
        low_clothing(_vr=vr, _vw=v_walk, _i_a_static=i_a_static, _i_cl=i_cl, _i_t=i_t),
        normal_clothing(_vr=vr, _vw=v_walk, _i_t=i_t),
    )
    return np.where(i_cl == 0, nude(_vr=vr, _vw=v_walk, _i_a_static=i_a_static), i_t_r)
