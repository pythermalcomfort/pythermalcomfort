from __future__ import annotations

import numpy as np

from pythermalcomfort.environment import v_relative
from pythermalcomfort.utilities import Models, NumericInput, met_to_w_m2

from .clo_area_factor import clo_area_factor
from .clo_insulation_air_layer import clo_insulation_air_layer
from .clo_total_insulation import clo_total_insulation


def clo_dynamic_iso(
    clo: NumericInput,
    met: NumericInput,
    v: NumericInput,
    i_a: NumericInput = 0.7,
    model: str = Models.iso_9920_2007.value,
) -> np.ndarray:
    """Estimates the dynamic intrinsic clothing insulation (I :sub:`cl,r`).

    The activity as well as the air speed modify the insulation characteristics of the
    clothing. Consequently, ISO 7730 states that (I :sub:`cl,`) shall be corrected
    [7730ISO2005]_ [7730ISO2025]_. Both editions of ISO 7730 give the correction
    equations in their (informative) Annex C, adapted from the ISO 9920:2007 standard
    [ISO9920]_, which is what we have implemented here.

    .. note::
        The walking speed is not a function input; it is estimated from the metabolic
        rate using the formula given in ISO 7730 Annex C / ISO 9920 for when the actual
        walking speed is undefined: v_walk = 0.0052 * (M - 58), clipped to 0-0.7 m/s,
        where M is the metabolic rate in W/m2. This is a different formula from
        :py:meth:`pythermalcomfort.environment.v_relative`'s activity-generated air speed
        (0.3 * (met - 1)), which is used in the whole-body PMV heat balance rather than
        for the clothing dynamic insulation correction.

    Parameters
    ----------
    clo : float or list of floats
        clothing insulation, [clo]
    met : float or list of floats
        metabolic rate, [met]
    v : float or list of floats
        air speed, [m/s]
    i_a : float or list of floats
        thermal insulation of the boundary (surface) air layer around the outer clothing
        or, when nude, around the skin surface, [clo]
    model : str, optional
        Select the version of the ISO standard to use. Currently, the only
        option available is "9920-2007".

    Returns
    -------
    clo : float or list of floats
        dynamic clothing insulation, [clo]
    """
    model = model.lower()
    if model not in [Models.iso_9920_2007.value]:
        invalid_model_msg = (
            f"PMV calculations can only be performed in "
            f"compliance with ISO {Models.iso_9920_2007.value}"
        )
        raise ValueError(invalid_model_msg)

    clo = np.asarray(clo)
    met = np.asarray(met)
    i_a = np.asarray(i_a)
    v = np.asarray(v)

    f_cl = clo_area_factor(i_cl=clo)
    i_t = clo + i_a / f_cl
    # walking speed when undefined, per ISO 7730 Annex C / ISO 9920: vw = 0.0052 * (M - 58),
    # clipped to [0, 0.7] m/s; M is the metabolic rate in W/m2
    v_walk = np.clip(0.0052 * (met * met_to_w_m2 - 58), 0, 0.7)
    v_r = v_relative(v=v, met=met)
    i_t_r = clo_total_insulation(
        i_t=i_t,
        vr=v_r,
        v_walk=v_walk,
        i_a_static=i_a,
        i_cl=clo,
    )
    i_a_r = clo_insulation_air_layer(vr=v_r, v_walk=v_walk, i_a_static=i_a)
    return i_t_r - i_a_r / f_cl
