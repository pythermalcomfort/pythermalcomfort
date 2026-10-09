from __future__ import annotations

import math

import numpy as np
from numba import njit

from pythermalcomfort.classes_input import IREQInputs
from pythermalcomfort.classes_return import IREQ
from pythermalcomfort.utilities import met_to_w_m2


def ireq(
    tdb,
    tr,
    vr,
    rh,
    met,
    clo,
    p,
    walk_sp,
    wme=0,
    limit_inputs=True,
    round_output=True,
) -> IREQ:
    """Calculate Required Clothing Insulation (IREQ) and exposure duration.

    The model estimates the clothing insulation required to maintain thermal
    equilibrium in cold environments and the Duration Limited Exposure (DLE)
    when available clothing insulation is insufficient, in accordance with
    ISO 11079 [11079ISO2007]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    tr : float or list of floats
        Mean radiant temperature, [°C].
    vr : float or list of floats
        Relative air speed, [m/s].
    rh : float or list of floats
        Relative humidity, [%].
    met : float or list of floats
        Metabolic rate, [met].
    clo : float or list of floats
        Clothing insulation, [clo].
    p : float or list of floats
        Air permeability of clothing, [l/(m2 s)].
    walk_sp : float or list of floats
        Walking speed, [m/s].
    wme : float or list of floats, optional
        External work, [met]. Defaults to 0.
    limit_inputs : bool, optional
        When True, results outside the ISO 11079 applicability limits are set
        to nan. Non-physical negative clothing insulation outputs are also set
        to nan. Defaults to True.

        .. note::
            The ISO 11079 applicability limits used by this function are
            58 <= met [W/m2] <= 400 after converting the input metabolic rate
            from met to W/m2, tdb <= 10 [°C], 0.4 <= vr [m/s] <= 18, and
            minimum_walking_speed <= walk_sp [m/s] <= 1.2, where
            minimum_walking_speed = min(0.0052 * (met [W/m2] - 58), 1.2).

    round_output : bool, optional
        When True, rounds numeric output values to one decimal place. Defaults
        to True.

    Returns
    -------
    IREQ
        A dataclass containing ``ireq_min``, ``ireq_neutral``, ``icl_min``,
        ``icl_neutral``, ``dle_min``, and ``dle_neutral``. See
        :py:class:`~pythermalcomfort.classes_return.IREQ` for more details.

    Raises
    ------
    TypeError
        If an input has an unsupported type, or if ``limit_inputs`` or
        ``round_output`` are not booleans.
    ValueError
        If inputs are not broadcastable to a common shape, contain non-finite
        values, or fail physical validation. The physical validation rejects
        ``met <= 0``, ``wme < 0``, ``p <= 0``, ``vr < 0``, ``walk_sp < 0``,
        relative humidity outside 0 to 100 %, and ``clo < 0``.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import ireq

        result = ireq(
            tdb=-15.0,
            tr=-15.0,
            vr=2.0,
            rh=55.0,
            met=175.0 / 58.15,
            clo=2.8,
            p=50.0,
            walk_sp=1.1,
        )
        print(result.ireq_min)  # 1.6
        print(result.dle_min)  # more than 8

    References
    ----------
    ISO 11079:2007 [11079ISO2007]_.
    """

    is_scalar = all(
        np.isscalar(value) for value in [tdb, tr, vr, rh, met, clo, p, walk_sp, wme]
    )

    inputs = IREQInputs(
        tdb=tdb,
        tr=tr,
        vr=vr,
        rh=rh,
        met=met,
        clo=clo,
        p=p,
        walk_sp=walk_sp,
        wme=wme,
        limit_inputs=limit_inputs,
        round_output=round_output,
    )

    tdb = inputs.tdb
    tr = inputs.tr
    vr = inputs.vr
    rh = inputs.rh
    clo = inputs.clo
    p = inputs.p
    walk_sp = inputs.walk_sp
    met = inputs.met * met_to_w_m2
    wme = inputs.wme * met_to_w_m2

    valid_inputs = _valid_iso_11079_inputs(
        met=met,
        tdb=tdb,
        vr=vr,
        walk_sp=walk_sp,
    )

    clo_m2c_w = clo * 0.155
    results = {}

    calculation_criteria = (
        ("min", 33.34 - 0.0354 * met, 0.06),
        ("neutral", 35.7 - 0.0285 * met, 0.001 * met),
    )

    for suffix, skin_temperature, wetness in calculation_criteria:
        ireq_final, icl_raw, dle = _solve_ireq_criterion(
            tdb=tdb,
            tr=tr,
            met=met,
            wme=wme,
            vr=vr,
            walk_sp=walk_sp,
            p=p,
            clo_m2c_w=clo_m2c_w,
            rh=rh,
            skin_temperature=skin_temperature,
            wetness=wetness,
        )

        ireq_out = ireq_final / 0.155
        icl_out = icl_raw / 0.155

        non_physical = (ireq_out < 0) | (icl_out < 0)

        if round_output:
            ireq_out = np.round(ireq_out, 1)
            icl_out = np.round(icl_out, 1)

        dle_out = _format_dle(dle=dle, round_output=round_output)

        if limit_inputs:
            ireq_out[non_physical] = np.nan
            icl_out[non_physical] = np.nan
            dle_out[non_physical] = np.nan
            ireq_out[~valid_inputs] = np.nan
            icl_out[~valid_inputs] = np.nan
            dle_out[~valid_inputs] = np.nan

        results[f"ireq_{suffix}"] = _scalar(ireq_out) if is_scalar else ireq_out
        results[f"icl_{suffix}"] = _scalar(icl_out) if is_scalar else icl_out
        results[f"dle_{suffix}"] = _scalar(dle_out) if is_scalar else dle_out

    return IREQ(
        ireq_min=results["ireq_min"],
        ireq_neutral=results["ireq_neutral"],
        icl_min=results["icl_min"],
        icl_neutral=results["icl_neutral"],
        dle_min=results["dle_min"],
        dle_neutral=results["dle_neutral"],
    )


@njit(cache=True)
def _clothing_constant_part(p, vr, walk_sp):
    """Compute the ISO 11079 Annex A wind/permeability correction factor shared by the
    total and resultant clothing insulation calculations."""
    return (
        0.54 * math.exp(-0.15 * vr - 0.22 * walk_sp) * (p**0.075)
        - 0.06 * math.log(p)
        + 0.5
    )


@njit(cache=True)
def _solve_single_criterion(
    tdb, tr, met, wme, vr, walk_sp, p, clo_m2c_w, rh, skin_temperature, wetness
):
    """Solve the ISO 11079 Annex A thermal balance for one physiological criterion
    (minimal or neutral), returning IREQ, ICL, and DLE in m2.K/W and hours.

    This mirrors the array-vectorized version of the same solve, but works on plain
    floats so it can be JIT-compiled with Numba: each element gets its own
    early-converging loop instead of the whole array running the full iteration
    budget.
    """
    ar_adu = 0.77
    air_insulation = 0.092 * math.exp(-0.15 * vr - 0.22 * walk_sp) - 0.0045
    constant_part = _clothing_constant_part(p=p, vr=vr, walk_sp=walk_sp)

    expired_air_temperature = 29.0 + 0.2 * tdb
    expired_air_vapor_pressure = 0.1333 * math.exp(
        18.6686 - 4030.183 / (expired_air_temperature + 235.0)
    )
    ambient_vapor_pressure = (
        (rh / 100.0) * 0.1333 * math.exp(18.6686 - 4030.183 / (tdb + 235.0))
    )
    skin_saturated_pressure = 0.1333 * math.exp(
        18.6686 - 4030.183 / (skin_temperature + 235.0)
    )
    tr_k = 273.0 + tr

    ireq_clo = 0.5
    factor = 0.5
    balance = 1.0
    clothing_temperature = 0.0
    radiation_heat_loss = 0.0
    convective_heat_loss = 0.0

    for _ in range(150):
        if abs(balance) <= 0.01:
            break

        fcl = 1.0 + 1.197 * ireq_clo
        total_evaporative_resistance = (0.06 / 0.38) * (air_insulation + ireq_clo)
        evaporative_heat_loss = (
            wetness
            * (skin_saturated_pressure - ambient_vapor_pressure)
            / total_evaporative_resistance
        )
        respiratory_heat_loss = 1.73e-02 * met * (
            expired_air_vapor_pressure - ambient_vapor_pressure
        ) + 1.4e-03 * met * (expired_air_temperature - tdb)

        clothing_temperature = skin_temperature - ireq_clo * (
            met - wme - evaporative_heat_loss - respiratory_heat_loss
        )
        clothing_temperature_k = 273.0 + clothing_temperature

        delta_t = clothing_temperature - tr
        if abs(delta_t) < 1e-4:
            radiation_coefficient = (
                5.67e-08
                * 0.95
                * ar_adu
                * 4
                * (273.0 + (clothing_temperature + tr) / 2.0) ** 3
            )
        else:
            radiation_coefficient = (
                5.67e-08 * 0.95 * ar_adu * (clothing_temperature_k**4 - tr_k**4)
            ) / delta_t

        convection_coefficient = 1.0 / air_insulation - radiation_coefficient
        radiation_heat_loss = fcl * radiation_coefficient * (clothing_temperature - tr)
        convective_heat_loss = (
            fcl * convection_coefficient * (clothing_temperature - tdb)
        )

        balance = (
            met
            - wme
            - evaporative_heat_loss
            - respiratory_heat_loss
            - radiation_heat_loss
            - convective_heat_loss
        )

        if balance > 0:
            ireq_clo -= factor
            factor /= 2.0
        else:
            ireq_clo += factor

    ireq_final = (skin_temperature - clothing_temperature) / (
        radiation_heat_loss + convective_heat_loss
    )

    storage = -40.0
    storage_factor = 500.0
    resultant_clothing_insulation = clo_m2c_w
    storage_balance = 1.0

    for _ in range(150):
        if abs(storage_balance) <= 0.01:
            break

        fcl_storage = 1.0 + 1.197 * resultant_clothing_insulation
        resultant_clothing_insulation = (
            clo_m2c_w + 0.085 / fcl_storage
        ) * constant_part - air_insulation / fcl_storage

        total_evaporative_resistance = (0.06 / 0.38) * (
            air_insulation + resultant_clothing_insulation
        )
        evaporative_heat_loss = (
            wetness
            * (skin_saturated_pressure - ambient_vapor_pressure)
            / total_evaporative_resistance
        )
        respiratory_heat_loss = 1.73e-02 * met * (
            expired_air_vapor_pressure - ambient_vapor_pressure
        ) + 1.4e-03 * met * (expired_air_temperature - tdb)

        clothing_temperature_storage = skin_temperature - (
            resultant_clothing_insulation
            * (met - wme - evaporative_heat_loss - respiratory_heat_loss - storage)
        )
        clothing_temperature_k = 273.0 + clothing_temperature_storage

        delta_t = clothing_temperature_storage - tr
        if abs(delta_t) < 1e-4:
            radiation_coefficient_storage = (
                5.67e-08
                * 0.95
                * ar_adu
                * 4
                * (273.0 + (clothing_temperature_storage + tr) / 2.0) ** 3
            )
        else:
            radiation_coefficient_storage = (
                5.67e-08 * 0.95 * ar_adu * (clothing_temperature_k**4 - tr_k**4)
            ) / delta_t

        convection_coefficient_storage = (
            1.0 / air_insulation - radiation_coefficient_storage
        )
        radiation_heat_loss_storage = (
            fcl_storage
            * radiation_coefficient_storage
            * (clothing_temperature_storage - tr)
        )
        convective_heat_loss_storage = (
            fcl_storage
            * convection_coefficient_storage
            * (clothing_temperature_storage - tdb)
        )

        storage_balance = (
            met
            - wme
            - evaporative_heat_loss
            - respiratory_heat_loss
            - radiation_heat_loss_storage
            - convective_heat_loss_storage
            - storage
        )

        if storage_balance > 0:
            storage += storage_factor
            storage_factor /= 2.0
        else:
            storage -= storage_factor

    dle = math.inf if storage == 0.0 else -40.0 / storage

    fcl_final = 1.0 + 1.197 * ireq_final
    icl_raw = (ireq_final + air_insulation / fcl_final) / constant_part - (
        0.085 / fcl_final
    )

    return ireq_final, icl_raw, dle


@np.vectorize(otypes=[float, float, float])
def _solve_ireq_criterion(
    tdb, tr, met, wme, vr, walk_sp, p, clo_m2c_w, rh, skin_temperature, wetness
):
    """Vectorize the Numba-compiled single-criterion solver over array inputs."""
    return _solve_single_criterion(
        tdb=tdb,
        tr=tr,
        met=met,
        wme=wme,
        vr=vr,
        walk_sp=walk_sp,
        p=p,
        clo_m2c_w=clo_m2c_w,
        rh=rh,
        skin_temperature=skin_temperature,
        wetness=wetness,
    )


def _valid_iso_11079_inputs(met, tdb, vr, walk_sp):
    """Return a boolean mask of inputs within the ISO 11079 applicability limits."""
    minimum_walking_speed = np.minimum(0.0052 * (met - 58.0), 1.2)

    return (
        (met >= 58.0)
        & (met <= 400.0)
        & (tdb <= 10.0)
        & (vr >= 0.4)
        & (vr <= 18.0)
        & (walk_sp >= minimum_walking_speed)
        & (walk_sp <= 1.2)
    )


def _format_dle(dle, round_output):
    """Convert raw Duration Limited Exposure hours to the ISO 11079 output format,
    replacing values above the 8 h ceiling with "more than 8"."""
    dle_out = np.empty(dle.shape, dtype=object)
    unlimited = (dle > 8.0) | (dle < 0)
    dle_out[unlimited] = "more than 8"

    if round_output:
        dle_out[~unlimited] = np.round(dle[~unlimited], 1)
    else:
        dle_out[~unlimited] = dle[~unlimited]

    return dle_out


def _scalar(value):
    """Extract the single element of a length-1 array as a Python scalar."""
    item = value[0]
    if isinstance(item, np.generic):
        return item.item()
    return item
