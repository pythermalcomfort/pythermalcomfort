from __future__ import annotations

import math

import numpy as np
from numba import jit, prange

from pythermalcomfort._internal.validation import _valid_range
from pythermalcomfort.classes_input import NumericInput, PHSInputs
from pythermalcomfort.classes_return import PHS
from pythermalcomfort.psychrometrics import p_sat
from pythermalcomfort.utilities import Models, Postures, met_to_w_m2


def phs(
    tdb: NumericInput,
    tr: NumericInput,
    v: NumericInput,
    rh: NumericInput,
    met: NumericInput,
    clo: NumericInput,
    posture: str | list[str],
    wme: NumericInput = 0,
    round_output: bool = True,
    model: str = Models.iso_7933_2023.value,
    **kwargs,
) -> PHS:
    """Calculate the Predicted Heat Strain (PHS).

    The PHS is calculated in compliance with the ISO 7933:2004 [7933ISO2004]_ or
    2023 Standard [7933ISO2023]_. The ISO 7933 provides a method for the analytical evaluation
    and interpretation of the thermal stress experienced by a subject in a hot environment.
    It describes a method for predicting the sweat rate and the internal core temperature that
    the human body will develop in response to the working conditions.

    The PHS model can be used to predict the: heat by respiratory convection, heat flow
    by respiratory evaporation, steady state mean skin temperature, instantaneous value
    of skin temperature, heat accumulation associated with the metabolic rate, maximum
    evaporative heat flow at the skin surface, predicted sweat rate, predicted evaporative
    heat flow, and rectal temperature.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    tr : float or list of floats
        Mean radiant temperature, [°C].
    v : float or list of floats
        Air speed, [m/s].
    rh : float or list of floats
        Relative humidity, [%].
    met : float or list of floats
        Metabolic rate, [met].
    clo : float or list of floats
        Clothing insulation, [clo].
    posture: string or list of strings
        a string value presenting posture of person "sitting", "standing", or "crouching"
    wme : float or list of floats
        external work, [met] default 0
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.
    model : str, optional
        Select the model you want to use to calculate the PHS. The default option is
        "7933-2023", and the other option is "7933-2004".

    Other Parameters
    ----------------
    limit_inputs : bool, optional
        If True, limits the input parameters to the standard's applicability limits. Defaults to True.

        .. note::
            By default, if the inputs are outside the standard applicability limits the
            function returns nan. If False returns values even if input values are
            outside the applicability limits of the model.

            The 7933 limits are 15 < tdb [°C] < 50, 0 < (tr - tdb) [°C] < 60,
            0 < vr [m/s] < 3, and 0.1 < clo [clo] < 1. The metabolic rate limit is
            standard-specific: 1.7 < met [met] < 7.5 (100-450 W/m2) in the 2004
            standard, and 0.96 < met [met] < 4.3 (56-250 W/m2) in the 2023 standard.

    i_mst : float, optional
        Static moisture permeability index, [dimensionless]. Defaults to 0.38.
    a_p : float, optional
        Fraction of the body surface covered by the reflective clothing, [dimensionless]. Defaults to 0.54.
    drink : int, optional
        1 if workers can drink freely, 0 otherwise. Defaults to 1.
    weight : float, optional
        Body weight, [kg]. Defaults to 75.
    height : float, optional
        Height, [m]. Defaults to 1.8.
    walk_sp : float, optional
        Walking speed, [m/s]. Defaults to 0.
    theta : float, optional
        Angle between walking direction and wind direction, [degrees]. Defaults to 0.
    acclimatized : int, optional
        100 if acclimatized subject, 0 otherwise. Defaults to 100.
    duration : int, optional
        Duration of the work sequence, [minutes]. Defaults to 480.
    f_r : float, optional
        Emissivity of the reflective clothing, [dimensionless]. Defaults to 0.97 in the 2004 standard and
        0.42 in the 2023 standard.
    t_sk : float, optional
        Mean skin temperature when worker starts working, [°C]. Defaults to 34.1.
    t_cr : float, optional
        Mean core temperature when worker starts working, [°C]. Defaults to 36.8.
    t_re : float, optional
        Mean rectal temperature when worker starts working, [°C]. If False in the 2004 standard,
        then t_re = t_cr, whereas in the 2023 standard t_re = 36.8 °C
    t_cr_eq : float, optional
        Mean core temperature as a function of met when worker starts working, [°C]. If False in the 2004
        standard, then t_cr_eq = t_cr, whereas in the 2023 standard t_cr_eq = 36.8 °C.
    t_sk_t_cr_wg : float, optional
        Initial weighting fraction for skin/core temperature coupling, dimensionless.
    sweat_rate_watt : float, optional
        Initial instantaneous regulatory sweat (evaporative) rate at the skin, per unit area,
        [W·m⁻²]. This is an instantaneous rate (W/m²) used at each simulation time step.
    evap_load_wm2_min : float, optional
        Initial accumulated evaporative load per unit area. Input value is expected as an
        instantaneous rate in [W·m⁻²]; during the simulation the value is updated by adding
        instantaneous rates each minute and therefore the accumulated quantity represents the
        sum over minutes (units W·min·m⁻²). It is intended for carry over between consecutive
        simulation segments.

    Returns
    -------
    PHS
        A dataclass containing the Predicted Heat Strain. See :py:class:`~pythermalcomfort.classes_return.PHS` for more details.
        To access the individual attributes, use the corresponding attribute of the returned `PHS` instance, e.g., `result.t_re`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import phs

        result = phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.5,
            clo=0.5,
            posture="standing",
            wme=0,
            duration=480,
        )
        print(result.t_re)  # 37.5

        result = phs(
            tdb=[40, 45],
            tr=[40, 45],
            v=[0.3, 0.4],
            rh=[33.85, 40],
            met=[2.5, 2.6],
            clo=[0.5, 0.6],
            posture=["standing", "standing"],
            wme=[0, 0],
            duration=480,
        )
        print(result.t_re)  # [37.5 42.5]

        # example: pass previous results as initial values to chain simulations
        from pythermalcomfort.models import phs

        # first simulation
        result = phs(
            tdb=40,
            tr=40,
            v=0.3,
            rh=50,
            met=2.5,
            clo=0.5,
            posture="standing",
            wme=0,
            round_output=False,
            duration=60,
        )
        print(result.t_re)  # 37.8
        # second simulation - using previous results as initial values
        # NOTE: when chaining runs, prefer round_output=False to avoid rounding drift.
        result = phs(
            tdb=40,
            tr=40,
            v=0.3,
            rh=50,
            met=2.5,
            clo=0.5,
            posture="standing",
            wme=0,
            round_output=False,
            duration=60,
            t_sk=result.t_sk,
            t_cr=result.t_cr,
            t_re=result.t_re,
            t_cr_eq=result.t_cr_eq,
            t_sk_t_cr_wg=result.t_sk_t_cr_wg,
            sweat_rate_watt=result.sweat_rate_watt,
            evap_load_wm2_min=result.evap_load_wm2_min,
        )
        print(result.t_re)  # 38.5
    """
    if model not in [Models.iso_7933_2004.value, Models.iso_7933_2023.value]:
        error_msg = (
            f"Model '{model}' is not supported. "
            f"Supported models are: {Models.iso_7933_2004.value} and {Models.iso_7933_2023.value}."
        )
        raise ValueError(error_msg)

    PHSInputs(
        tdb=tdb,
        tr=tr,
        v=v,
        rh=rh,
        met=met,
        clo=clo,
        round_output=round_output,
        wme=wme,
        posture=posture,
    )

    default_kwargs = {
        "i_mst": 0.38,
        "a_p": 0.54,
        "drink": 1,
        "weight": 75,
        "height": 1.8,
        "walk_sp": 0,
        "theta": 0,
        "acclimatized": 100,
        "duration": 480,
        "f_r": 0.97,
        "t_sk": 34.1,
        "t_cr": 36.8,
        "t_re": None,
        "t_cr_eq": None,
        "t_sk_t_cr_wg": 0.3,
        "sweat_rate_watt": 0,
        "limit_inputs": True,
        "evap_load_wm2_min": 0,
    }

    if model == Models.iso_7933_2023.value:
        # override changed default kwargs for 2023 standard
        overrides_2023 = {
            "f_r": 0.42,
            "t_re": 36.8,
            "t_cr_eq": 36.8,
        }
        default_kwargs.update(overrides_2023)

    kwargs = {**default_kwargs, **kwargs}

    tdb = np.asarray(tdb)
    tr = np.asarray(tr)
    v = np.asarray(v)
    rh = np.asarray(rh)
    met = np.asarray(met) * met_to_w_m2
    clo = np.asarray(clo)
    wme = np.asarray(wme) * met_to_w_m2
    posture = np.asarray(posture)

    i_mst = kwargs["i_mst"]
    a_p = kwargs["a_p"]
    drink = kwargs["drink"]
    weight = kwargs["weight"]
    height = kwargs["height"]
    walk_sp = kwargs["walk_sp"]
    theta = kwargs["theta"]
    acclimatized = kwargs["acclimatized"]
    duration = kwargs["duration"]
    f_r = kwargs["f_r"]
    t_sk = kwargs["t_sk"]
    t_cr = kwargs["t_cr"]
    t_re = kwargs["t_re"]
    t_cr_eq = kwargs["t_cr_eq"]
    t_sk_t_cr_wg = kwargs["t_sk_t_cr_wg"]
    evap_load_wm2_min = kwargs[
        "evap_load_wm2_min"
    ]  # accumulated evaporative load per area (W·min·m⁻² when accumulated)
    sweat_rate_watt = kwargs[
        "sweat_rate_watt"
    ]  # instantaneous regulatory sweat rate at skin, [W·m⁻²]
    # basic physical validation for carry-over state (supports scalar and array-like)
    t_arr = np.asarray(t_sk_t_cr_wg)
    sweat_arr = np.asarray(sweat_rate_watt)
    evap_arr = np.asarray(evap_load_wm2_min)
    if np.any(sweat_arr < 0):
        raise ValueError("sweat_rate_watt must be >= 0")
    if np.any(evap_arr < 0):
        raise ValueError("evap_load_wm2_min must be >= 0")
    if np.any((t_arr < 0.0) | (t_arr > 1.0)):
        raise ValueError("t_sk_t_cr_wg must be within [0, 1]")
    limit_inputs = kwargs["limit_inputs"]

    if model == Models.iso_7933_2023.value:
        p_a = 0.6105 * np.exp(17.27 * tdb / (tdb + 237.3)) * rh / 100
    else:  # model == Models.iso_7933_2004.value:
        p_a = p_sat(tdb) / 1000 * rh / 100

    acclimatized = int(acclimatized)
    if acclimatized not in [0, 100]:
        raise ValueError("Acclimatized should be 0 or 100")

    if drink not in [0, 1]:
        raise ValueError("Drink should be 0 or 1")

    if weight <= 0 or weight > 1000:
        raise ValueError(
            "The weight of the person should be in kg and it cannot exceed 1000",
        )

    # Use explicit None sentinel for missing t_re and t_cr_eq
    if t_re is None:
        t_re = t_cr
    if t_cr_eq is None:
        t_cr_eq = t_cr

    posture_code = _posture_to_code(posture)
    model_code = _MODEL_2023 if model == Models.iso_7933_2023.value else _MODEL_2004

    (
        tdb_b,
        tr_b,
        v_b,
        p_a_b,
        met_b,
        clo_b,
        posture_code_b,
        t_sk_b,
        t_cr_b,
        t_re_b,
        t_cr_eq_b,
        t_sk_t_cr_wg_b,
        evap_load_wm2_min_b,
        sweat_rate_watt_b,
        wme_b,
    ) = np.broadcast_arrays(
        tdb,
        tr,
        v,
        p_a,
        met,
        clo,
        posture_code,
        t_sk,
        t_cr,
        t_re,
        t_cr_eq,
        t_sk_t_cr_wg,
        evap_load_wm2_min,
        sweat_rate_watt,
        wme,
    )
    output_shape = tdb_b.shape

    (
        t_re,
        t_sk,
        t_cr,
        t_cr_eq,
        t_sk_t_cr_wg,
        sweat_rate_watt,
        evap_load_wm2_min,
        sw_tot_g,
        d_lim_loss_50,
        d_lim_loss_95,
        d_lim_t_re,
    ) = _phs_optimized_array(
        tdb=np.ravel(tdb_b),
        tr=np.ravel(tr_b),
        v=np.ravel(v_b),
        p_a=np.ravel(p_a_b),
        met=np.ravel(met_b),
        clo=np.ravel(clo_b),
        posture_code=np.ravel(posture_code_b),
        drink=drink,
        acclimatized=acclimatized,
        weight=weight,
        wme=np.ravel(wme_b),
        i_mst=i_mst,
        a_p=a_p,
        height=height,
        walk_sp=walk_sp,
        theta=theta,
        duration=duration,
        f_r=f_r,
        t_sk=np.ravel(t_sk_b),
        t_cr=np.ravel(t_cr_b),
        t_re=np.ravel(t_re_b),
        t_cr_eq=np.ravel(t_cr_eq_b),
        t_sk_t_cr_wg=np.ravel(t_sk_t_cr_wg_b),
        evap_load_wm2_min=np.ravel(evap_load_wm2_min_b),
        sweat_rate_watt=np.ravel(sweat_rate_watt_b),
        model_code=model_code,
    )

    t_re = t_re.reshape(output_shape)
    t_sk = t_sk.reshape(output_shape)
    t_cr = t_cr.reshape(output_shape)
    t_cr_eq = t_cr_eq.reshape(output_shape)
    t_sk_t_cr_wg = t_sk_t_cr_wg.reshape(output_shape)
    sweat_rate_watt = sweat_rate_watt.reshape(output_shape)
    evap_load_wm2_min = evap_load_wm2_min.reshape(output_shape)
    sw_tot_g = sw_tot_g.reshape(output_shape)
    d_lim_loss_50 = d_lim_loss_50.reshape(output_shape)
    d_lim_loss_95 = d_lim_loss_95.reshape(output_shape)
    d_lim_t_re = d_lim_t_re.reshape(output_shape)

    output = {
        "t_re": t_re,
        "t_sk": t_sk,
        "t_cr": t_cr,
        "t_cr_eq": t_cr_eq,
        "t_sk_t_cr_wg": t_sk_t_cr_wg,
        "d_lim_loss_50": d_lim_loss_50,
        "d_lim_loss_95": d_lim_loss_95,
        "d_lim_t_re": d_lim_t_re,
        "sweat_rate_watt": sweat_rate_watt,
        "sweat_loss_g": sw_tot_g,
        "evap_load_wm2_min": evap_load_wm2_min,
    }

    if limit_inputs:
        # ISO 7933 Annex A, Table A.1 applicability limits. p_a lower bound is 0.5 in
        # the 2023 revision and 0 in the 2004 edition. The metabolic rate range is also
        # standard-specific: 100-450 W/m2 in the 2004 edition, 56-250 W/m2 in the 2023
        # edition. All other limits are identical, including the range on (tr - tdb),
        # not on tr alone.
        p_a_lower = 0.5 if model == Models.iso_7933_2023.value else 0
        met_range = (56, 250) if model == Models.iso_7933_2023.value else (100, 450)
        tdb_valid = _valid_range(tdb, (15.0, 50.0))
        tr_valid = _valid_range(tr - tdb, (0.0, 60.0), param_name="tr - tdb")
        v_valid = _valid_range(v, (0.0, 3.0))
        p_a_valid = _valid_range(p_a, (p_a_lower, 4.5))
        met_valid = _valid_range(met, met_range)
        clo_valid = _valid_range(clo, (0.1, 1.0))
        all_valid = ~(
            np.isnan(tdb_valid)
            | np.isnan(tr_valid)
            | np.isnan(v_valid)
            | np.isnan(p_a_valid)
            | np.isnan(met_valid)
            | np.isnan(clo_valid)
        )
        for key in output:
            output[key] = np.where(all_valid, output[key], np.nan)

    if round_output:
        for key in output:
            if key != "t_sk_t_cr_wg":
                output[key] = np.around(output[key], 1)
            else:
                output[key] = np.around(output[key], 2)

    return PHS(**output)


# Constants
const_t_eq = math.exp(-1 / 10)
const_t_sk = math.exp(-1 / 3)
const_sw = math.exp(-1 / 10)

_MODEL_2004 = 0
_MODEL_2023 = 1
_POSTURE_STANDING = 0
_POSTURE_SITTING = 1
_POSTURE_CROUCHING = 2


def _posture_to_code(posture: np.ndarray | str) -> np.ndarray | int:
    """Map posture string(s) to integer code(s) for PHS calculations.

    Accepts either a scalar string or an array of strings representing posture.
    Valid values are Postures.standing.value, Postures.sitting.value, and Postures.crouching.value.
    Comparisons are vectorized using numpy for array inputs.

    Parameters
    ----------
    posture : np.ndarray or str
        Scalar or array of posture strings. Valid values are
        Postures.standing.value, Postures.sitting.value, Postures.crouching.value.

    Returns
    -------
    np.ndarray or int
        If input is scalar, returns the corresponding integer code:
        _POSTURE_STANDING, _POSTURE_SITTING, or _POSTURE_CROUCHING.
        If input is array-like, returns a np.ndarray of dtype int64 with mapped codes.

    Raises
    ------
    ValueError
        If any supplied posture is not one of the valid values.

    Notes
    -----
    - Scalar input returns an int code; array input returns a np.ndarray of codes.
    - Comparisons are vectorized for array inputs using numpy.
    """
    posture_arr = np.asarray(posture)
    if posture_arr.ndim == 0:
        posture_value = posture_arr.item()
        if posture_value == Postures.standing.value:
            return _POSTURE_STANDING
        if posture_value == Postures.sitting.value:
            return _POSTURE_SITTING
        if posture_value == Postures.crouching.value:
            return _POSTURE_CROUCHING
        error_msg = "Posture has to be either 'standing', 'sitting', or 'crouching'."
        raise ValueError(error_msg)

    valid = (
        (posture_arr == Postures.standing.value)
        | (posture_arr == Postures.sitting.value)
        | (posture_arr == Postures.crouching.value)
    )
    if not np.all(valid):
        error_msg = "Posture has to be either 'standing', 'sitting', or 'crouching'."
        raise ValueError(error_msg)

    posture_code = np.empty(posture_arr.shape, dtype=np.int64)
    posture_code[posture_arr == Postures.standing.value] = _POSTURE_STANDING
    posture_code[posture_arr == Postures.sitting.value] = _POSTURE_SITTING
    posture_code[posture_arr == Postures.crouching.value] = _POSTURE_CROUCHING
    return posture_code


@jit(nopython=True, cache=True)
def _phs_optimized_scalar(
    tdb,
    tr,
    v,
    p_a,
    met,
    clo,
    posture_code,
    drink,
    acclimatized,
    weight,
    wme,
    i_mst,
    a_p,
    height,
    walk_sp,
    theta,
    duration,
    f_r,
    t_sk,
    t_cr,
    t_re,
    t_cr_eq,
    t_sk_t_cr_wg,
    evap_load_wm2_min,
    sweat_rate_watt,
    model_code,
):
    # DuBois body surface area [m2]
    a_dubois = 0.202 * (weight**0.425) * (height**0.725)
    # specific heat of the body [J/kg/C/min]
    sp_heat = met_to_w_m2 * weight / a_dubois

    d_lim_t_re = 0  # maximum allowable exposure time for heat storage [min]
    # maximum allowable exposure time for sweat rate grams, mean subject [min]
    d_lim_loss_50 = 0
    # maximum allowable exposure time for sweat rate grams, 95 % of the working population [min]
    d_lim_loss_95 = 0
    # set the sweat rate in grams to zero
    sw_tot_g = 0

    if model_code == _MODEL_2023:
        # 2023 standard only has one d_max value
        d_max_50 = (0.03 if drink == 0 else 0.05) * weight * 1000
        d_max_95 = (0.03 if drink == 0 else 0.05) * weight * 1000
    else:  # model == Models.iso_7933_2004.value:
        # maximum sweat rate grams to protect a mean subject [g]
        d_max_50 = 0.075 * weight * 1000
        # maximum sweat rate grams to protect 95 % of the working population [g]
        d_max_95 = 0.05 * weight * 1000

    # def_dir = 1 for unidirectional walking, def_dir = 0 for omni-directional walking
    def_dir = 1 if theta != 0 else 0
    def_speed = 0 if walk_sp == 0 else 1

    # radiating area dubois
    if posture_code == _POSTURE_STANDING:
        a_r_du = 0.77
    elif posture_code == _POSTURE_SITTING:
        a_r_du = 0.7
    else:  # posture == Postures.crouching.value:
        a_r_du = 0.67

    # evaluation of the max sweat rate as a function of the metabolic rate
    if model_code == _MODEL_2004:
        sw_max = (met - 32) * a_dubois
        sw_max = min(sw_max, 400)
        sw_max = max(sw_max, 250)
        if acclimatized == 100:
            sw_max = sw_max * 1.25
    else:  # model == Models.iso_7933_2023.value:
        sw_max = 400 if acclimatized == 0 else 500

    # max skin wettedness
    w_max = 0.85 if acclimatized == 0 else 1

    # static clothing insulation
    i_cl_st = clo * 0.155

    fcl = 1 + 0.28 * clo if model_code == _MODEL_2023 else 1 + 0.3 * clo

    # Static boundary layer thermal insulation in quiet air
    i_a_st = 0.111

    # Total static insulation
    i_tot_st = i_cl_st + i_a_st / fcl
    if def_speed > 0:
        if def_dir == 1:  # Unidirectional walking
            v_r = abs(v - walk_sp * math.cos(3.14159 * theta / 180))
        elif v < walk_sp:
            v_r = walk_sp
        else:
            v_r = v
    else:
        walk_sp = 0.0052 * (met - 58)
        walk_sp = min(walk_sp, 0.7)
        v_r = v

    # Dynamic clothing insulation - correction for wind (Var) and walking speed
    v_ux = v_r
    if v_r > 3:
        v_ux = 3
    w_a_ux = walk_sp
    if walk_sp > 1.5:
        w_a_ux = 1.5
    # correction for the dynamic total dry thermal insulation at or above 0.6 clo
    corr_cl = 1.044 * math.exp(
        (0.066 * v_ux - 0.398) * v_ux + (0.094 * w_a_ux - 0.378) * w_a_ux,
    )
    corr_cl = min(corr_cl, 1)
    # correction for the dynamic total dry thermal insulation at 0 clo
    corr_ia = math.exp((0.047 * v_r - 0.472) * v_r + (0.117 * w_a_ux - 0.342) * w_a_ux)
    corr_ia = min(corr_ia, 1)
    corr_tot = corr_cl
    if clo <= 0.6:
        corr_tot = ((0.6 - clo) * corr_ia + clo * corr_cl) / 0.6
    # total dynamic clothing insulation
    i_tot_dyn = i_tot_st * corr_tot
    # dynamic boundary layer thermal insulation
    i_a_dyn = corr_ia * i_a_st
    i_cl_dyn = i_tot_dyn - i_a_dyn / fcl
    # correction for the dynamic permeability index
    corr_e = (2.6 * corr_tot - 6.5) * corr_tot + 4.9
    im_dyn = i_mst * corr_e
    im_dyn = min(im_dyn, 0.9)
    r_t_dyn = i_tot_dyn / im_dyn / 16.7
    t_exp = 28.56 + 0.115 * tdb + 0.641 * p_a  # expired air temperature
    # respiratory convective heat flow [W/m2]
    c_res = 0.001516 * met * (t_exp - tdb)
    # respiratory evaporative heat flow [W/m2]
    e_res = 0.00127 * met * (59.34 + 0.53 * tdb - 11.63 * p_a)
    z = 3.5 + 5.2 * v_r
    if v_r > 1:
        z = 8.7 * v_r**0.6

    # dynamic convective heat transfer coefficient
    if model_code == _MODEL_2004:
        hc_dyn = 2.38 * abs(t_sk - tdb) ** 0.25
    else:  # model == Models.iso_7933_2023.value:
        t_cl = tr + 0.1  # clothing surface temperature
        hc_dyn = 2.38 * abs(t_cl - tdb) ** 0.25

    hc_dyn = max(hc_dyn, z)

    aux_r = 5.67e-08 * a_r_du

    if model_code == _MODEL_2023:
        f_cl_r = (1 - a_p) * 0.97 + a_p * (1 - f_r)
    else:  # model == Models.iso_7933_2004.value:
        f_cl_r = (1 - a_p) * 0.97 + a_p * f_r

    # Pre-calculate constants
    t_cr_eq_m = 0.0036 * met + 36.6
    t_sk_eq_cl_base = (
        12.165
        + 0.02017 * tdb
        + 0.04361 * tr
        + 0.19354 * p_a
        - 0.25315 * v
        + 0.005346 * met
    )
    t_sk_eq_nu_base = 7.191 + 0.064 * tdb + 0.061 * tr + 0.198 * p_a - 0.348 * v

    for time in range(1, duration + 1):
        t_sk0 = t_sk
        t_re0 = t_re
        t_cr0 = t_cr
        t_cr_eq0 = t_cr_eq
        t_sk_t_cr_wg0 = t_sk_t_cr_wg

        # Core temperature at this minute, by exponential averaging
        t_cr_eq = t_cr_eq0 * const_t_eq + t_cr_eq_m * (1 - const_t_eq)
        # Heat storage associated with this core temperature increase during the last minute
        d_stored_eq = sp_heat * (t_cr_eq - t_cr_eq0) * (1 - t_sk_t_cr_wg0)
        # skin temperature prediction -- clothed model
        t_sk_eq_cl = t_sk_eq_cl_base + 0.51274 * t_re
        # nude model
        t_sk_eq_nu = t_sk_eq_nu_base + 0.616 * t_re
        if clo >= 0.6:
            t_sk_eq = t_sk_eq_cl
        elif clo <= 0.2:
            t_sk_eq = t_sk_eq_nu
        else:
            t_sk_eq = t_sk_eq_nu + 2.5 * (t_sk_eq_cl - t_sk_eq_nu) * (clo - 0.2)

        # skin temperature [C]
        t_sk = t_sk0 * const_t_sk + t_sk_eq * (1 - const_t_sk)
        if time == 1 and model_code == _MODEL_2023:
            # ISO 7933:2023 Annex E forces the skin temperature to its equilibrium
            # value on the first minute, removing the exponential lag for that step.
            # This special case is not present in the 2004 Annex E reference code.
            t_sk = t_sk_eq
        # Saturated water vapour pressure at the surface of the skin
        p_sk = 0.6105 * math.exp(17.27 * t_sk / (t_sk + 237.3))
        t_cl = tr + 0.1  # clothing surface temperature
        while True:
            # radiative heat transfer coefficient
            h_r = f_cl_r * aux_r * ((t_cl + 273) ** 4 - (tr + 273) ** 4) / (t_cl - tr)
            t_cl_new = (fcl * (hc_dyn * tdb + h_r * tr) + t_sk / i_cl_dyn) / (
                fcl * (hc_dyn + h_r) + 1 / i_cl_dyn
            )
            if abs(t_cl - t_cl_new) <= 0.001:
                break
            t_cl = (t_cl + t_cl_new) / 2

        convection = fcl * hc_dyn * (t_cl - tdb)
        radiation = fcl * h_r * (t_cl - tr)
        # maximum evaporative heat flow at the skin surface [W/m2]
        e_max = (p_sk - p_a) / r_t_dyn
        if e_max == 0:  # added this otherwise e_req / e_max cannot be calculated
            e_max = 0.001
        # required evaporative heat flow [W/m2]
        e_req = met - d_stored_eq - wme - c_res - e_res - convection - radiation
        # required skin wettedness
        w_req = e_req / e_max

        if e_req <= 0:
            e_req = 0
            sw_req = 0  # required sweat rate [W/m2]
        elif e_max <= 0:
            e_max = 0
            sw_req = sw_max
        elif w_req >= 1.7:
            sw_req = sw_max
        else:
            e_v_eff = (2 - w_req) ** 2 / 2 if w_req > 1 else 1 - w_req**2 / 2

            e_v_eff = max(0.05, e_v_eff)

            sw_req = e_req / e_v_eff
            sw_req = min(sw_req, sw_max)
        sweat_rate_watt = sweat_rate_watt * const_sw + sw_req * (1 - const_sw)

        if sweat_rate_watt <= 0:
            e_p = 0  # predicted evaporative heat flow [W/m2]
            sweat_rate_watt = 0
        else:
            # Use a small epsilon to avoid division by zero or tiny values
            # This ensures k remains numerically stable
            EPSILON = 1e-6
            k = e_max / max(sweat_rate_watt, EPSILON)
            wp = 1
            if k >= 0.5:
                wp = -k + math.sqrt(k * k + 2)
            wp = min(wp, w_max)
            e_p = wp * e_max

        # body heat storage rate [W/m2]
        d_storage = e_req - e_p + d_stored_eq
        t_cr_new = t_cr0
        while True:
            t_sk_t_cr_wg = 0.3 - 0.09 * (t_cr_new - 36.8)
            t_sk_t_cr_wg = min(t_sk_t_cr_wg, 0.3)
            t_sk_t_cr_wg = max(t_sk_t_cr_wg, 0.1)
            t_cr = (
                d_storage / sp_heat
                + t_sk0 * t_sk_t_cr_wg0 / 2
                - t_sk * t_sk_t_cr_wg / 2
            )
            t_cr = (t_cr + t_cr0 * (1 - t_sk_t_cr_wg0 / 2)) / (1 - t_sk_t_cr_wg / 2)
            if abs(t_cr - t_cr_new) <= 0.001:
                break
            t_cr_new = (t_cr_new + t_cr) / 2

        t_re = t_re0 + (2 * t_cr - 1.962 * t_re0 - 1.31) / 9
        if d_lim_t_re == 0 and t_re >= 38:
            d_lim_t_re = time
        evap_load_wm2_min = evap_load_wm2_min + sweat_rate_watt + e_res
        # sw_tot_g: convert accumulated evaporative load per unit area into total
        # evaporated sweat mass for the whole person (grams, g). This is a per-person
        # cumulative mass over the simulated duration (not per m²).
        sw_tot_g = evap_load_wm2_min * 2.67 * a_dubois / 1.8 / 60
        if d_lim_loss_50 == 0 and sw_tot_g >= d_max_50:
            d_lim_loss_50 = time
        if d_lim_loss_95 == 0 and sw_tot_g >= d_max_95:
            d_lim_loss_95 = time

    # in the standard the if statement is within the while loop, causing it to decay exponentially
    if drink == 0:
        d_lim_loss_95 = d_lim_loss_95 * 0.6
        d_lim_loss_50 = d_lim_loss_95
    if d_lim_loss_50 == 0:
        d_lim_loss_50 = duration
    if d_lim_loss_95 == 0:
        d_lim_loss_95 = duration
    if d_lim_t_re == 0:
        d_lim_t_re = duration

    return (
        t_re,
        t_sk,
        t_cr,
        t_cr_eq,
        t_sk_t_cr_wg,
        sweat_rate_watt,
        evap_load_wm2_min,
        sw_tot_g,
        d_lim_loss_50,
        d_lim_loss_95,
        d_lim_t_re,
    )


@jit(nopython=True, parallel=True, cache=True)
def _phs_optimized_array(
    tdb,
    tr,
    v,
    p_a,
    met,
    clo,
    posture_code,
    drink,
    acclimatized,
    weight,
    wme,
    i_mst,
    a_p,
    height,
    walk_sp,
    theta,
    duration,
    f_r,
    t_sk,
    t_cr,
    t_re,
    t_cr_eq,
    t_sk_t_cr_wg,
    evap_load_wm2_min,
    sweat_rate_watt,
    model_code,
):
    # n == number of flattened input elements
    out_t_re = np.empty_like(tdb, dtype=np.float64)
    out_t_sk = np.empty_like(tdb, dtype=np.float64)
    out_t_cr = np.empty_like(tdb, dtype=np.float64)
    out_t_cr_eq = np.empty_like(tdb, dtype=np.float64)
    out_t_sk_t_cr_wg = np.empty_like(tdb, dtype=np.float64)
    out_sweat_rate_watt = np.empty_like(tdb, dtype=np.float64)
    out_evap_load_wm2_min = np.empty_like(tdb, dtype=np.float64)
    out_sw_tot_g = np.empty_like(tdb, dtype=np.float64)
    out_d_lim_loss_50 = np.empty_like(tdb, dtype=np.float64)
    out_d_lim_loss_95 = np.empty_like(tdb, dtype=np.float64)
    out_d_lim_t_re = np.empty_like(tdb, dtype=np.float64)

    n = tdb.size

    for i in prange(n):
        (
            out_t_re[i],
            out_t_sk[i],
            out_t_cr[i],
            out_t_cr_eq[i],
            out_t_sk_t_cr_wg[i],
            out_sweat_rate_watt[i],
            out_evap_load_wm2_min[i],
            out_sw_tot_g[i],
            out_d_lim_loss_50[i],
            out_d_lim_loss_95[i],
            out_d_lim_t_re[i],
        ) = _phs_optimized_scalar(
            tdb[i],
            tr[i],
            v[i],
            p_a[i],
            met[i],
            clo[i],
            posture_code[i],
            drink,
            acclimatized,
            weight,
            wme[i],
            i_mst,
            a_p,
            height,
            walk_sp,
            theta,
            duration,
            f_r,
            t_sk[i],
            t_cr[i],
            t_re[i],
            t_cr_eq[i],
            t_sk_t_cr_wg[i],
            evap_load_wm2_min[i],
            sweat_rate_watt[i],
            model_code,
        )

    return (
        out_t_re,
        out_t_sk,
        out_t_cr,
        out_t_cr_eq,
        out_t_sk_t_cr_wg,
        out_sweat_rate_watt,
        out_evap_load_wm2_min,
        out_sw_tot_g,
        out_d_lim_loss_50,
        out_d_lim_loss_95,
        out_d_lim_t_re,
    )
