from __future__ import annotations

import numpy as np

from pythermalcomfort._internal.bfu import _fan_convection, _OccupationalHeatBalance
from pythermalcomfort.classes_input import BFUOccupationalInputs, NumericInput
from pythermalcomfort.classes_return import BFUOccupational
from pythermalcomfort.psychrometrics import p_sat
from pythermalcomfort.utilities import Postures, met_to_w_m2


def bfu_occupational(
    tdb: NumericInput,
    rh: NumericInput,
    met: NumericInput,
    clo: NumericInput,
    tr: NumericInput | None = None,
    v: NumericInput = 3.5,
    v_no_fan: NumericInput = 0.2,
    posture: str | Postures | list[str] | np.ndarray = Postures.standing.value,
    activity: str | list[str] | np.ndarray = "walk",
    speed: NumericInput = 1.66,
    max_sweating: NumericInput | None = 1000 / 1.81,
    body_surface_area: NumericInput = 1.81,
    round_output: bool = True,
) -> BFUOccupational:
    """Calculate the change in occupational heat storage from fan use.

    Implement the biophysical model described by Foster et al. (2021)
    [Foster2021]_. Compare heat storage with and without a fan using the
    air speed at the worker's body in each condition. The model accounts for
    clothing, metabolic heat production, posture, movement, and sweat production.
    ``activity`` selects the movement-induced airflow coefficients and is
    distinct from ``met``, which determines metabolic heat production. The same
    metabolic rate can occur during walking, cycling, or stationary work.
    Clothing corrections also use ``speed``; it is not inferred from ``met``.
    The source simulations used walking at 1.66 m/s, 0.28 clo, and fan-on and
    fan-off air speeds of 3.5 and 0.2 m/s. The published experimental conditions
    covered 35-50 °C and 20-80 % RH [Foster2021]_. Other activities are extensions
    of the heat-transfer equations, not separately validated model predictions.
    The function evaluates the equations for the supplied inputs and does not
    screen them against the paper's experimental applicability domain. Results
    outside that domain are extrapolations, not individual safe-use limits.
    Negative ``delta_storage`` means the fan reduces modelled heat storage.
    Use ``bfu_rest`` for the separate resting model and its population groups.

    Parameters
    ----------
    tdb : float or list of floats
        Dry-bulb air temperature, :math:`°C`.
    rh : float or list of floats
        Relative humidity, :math:`%`.
    met : float or list of floats
        Metabolic rate, [met], where 1 met = 58.15 W/m². This input is required.
    clo : float or list of floats
        Clothing insulation, [clo]. This input is required.
    tr : float or list of floats, optional
        Mean radiant temperature, [°C]. If omitted it is assumed equal to ``tdb``.
    v : float or list of floats, optional
        Air speed at the worker's body with the fan, [m/s]. Defaults to 3.5.
    v_no_fan : float or list of floats, optional
        Air speed at the worker's body without the fan, [m/s]. Defaults to 0.2.
        Both air speeds are needed to compare the two heat balances.
    posture : str, optional
        Worker posture: ``"standing"`` or ``"sitting"``, or the corresponding
        :py:class:`~pythermalcomfort.utilities.Postures` member.
        ``"seated"`` is accepted as an alias. Defaults to ``"standing"``.
    activity : str, optional
        Activity classification: ``"walk"``, ``"rest"``, or ``"cycle"``,
        case-insensitively. Walking uses the walking airflow coefficients;
        rest removes airflow from movement; cycling interprets ``speed`` as rpm.
        Unknown values raise ``ValueError``. Defaults to ``"walk"``.
    speed : float or list of floats, optional
        Walking speed, [m/s], or cycling cadence, [rpm], according to activity.
        Defaults to 1.66. Cycling cadence contributes to convective airflow;
        walking speed is zero for cycling in the clothing correction equations.
    max_sweating : float or list of floats, optional
        Maximum sweat production per body area, [g/(h m²)], before evaporative
        efficiency is applied. The default, 1000 / 1.81, corresponds to 1 L/h
        for a 1.81 m² person. If ``None``, use Foster's humidity-dependent
        whole-body sweat production and divide by ``body_surface_area``.
    body_surface_area : float or list of floats, optional
        Body surface area, [m²]. Defaults to 1.81.
    round_output : bool, optional
        If ``True`` numeric outputs are rounded (storage components to one decimal, efficiencies to two decimals). Defaults to ``True``.

    Returns
    -------
    BFUOccupational
        A dataclass containing fan-on / fan-off storage rates, their components, and
        a ``fan_benefit`` code where ``1`` indicates beneficial fan use, ``0`` neutral,
        and ``-1`` harmful. The thresholds use unrounded ``delta_storage``:
        below -50 W/m² is beneficial, above 50 W/m² is harmful, and the
        inclusive interval [-50, 50] W/m² is neutral. See
        :py:class:`~pythermalcomfort.classes_return.BFUOccupational`.

    Raises
    ------
    ValueError
        If humidity is outside 0-100 %, posture or activity is unknown, speed or sweat rate
        is negative, or input arrays cannot be broadcast together.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import bfu_occupational

        result = bfu_occupational(tdb=30, rh=60, clo=0.28, met=125 / 58.15)
        print(result.delta_storage)

    References
    ----------
    See [Foster2021]_ for the model and its physiological assumptions.
    """
    inputs = BFUOccupationalInputs(
        tdb=tdb,
        rh=rh,
        tr=tr,
        clo=clo,
        met=met,
        v=v,
        v_no_fan=v_no_fan,
        posture=posture,
        activity=activity,
        speed=speed,
        max_sweating=max_sweating,
        body_surface_area=body_surface_area,
        round_output=round_output,
    )

    tdb_arr = np.asarray(inputs.tdb, dtype=float)
    tr_arr = np.asarray(inputs.tr, dtype=float)
    rh_arr = np.asarray(inputs.rh, dtype=float)
    clo_arr = np.asarray(inputs.clo, dtype=float)
    met_arr = np.asarray(inputs.met, dtype=float)
    bsa_arr = np.asarray(inputs.body_surface_area, dtype=float)
    speed_arr = np.asarray(inputs.speed, dtype=float)
    fan_on_arr = np.asarray(inputs.v, dtype=float)
    fan_off_arr = np.asarray(inputs.v_no_fan, dtype=float)

    # Include every input in broadcasting, including categorical and sweat arrays.
    values = [
        tdb_arr,
        tr_arr,
        rh_arr,
        clo_arr,
        met_arr,
        bsa_arr,
        speed_arr,
        fan_on_arr,
        fan_off_arr,
        np.asarray(inputs.posture),
        np.asarray(inputs.activity),
    ]
    if inputs.max_sweating is not None:
        values.append(np.asarray(inputs.max_sweating, dtype=float))
    broadcast = np.broadcast_arrays(*values)
    (
        tdb_arr,
        tr_arr,
        rh_arr,
        clo_arr,
        met_arr,
        bsa_arr,
        speed_arr,
        fan_on_arr,
        fan_off_arr,
        posture_arr,
        activity_arr,
    ) = broadcast[:11]
    max_sweating_arr = broadcast[11] if len(broadcast) == 12 else None
    met_arr = met_arr * met_to_w_m2  # Heat balance uses W/m² internally.

    is_seated = posture_arr == Postures.sitting.value
    is_walk = activity_arr == "walk"
    is_rest = activity_arr == "rest"
    effective_speed = np.where(is_rest, 0.0, speed_arr)
    # Clothing corrections use walking speed [m/s], never cycling cadence [rpm].
    walk_sp = np.where(is_walk, effective_speed, 0.0)

    # Base physiological and physical properties
    # Posture and activity contributions to effective air speed, [m/s].
    v_posture = np.where(is_seated, 0.07, 0.11)
    v_activity = np.where(
        is_walk,
        0.67 * effective_speed,
        0.0043 * effective_speed,
    )
    emissivity = 0.95
    sigma = 5.67e-8
    # Effective radiating area divided by total body surface area, [-].
    ar_ad = np.where(is_seated, 0.70, 0.77)

    t_sk = 25.883 + (0.23 * tdb_arr) + (0.024 * rh_arr) - (0.304 * clo_arr)
    hr_kelvin = 273.2 + (t_sk + tr_arr) / 2.0
    h_r = 4.0 * emissivity * sigma * ar_ad * (hr_kelvin**3)

    hc_off = _fan_convection(v=fan_off_arr, v_posture=v_posture, v_activity=v_activity)
    t_o = (h_r * tr_arr + hc_off * tdb_arr) / (h_r + hc_off)

    i_cl = 0.155 * clo_arr
    f_cl = 1.0 + 1.81 * i_cl
    i_a_off = 1.0 / np.maximum(hc_off + h_r, 1e-9)
    i_t_off = i_cl + i_a_off / f_cl

    psat_ta = p_sat(tdb=tdb_arr) / 1000  # Shared helper returns Pa; model uses kPa.
    pa = psat_ta * (rh_arr / 100.0)
    psat_tsk = p_sat(tdb=t_sk) / 1000

    if max_sweating_arr is None:
        # Empirical whole-body sweat production [g/h] as a function of humidity.
        sweat_gph = (459.5 - 1000) * np.exp(-0.519 * pa) + 1000
        max_sweating_arr = np.maximum(sweat_gph, 0) / bsa_arr
    sweat_rate_watt = max_sweating_arr * 2430 / 3600  # Latent heat, [W/m²].

    r_ecl_off = i_cl * 0.18
    r_ea_off = 1.0 / np.maximum(16.5 * hc_off, 1e-9)
    r_et_off = r_ecl_off + r_ea_off / f_cl

    resp = (0.0014 * met_arr * (34.0 - tdb_arr)) + (0.0173 * met_arr * (5.87 - pa))

    balance = _OccupationalHeatBalance(
        v_posture=v_posture,
        v_activity=v_activity,
        walk_sp=walk_sp,
        t_sk=t_sk,
        t_o=t_o,
        i_cl=i_cl,
        f_cl=f_cl,
        i_a_off=i_a_off,
        i_t_off=i_t_off,
        r_et_off=r_et_off,
        psat_tsk=psat_tsk,
        pa=pa,
        sweat_rate_watt=sweat_rate_watt,
        met_arr=met_arr,
        resp=resp,
    )
    heat_storage_fan, dry_fan, evap_fan, eff_fan, _ = balance.fan_state(
        vwind=fan_on_arr
    )
    heat_storage_no_fan, dry_off, evap_off, eff_off, _ = balance.fan_state(
        vwind=fan_off_arr
    )

    delta_storage_raw = heat_storage_fan - heat_storage_no_fan
    fan_benefit = np.select(
        [delta_storage_raw < -50, delta_storage_raw <= 50],
        [1, 0],
        default=-1,
    )
    if isinstance(fan_benefit, np.ndarray) and fan_benefit.shape == ():
        fan_benefit = fan_benefit.item()

    if round_output:
        heat_storage_fan = np.around(heat_storage_fan, 1)
        heat_storage_no_fan = np.around(heat_storage_no_fan, 1)
        delta_storage = np.around(delta_storage_raw, 1)
        dry_fan = np.around(dry_fan, 1)
        dry_off = np.around(dry_off, 1)
        evap_fan = np.around(evap_fan, 1)
        evap_off = np.around(evap_off, 1)
        resp = np.around(resp, 1)
        eff_fan = np.around(eff_fan, 2)
        eff_off = np.around(eff_off, 2)
    else:
        delta_storage = delta_storage_raw

    return BFUOccupational(
        heat_storage_fan=heat_storage_fan,
        heat_storage_no_fan=heat_storage_no_fan,
        delta_storage=delta_storage,
        dry_heat_fan=dry_fan,
        dry_heat_no_fan=dry_off,
        evaporative_heat_fan=evap_fan,
        evaporative_heat_no_fan=evap_off,
        respiratory_heat=resp,
        sweat_efficiency_fan=eff_fan,
        sweat_efficiency_no_fan=eff_off,
        fan_benefit=fan_benefit,
    )
