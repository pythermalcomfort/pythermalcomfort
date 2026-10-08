from __future__ import annotations

import math

import numpy as np
from numba import njit, vectorize

from pythermalcomfort.classes_input import NumericInput, PETSteadyInputs
from pythermalcomfort.classes_return import PETSteady
from pythermalcomfort.psychrometrics.p_sat import p_sat_numba
from pythermalcomfort.utilities import Postures, Sex

_MET_FACTOR = 58.2
_TC_SET = 36.6  # core temperature set value
_TSK_SET = 34.0  # skin temperature set value


@njit(cache=True)
def _vasomotricity(t_cr, t_sk):
    """Vasomotricity (blood flow) in function of the core and skin temperatures.

    Parameters
    ----------
    t_cr : float
        The body core temperature, [°C]
    t_sk : float
        The body skin temperature, [°C]

    Returns
    -------
    tuple of (float, float)
        (m_blood, alpha):
        - m_blood : Blood flow rate, [kg/m2/h]
        - alpha   : Repartition of body mass between core and skin [-]
    """
    # Set value signals
    sig_skin = _TSK_SET - t_sk
    sig_core = t_cr - _TC_SET

    if sig_core < 0:
        # In this case, T_core<Tc_set --> the blood flow is reduced
        sig_core = 0.0
    if sig_skin < 0:
        # In this case, Tsk>Tsk_set --> the blood flow is increased
        sig_skin = 0.0

    # 6.3 L/m^2/h is the set value of the blood flow
    m_blood = (6.3 + 75.0 * sig_core) / (1.0 + 0.5 * sig_skin)

    # 90 L/m^2/h is the blood flow upper limit
    if m_blood > 90:
        m_blood = 90.0

    # in other models, alpha is used to update tbody
    alpha = 0.0417737 + 0.7451833 / (m_blood + 0.585417)

    return m_blood, alpha


@njit(cache=True)
def _sweat_rate(t_body):
    """Sweating mechanism depending on the body and core temperatures.

    Parameters
    ----------
    t_body : float
        weighted average between skin and core temperatures, [°C]

    Returns
    -------
    m_rsw : float
        The sweating flow rate, [g/m2/h].
    """
    tbody_set = 0.1 * _TSK_SET + 0.9 * _TC_SET  # Weighted set temperature

    sig_body = t_body - tbody_set
    if sig_body < 0:
        # In this case, Tbody<Tbody_set --> The sweat flow is 0
        sig_body = 0.0

    # from Gagge's model
    m_rsw = 304.94 * sig_body

    # 500.0 g/m^2/h upper limit
    return min(m_rsw, 500.0)


@njit(cache=True)
def _body_surface_area_numba(weight, height):
    """DuBois body surface area in m2."""
    return 0.202 * (weight**0.425) * (height**0.725)


@njit(cache=True, fastmath=True)
def _solve_3x3(J, F):
    """Direct 3x3 linear system solver J * dx = -F using Cramer's rule / adjugate."""
    det = (
        J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])
        - J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0])
        + J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0])
    )
    if abs(det) < 1e-13:
        return False, 0.0, 0.0, 0.0

    inv_det = 1.0 / det

    inv_00 = (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1]) * inv_det
    inv_01 = (J[0, 2] * J[2, 1] - J[0, 1] * J[2, 2]) * inv_det
    inv_02 = (J[0, 1] * J[1, 2] - J[0, 2] * J[1, 1]) * inv_det

    inv_10 = (J[1, 2] * J[2, 0] - J[1, 0] * J[2, 2]) * inv_det
    inv_11 = (J[0, 0] * J[2, 2] - J[0, 2] * J[2, 0]) * inv_det
    inv_12 = (J[0, 2] * J[1, 0] - J[0, 0] * J[1, 2]) * inv_det

    inv_20 = (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0]) * inv_det
    inv_21 = (J[0, 1] * J[2, 0] - J[0, 0] * J[2, 1]) * inv_det
    inv_22 = (J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]) * inv_det

    dx0 = -(inv_00 * F[0] + inv_01 * F[1] + inv_02 * F[2])
    dx1 = -(inv_10 * F[0] + inv_11 * F[1] + inv_12 * F[2])
    dx2 = -(inv_20 * F[0] + inv_21 * F[1] + inv_22 * F[2])

    return True, dx0, dx1, dx2


@njit(cache=True)
def _solve_pet(
    t_cr,
    t_sk,
    t_clo,
    _tdb,
    _tr,
    weight,
    height,
    age,
    sex_code,  # 0: male, 1: female
    pos_code,  # 0: sitting, 1: standing, 2: standing forced
    p_atm,  # [hPa]
    wme,
    _v=0.1,
    _rh=50,
    _met=80,
    _clo=0.9,
    actual_environment=False,
):
    """Solve PET by either computing the vectorial balance of the three unknown
    temperatures (T_core, T_sk, T_clo) or by finding the environment operative
    temperature that yields the same energy balance as the actual environment.

    Parameters
    ----------
    t_arr : list or list of floats
        [T_core, T_sk, T_clo], [°C]
    _tdb : float
        dry bulb air temperature, [°C]
    _tr : float
        mean radiant temperature, [°C]
    _v : float, default 0.1 m/s for the reference environment
        air speed, [m/s]
    _rh : float, default 50 % for the reference environment
        relative humidity, [%]
    _met : float, default 80 W for the reference environment
        metabolic rate, [W/m2]
    _clo : float, default 0.9 clo for the reference environment
        clothing insulation, [clo]
    actual_environment : boolean
        True=solve 3eqs/3unknowns, False=solve for PET

    Returns
    -------
    When actual_environment is True:
        f0, f1, f2, e_bal_scal) -> 3-node residual vector + scalar sum.
    When actual_environment is False:
        (0.0, 0.0, 0.0, e_bal_scal) -> scalar energy balance for reference room.
    """
    e_skin = 0.99
    e_clo = 0.95
    h_vap = 2.42e6
    sbc = 5.67e-8
    cb = 3640.0

    a_dubois = _body_surface_area_numba(weight, height)

    # Base metabolism [W]
    if sex_code == 0:  # male
        met_correction = (
            3.45
            * (weight**0.75)
            * (
                1.0
                + 0.004 * (30.0 - age)
                + 0.01 * (height * 100.0 / (weight ** (1.0 / 3.0)) - 43.4)
            )
        )
    else:  # female
        met_correction = (
            3.19
            * (weight**0.75)
            * (
                1.0
                + 0.004 * (30.0 - age)
                + 0.018 * (height * 100.0 / (weight ** (1.0 / 3.0)) - 42.1)
            )
        )

    # Source term : metabolic activity [W/m2]
    he = (_met + met_correction) / a_dubois
    h = he * (1.0 - wme)

    i_m = 0.38  # Woodcock ratio
    fcl = 1.0 + 0.31 * _clo
    f_a_cl = (173.51 * _clo - 2.36 - 100.76 * (_clo**2) + 19.28 * (_clo**3)) / 100.0
    a_clo = a_dubois * f_a_cl + a_dubois * (fcl - 1.0)

    f_eff = 0.696 if pos_code == 1 else 0.725
    a_r_eff = a_dubois * f_eff

    # Vapor pressure in air [hPa]
    if actual_environment:
        vpa = (_rh / 100.0) * (p_sat_numba(_tdb) / 100.0)
    else:
        vpa = 12.0  # reference environment (12 hPa)

    # Convective heat transfer coefficient
    if pos_code == 0:  # sitting
        hc = 2.67 + 6.5 * (_v**0.67)
    elif pos_code == 1:  # standing
        hc = 2.26 + 7.42 * (_v**0.67)
    else:  # standing, forced
        hc = 8.6 * (_v**0.513)

    h_cc = 3.0 * ((p_atm / 1013.25) ** 0.53)
    if hc < h_cc:
        hc = h_cc
    hc = hc * ((p_atm / 1013.25) ** 0.55)

    # Respiratory losses
    t_exp = 0.47 * _tdb + 21.0
    d_vent_pulm = he * 1.44e-6
    c_res = 1010.0 * (_tdb - t_exp) * d_vent_pulm
    vpexp = p_sat_numba(t_exp) / 100.0
    q_res = 0.623 * (h_vap / p_atm) * (vpa - vpexp) * d_vent_pulm
    ere = c_res + q_res

    # Vasomotricity & Body temperature
    m_blood, alpha = _vasomotricity(t_cr, t_sk)
    tbody = alpha * t_sk + (1.0 - alpha) * t_cr

    # Clothing thermal resistance & geometry
    r_cl = _clo / 6.45  # [m2.K/W]
    y = 0.0
    f_a_cl = min(f_a_cl, 1.0)
    if _clo >= 2.0:
        y = 1.0
    elif _clo > 0.6:
        y = (height - 0.2) / height
    elif _clo > 0.3:
        y = 0.5
    elif _clo > 0.0:
        y = 0.1

    if _clo > 0.0 and y > 0.0:
        r2 = a_dubois * (fcl - 1.0 + f_a_cl) / (6.28 * height * y)
        r1 = f_a_cl * a_dubois / (6.28 * height * y)
        di = r2 - r1
        if r2 > r1 and r1 > 0.0:
            htcl = 6.28 * height * y * di / (r_cl * math.log(r2 / r1) * a_clo)
        else:
            htcl = 1.0 / r_cl
    else:
        htcl = 1e4  # virtually no clothing resistance

    # Sweat & Evaporative losses
    qmsw = _sweat_rate(tbody)
    esw = (h_vap / 1000.0) * (qmsw / 3600.0)  # [W/m2]
    p_v_sk = p_sat_numba(t_sk) / 100.0  # [hPa]

    lr = 1.67  # Lewis ratio [K/hPa]
    he_diff = hc * lr
    fecl = 1.0 / (1.0 + 0.92 * hc * r_cl)
    e_max = he_diff * fecl * (p_v_sk - vpa)
    if e_max == 0.0:
        e_max = 0.001

    w = esw / e_max
    if w > 1.0:
        w = 1.0
        delta = esw - e_max
        if delta < 0.0:
            esw = e_max
    if esw < 0.0:
        esw = 0.0

    r_ecl = (1.0 / (fcl * hc) + r_cl) / (lr * i_m)
    ediff = (1.0 - w) * (p_v_sk - vpa) / r_ecl
    evap = -(ediff + esw)

    # Radiation losses
    tr_k4 = (_tr + 273.15) ** 4.0
    r_bare = (
        a_r_eff
        * (1.0 - f_a_cl)
        * e_skin
        * sbc
        * (tr_k4 - ((t_sk + 273.15) ** 4.0))
        / a_dubois
    )
    r_clo = f_eff * a_clo * e_clo * sbc * (tr_k4 - ((t_clo + 273.15) ** 4.0)) / a_dubois
    r_sum = r_clo + r_bare

    # Convection losses
    c_bare = hc * (_tdb - t_sk) * (1.0 - f_a_cl)
    c_clo = hc * (_tdb - t_clo) * (a_clo / a_dubois)
    csum = c_clo + c_bare

    # 3-Node Residuals
    core_flow = (m_blood / 3600.0 * cb + 5.28) * (t_cr - t_sk)

    f0 = h + ere - core_flow
    f1 = r_bare + c_bare + evap + core_flow - htcl * (t_sk - t_clo)
    f2 = c_clo + r_clo + htcl * (t_sk - t_clo)
    e_bal_scal = h + ere + r_sum + csum + evap

    if actual_environment:
        return f0, f1, f2, e_bal_scal
    return 0.0, 0.0, 0.0, e_bal_scal


@njit(cache=True)
def _solve_actual_environment(
    tdb: float,
    tr: float,
    v: float,
    rh: float,
    met_w: float,
    clo: float,
    weight: float,
    height: float,
    age: float,
    sex_code: int,
    pos_code: int,
    p_atm: float,
    wme: float,
) -> tuple[float, float, float]:
    """Solves for (T_cr, T_sk, T_clo) reproducing MINPACK's trajectory."""
    t_cr = 36.7
    t_sk = 34.0
    t_clo = 0.5 * (tdb + tr)

    eps = 1e-5
    tol = 1e-6
    max_iter = 120

    for _ in range(max_iter):
        f0, f1, f2, _ = _solve_pet(
            t_cr,
            t_sk,
            t_clo,
            tdb,
            tr,
            weight,
            height,
            age,
            sex_code,
            pos_code,
            p_atm,
            wme,
            _v=v,
            _rh=rh,
            _met=met_w,
            _clo=clo,
            actual_environment=True,
        )

        res_norm = abs(f0) + abs(f1) + abs(f2)
        if res_norm < tol:
            break

        f0_cr, f1_cr, f2_cr, _ = _solve_pet(
            t_cr + eps,
            t_sk,
            t_clo,
            tdb,
            tr,
            weight,
            height,
            age,
            sex_code,
            pos_code,
            p_atm,
            wme,
            _v=v,
            _rh=rh,
            _met=met_w,
            _clo=clo,
            actual_environment=True,
        )
        f0_sk, f1_sk, f2_sk, _ = _solve_pet(
            t_cr,
            t_sk + eps,
            t_clo,
            tdb,
            tr,
            weight,
            height,
            age,
            sex_code,
            pos_code,
            p_atm,
            wme,
            _v=v,
            _rh=rh,
            _met=met_w,
            _clo=clo,
            actual_environment=True,
        )
        f0_cl, f1_cl, f2_cl, _ = _solve_pet(
            t_cr,
            t_sk,
            t_clo + eps,
            tdb,
            tr,
            weight,
            height,
            age,
            sex_code,
            pos_code,
            p_atm,
            wme,
            _v=v,
            _rh=rh,
            _met=met_w,
            _clo=clo,
            actual_environment=True,
        )

        J = np.empty((3, 3), dtype=np.float64)
        J[0, 0] = (f0_cr - f0) / eps
        J[0, 1] = (f0_sk - f0) / eps
        J[0, 2] = (f0_cl - f0) / eps
        J[1, 0] = (f1_cr - f1) / eps
        J[1, 1] = (f1_sk - f1) / eps
        J[1, 2] = (f1_cl - f1) / eps
        J[2, 0] = (f2_cr - f2) / eps
        J[2, 1] = (f2_sk - f2) / eps
        J[2, 2] = (f2_cl - f2) / eps

        F = np.array([f0, f1, f2], dtype=np.float64)
        ok, d_cr, d_sk, d_cl = _solve_3x3(J, F)

        if not ok:
            # Steepest descent fallback if Jacobian becomes singular
            d_cr = -(J[0, 0] * f0 + J[1, 0] * f1 + J[2, 0] * f2) * 1e-3
            d_sk = -(J[0, 1] * f0 + J[1, 1] * f1 + J[2, 1] * f2) * 1e-3
            d_cl = -(J[0, 2] * f0 + J[1, 2] * f1 + J[2, 2] * f2) * 1e-3

        max_d = max((abs(d_cr), abs(d_sk), abs(d_cl)))
        if max_d > 4.0:
            scale = 4.0 / max_d
            d_cr *= scale
            d_sk *= scale
            d_cl *= scale

        # Backtracking line search prevents bouncing across thresholds
        step = 1.0
        accepted = False
        best_cr, best_sk, best_cl = t_cr + d_cr, t_sk + d_sk, t_clo + d_cl

        for _ls in range(8):
            test_cr = t_cr + step * d_cr
            test_sk = t_sk + step * d_sk
            test_cl = t_clo + step * d_cl

            nf0, nf1, nf2, _ = _solve_pet(
                test_cr,
                test_sk,
                test_cl,
                tdb,
                tr,
                weight,
                height,
                age,
                sex_code,
                pos_code,
                p_atm,
                wme,
                _v=v,
                _rh=rh,
                _met=met_w,
                _clo=clo,
                actual_environment=True,
            )

            if abs(nf0) + abs(nf1) + abs(nf2) < res_norm:
                best_cr, best_sk, best_cl = test_cr, test_sk, test_cl
                accepted = True
                break
            step *= 0.5

        if not accepted:
            # Damped step if line search exhausted
            t_cr += 0.1 * d_cr
            t_sk += 0.1 * d_sk
            t_clo += 0.1 * d_cl
        else:
            t_cr, t_sk, t_clo = best_cr, best_sk, best_cl

    return t_cr, t_sk, t_clo


@njit(cache=True)
def _solve_reference_pet(
    t_cr: float,
    t_sk: float,
    t_clo: float,
    weight: float,
    height: float,
    age: float,
    sex_code: int,
    pos_code: int,
    p_atm: float,
    wme: float,
) -> float:
    """Numba replacement for pet_fc finding operative temperature tx where e_bal_scal is zero."""
    tx = t_clo  # starts with clothing temperature (exact match to original pet_guess)
    eps = 1e-4
    tol = 1e-6
    max_iter = 50

    for _ in range(max_iter):
        _, _, _, res = _solve_pet(
            t_cr,
            t_sk,
            t_clo,
            _tdb=tx,
            _tr=tx,
            weight=weight,
            height=height,
            age=age,
            sex_code=sex_code,
            pos_code=pos_code,
            p_atm=p_atm,
            wme=wme,
            _v=0.1,
            _rh=50.0,
            _met=80.0,
            _clo=0.9,
            actual_environment=False,
        )

        if abs(res) < tol:
            break

        _, _, _, res_eps = _solve_pet(
            t_cr,
            t_sk,
            t_clo,
            _tdb=tx + eps,
            _tr=tx + eps,
            weight=weight,
            height=height,
            age=age,
            sex_code=sex_code,
            pos_code=pos_code,
            p_atm=p_atm,
            wme=wme,
            _v=0.1,
            _rh=50.0,
            _met=80.0,
            _clo=0.9,
            actual_environment=False,
        )

        df = (res_eps - res) / eps
        if abs(df) < 1e-12:
            break

        step = res / df
        if abs(step) > 10.0:
            step = 10.0 if step > 0.0 else -10.0

        tx -= step

    return tx


@vectorize(cache=True)
def _pet_steady_kernel(
    tdb,
    tr,
    v,
    rh,
    met,
    clo,
    p_atm,
    position_code,
    age,
    sex_code,
    weight,
    height,
    wme,
):
    """Pure scalar kernel solving a single data point, vectorized by Numba."""
    met_w = met * _MET_FACTOR

    # 1. Solve actual environment for (t_cr, t_sk, t_clo)
    t_cr, t_sk, t_clo = _solve_actual_environment(
        tdb,
        tr,
        v,
        rh,
        met_w,
        clo,
        weight,
        height,
        age,
        sex_code,
        position_code,
        p_atm,
        wme,
    )

    # 2. Find reference operative temperature (PET)
    pet = _solve_reference_pet(
        t_cr, t_sk, t_clo, weight, height, age, sex_code, position_code, p_atm, wme
    )

    return round(pet, 2)


def pet_steady(
    tdb: NumericInput,
    tr: NumericInput,
    v: NumericInput,
    rh: NumericInput,
    met: NumericInput,
    clo: NumericInput,
    p_atm: NumericInput = 1013.25,
    position: str | list[str] = Postures.sitting.value,
    age: NumericInput = 23,
    sex: str | np.ndarray | list = Sex.male.value,
    weight: NumericInput = 75,
    height: NumericInput = 1.8,
    wme: NumericInput = 0,
) -> PETSteady:
    """Calculate the steady physiological equivalent temperature (PET) using the Munich
    Energy-balance Model for Individuals (MEMI) to simulate the human body's thermal
    state in a medically realistic manner.

    PET is defined as the air temperature
    at which, in a typical indoor setting the heat budget of the human body is balanced
    with the same core and skin temperature as under the complex outdoor conditions to be
    assessed [Hoppe1999]_.
    The following assumptions are made for the indoor reference climate: tdb = tr, v = 0.1
    m/s, water vapour pressure = 12 hPa, clo = 0.9 clo, and met = 1.37 met + basic
    metabolism.
    PET allows a layperson to compare the total effects of complex thermal circumstances
    outside with his or her own personal experience indoors in this way. This function
    solves the heat balances without accounting for heat storage in the human body.

    The PET was originally proposed by Hoppe [Hoppe1999]_. In 2018, Walther and Goestchel [Walther2018]_
    proposed a correction of the original model, purging the errors in the
    PET calculation routine, and implementing a state-of-the-art vapour diffusion model.
    Walther and Goestchel (2018) model is therefore used to calculate the PET.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    tr : float or list of floats
        Mean radiant temperature, [°C].
    v : float or list of floats
        Wind speed, [m/s].
    rh : float or list of floats
        Relative humidity, [%].
    met : float or list of floats
        Metabolic rate, [met].
    clo : float or list of floats
        Clothing insulation, [clo].
    p_atm : float or list of floats, optional
        Atmospheric pressure, [hPa]. Defaults to 1013.25.
    position : str or list of str, optional
        Position of the person "sitting", "standing", "standing, forced convection". Defaults to "sitting".
    age : int or list of ints, optional
        Age of the person. Defaults to 23.
    sex : str or list of str, optional
        Sex of the person "male" or "female". Defaults to "male"
    weight : float or list of floats, optional
        Weight of the person, [kg]. Defaults to 75.
    height : float or list of floats, optional
        Height of the person, [m]. Defaults to 1.8.
    wme : float or list of floats, optional
        External work, [met]. Defaults to 0.

    Returns
    -------
    PETSteady
        A dataclass containing the Physiological Equivalent Temperature. See :py:class:`~pythermalcomfort.classes_return.PETSteady` for more details.
        To access the `pet` value, use the `pet` attribute of the returned `PETSteady` instance, e.g., `result.pet`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import pet_steady

        result = pet_steady(tdb=25, tr=25, v=0.1, rh=50, met=1.2, clo=0.5)
        print(result.pet)  # 24.67

        result = pet_steady(
            tdb=[25, 30],
            tr=[25, 30],
            v=[0.1, 0.2],
            rh=[50, 60],
            met=[1.2, 1.4],
            clo=[0.5, 0.6],
        )
        print(result.pet)  # [24.67 31.46]
    """

    PETSteadyInputs(
        tdb=tdb,
        tr=tr,
        v=v,
        rh=rh,
        met=met,
        clo=clo,
        p_atm=p_atm,
        position=position,
        age=age,
        sex=sex,
        weight=weight,
        height=height,
        wme=wme,
    )

    tdb_a, tr_a, v_a, rh_a, met_a, clo_a = map(
        np.atleast_1d, (tdb, tr, v, rh, met, clo)
    )
    p_atm_a, age_a, weight_a, height_a, wme_a = map(
        np.atleast_1d, (p_atm, age, weight, height, wme)
    )
    pos_a = np.atleast_1d(position)
    sex_a = np.atleast_1d(sex)

    # Vectorized mapping of string categoricals to numeric codes
    pos_code = np.where(
        pos_a == Postures.sitting.value,
        0,
        np.where(pos_a == Postures.standing.value, 1, 2),
    )
    sex_code = np.where(sex_a == Sex.male.value, 0, 1)

    # Broadcast all arrays to a common shape
    (
        tdb_b,
        tr_b,
        v_b,
        rh_b,
        met_b,
        clo_b,
        p_atm_b,
        pos_code_b,
        age_b,
        sex_code_b,
        weight_b,
        height_b,
        wme_b,
    ) = np.broadcast_arrays(
        tdb_a,
        tr_a,
        v_a,
        rh_a,
        met_a,
        clo_a,
        p_atm_a,
        pos_code,
        age_a,
        sex_code,
        weight_a,
        height_a,
        wme_a,
    )

    # Execute the vectorized Numba kernel natively at C-speed
    result = _pet_steady_kernel(
        tdb_b,
        tr_b,
        v_b,
        rh_b,
        met_b,
        clo_b,
        p_atm_b,
        pos_code_b,
        age_b,
        sex_code_b,
        weight_b,
        height_b,
        wme_b,
    )

    # Preserve the existing API convention: scalar input -> scalar-like value, array input -> ndarray
    if result.size == 1:
        result = result.item()

    return PETSteady(pet=result)
