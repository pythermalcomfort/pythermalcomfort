from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import numpy as np
from numba import float64, njit, vectorize
from numpy.typing import NDArray

from pythermalcomfort.classes_input import HIInputs, NumericInput
from pythermalcomfort.classes_return import HI


def heat_index_lu(
    tdb: NumericInput,
    rh: NumericInput,
    round_output: bool = True,
) -> HI:
    """Calculate the Heat Index (HI) in accordance with the Lu and Romps (2022) model
    [lu]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    rh : float or list of floats
        Relative humidity, [%].
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    HI
        A dataclass containing the Heat Index. See :py:class:`~pythermalcomfort.classes_return.HI` for more details.
        To access the `hi` value, use the `hi` attribute of the returned `HI` instance, e.g., `result.hi`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import heat_index_lu

        result = heat_index_lu(tdb=25, rh=50)
        print(result.hi)  # 25.0
    """
    # Validate inputs using the HeatIndexInputs class
    HIInputs(
        tdb=tdb,
        rh=rh,
        round_output=round_output,
        limit_inputs=False,
    )

    tdb = np.asarray(tdb)
    rh = np.asarray(rh)

    hi = _lu_heat_index_optimized(tdb + 273.15, rh / 100) - 273.15

    if round_output:
        hi = np.around(hi, 1)

    return HI(hi=hi)


# Thermodynamic parameters
_T_C_K = 273.16  # K
_P_TRIPLE_POINT = 611.65  # Pa
_E0V = 2.3740e6  # J/kg
_E0S = 0.3337e6  # J/kg
_RGASA = 287.04  # J/kg/K
_RGASV = 461.0  # J/kg/K
_CVA = 719.0  # J/kg/K
_CVV = 1418.0  # J/kg/K
_CVL = 4119.0  # J/kg/K
_CVS = 1861.0  # J/kg/K
_CPA = _CVA + _RGASA
_CPV = _CVV + _RGASV

# Thermo-regulatory parameters
_SIGMA = 5.67e-8  # W/m^2/K^4 , Stefan-Boltzmann constant
_EPSILON = 0.97  # emissivity of surface, steadman1979
_MASS = 83.6  # kg, mass of average US adults, fryar2018
_HEIGHT = 1.69  # m, height of average US adults, fryar2018
_AREA = 0.202 * (_MASS**0.425) * (_HEIGHT**0.725)  # m^2, DuBois formula, parson2014
_CPC = 3492.0  # J/kg/K, specific heat capacity of core, gagge1972
_HC_CORE = _MASS * _CPC / _AREA  # heat capacity of core
_R = 124.0  # Pa/K, Zf/rf, steadman1979
_Q = 180.0  # W/m^2, metabolic rate per skin area, steadman1979
_PHI_SALT = 0.9  # vapor saturation pressure level of saline solution, steadman1979
_T_CR = 310.0  # K, core temperature, steadman1979
_P = 1.013e5  # Pa, atmospheric pressure
_ETA = 1.43e-6  # kg/J, "inhaled mass" / "metabolic rate", steadman1979
_PA0 = 1.6e3  # Pa, reference air vapor pressure in regions III, IV, V, VI, steadman1979

_ZA = 60.6 / 17.4  # Pa m^2/W, mass transfer resistance through air, exposed skin
_ZA_BAR = 60.6 / 11.6  # Pa m^2/W, mass transfer resistance through air, clothed skin
_ZA_UN = 60.6 / 12.3  # Pa m^2/W, mass transfer resistance through air, naked

# tolerance and maximum iteration for the root solver
_TOL = 1e-8
_TOL_T = 1e-8
_MAX_ITER = 100

# equivalent-variable codes, replacing the original string keys so this can be
# jitted in nopython mode ("rs" and "rs*" collapse to the same code because
# find_t/_find_t treats them identically; only the (unused) region label differed)
_EQ_PHI = 1
_EQ_RF = 2
_EQ_RS = 3
_EQ_DTCDT = 4

# residual-function codes dispatched by _residual/_bisect
_R_TS = 1
_R_TF = 2
_R_TF_REGION_II_III = 3
_R_TS_REGION_IV_V = 4
_R_TS_REGION_V = 5
_R_T_PHI = 6
_R_T_RF = 7
_R_T_RS = 8
_R_T_DTCDT = 9


@njit(cache=True)
def _pv_star(t):  # The saturation vapor pressure
    if t == 0.0:
        return 0.0
    if t < _T_C_K:
        return (
            _P_TRIPLE_POINT
            * (t / _T_C_K) ** ((_CPV - _CVS) / _RGASV)
            * math.exp(
                (_E0V + _E0S - (_CVV - _CVS) * _T_C_K)
                / _RGASV
                * (1.0 / _T_C_K - 1.0 / t)
            )
        )
    return (
        _P_TRIPLE_POINT
        * (t / _T_C_K) ** ((_CPV - _CVL) / _RGASV)
        * math.exp((_E0V - (_CVV - _CVL) * _T_C_K) / _RGASV * (1.0 / _T_C_K - 1.0 / t))
    )


@njit(cache=True)
def _latent_heat_vap(t):  # The latent heat of vaporization of water
    return _E0V + (_CVV - _CVL) * (t - _T_C_K) + _RGASV * t


# .py_func calls the plain-Python version of these njit functions, so computing
# these two module-level constants doesn't trigger numba compilation at import
# time (which would otherwise add a noticeable, unnecessary import-time cost).
_P_CR = _PHI_SALT * _pv_star.py_func(_T_CR)  # core vapor pressure
_LAT_HEAT = _latent_heat_vap.py_func(310.0)  # latent heat of vaporization at 310 K


@njit(cache=True)
def _qv(ta, pa):  # respiratory heat loss, W/m^2
    return (
        _ETA
        * _Q
        * (_CPA * (_T_CR - ta) + _LAT_HEAT * _RGASA / (_P * _RGASV) * (_P_CR - pa))
    )


@njit(cache=True)
def _zs(rs):  # mass transfer resistance through skin, Pa m^2/W
    return 52.1 if rs == 0.0387 else 6.0e8 * rs**5


@njit(cache=True)
def _ra(ts, ta):  # heat transfer resistance through air, exposed skin, K m^2/W
    hc = 17.4
    phi_rad = 0.85
    hr = _EPSILON * phi_rad * _SIGMA * (ts**2 + ta**2) * (ts + ta)
    return 1.0 / (hc + hr)


@njit(cache=True)
def _ra_bar(tf, ta):  # heat transfer resistance through air, clothed skin, K m^2/W
    hc = 11.6
    phi_rad = 0.79
    hr = _EPSILON * phi_rad * _SIGMA * (tf**2 + ta**2) * (tf + ta)
    return 1.0 / (hc + hr)


@njit(cache=True)
def _ra_un(ts, ta):  # heat transfer resistance through air, when naked, K m^2/W
    hc = 12.3
    phi_rad = 0.80
    hr = _EPSILON * phi_rad * _SIGMA * (ts**2 + ta**2) * (ts + ta)
    return 1.0 / (hc + hr)


@njit(cache=True)
def _find_eq_var(ta, rh):
    """Given air temperature [K] and relative humidity, return the equivalent variable.

    Returns (eq_var_code, phi, rf, rs, d_tc_dt).
    """
    pa = rh * _pv_star(ta)  # air vapor pressure
    rs = 0.0387  # m^2K/W, heat transfer resistance through skin
    phi = 0.84  # covering fraction
    d_tc_dt = 0.0  # K/s, rate of change in Tc

    m = (_P_CR - pa) / (_zs(rs) + _ZA)
    m_bar = (_P_CR - pa) / (_zs(rs) + _ZA_BAR)

    lo_ts = max(0.0, min(_T_CR, ta) - rs * abs(m))
    hi_ts = max(_T_CR, ta) + rs * abs(m)
    ts = _bisect(kind=_R_TS, p0=ta, p1=pa, p2=rs, p3=0.0, x1=lo_ts, x2=hi_ts, tol=_TOL)

    lo_tf = max(0.0, min(_T_CR, ta) - rs * abs(m_bar))
    hi_tf = max(_T_CR, ta) + rs * abs(m_bar)
    tf = _bisect(kind=_R_TF, p0=ta, p1=pa, p2=rs, p3=0.0, x1=lo_tf, x2=hi_tf, tol=_TOL)

    q_minus_qv = _Q - _qv(ta, pa)
    flux1 = q_minus_qv - (1.0 - phi) * (_T_CR - ts) / rs  # C*dTc/dt when rf=Zf=inf
    flux2 = (
        q_minus_qv - (1.0 - phi) * (_T_CR - ts) / rs - phi * (_T_CR - tf) / rs
    )  # C*dTc/dt when rf=Zf=0

    if flux1 <= 0.0:  # region I
        eq_var_code = _EQ_PHI
        phi = 1.0 - q_minus_qv * rs / (_T_CR - ts)
        rf = np.inf
    elif flux2 <= 0.0:  # region II&III
        eq_var_code = _EQ_RF
        ts_bar = _T_CR - q_minus_qv * rs / phi + (1.0 / phi - 1.0) * (_T_CR - ts)
        tf = _bisect(
            kind=_R_TF_REGION_II_III,
            p0=ta,
            p1=pa,
            p2=rs,
            p3=ts_bar,
            x1=ta,
            x2=ts_bar,
            tol=_TOL,
        )
        rf = _ra_bar(tf, ta) * (ts_bar - tf) / (tf - ta)
    else:  # region IV,V,VI
        rf = 0.0
        flux3 = (
            q_minus_qv
            - (_T_CR - ta) / _ra_un(_T_CR, ta)
            - (_PHI_SALT * _pv_star(_T_CR) - pa) / _ZA_UN
        )
        if flux3 < 0.0:  # region IV,V
            ts = _bisect(
                kind=_R_TS_REGION_IV_V,
                p0=ta,
                p1=pa,
                p2=0.0,
                p3=0.0,
                x1=0.0,
                x2=_T_CR,
                tol=_TOL,
            )
            rs = (_T_CR - ts) / q_minus_qv
            eq_var_code = _EQ_RS
            ps = _P_CR - (_P_CR - pa) * _zs(rs) / (_zs(rs) + _ZA_UN)
            if ps > _PHI_SALT * _pv_star(ts):  # region V
                ts = _bisect(
                    kind=_R_TS_REGION_V,
                    p0=ta,
                    p1=pa,
                    p2=0.0,
                    p3=0.0,
                    x1=0.0,
                    x2=_T_CR,
                    tol=_TOL,
                )
                rs = (_T_CR - ts) / q_minus_qv
                eq_var_code = _EQ_RS
        else:  # region VI
            rs = 0.0
            eq_var_code = _EQ_DTCDT
            d_tc_dt = (1.0 / _HC_CORE) * flux3

    return eq_var_code, phi, rf, rs, d_tc_dt


@njit(cache=True)
def _residual(kind, x, p0, p1, p2, p3):
    if kind == _R_TS:  # x=ts, p0=ta, p1=pa, p2=rs
        ta, pa, rs = p0, p1, p2
        return (x - ta) / _ra(x, ta) + (_P_CR - pa) / (_zs(rs) + _ZA) - (_T_CR - x) / rs
    if kind == _R_TF:  # x=tf, p0=ta, p1=pa, p2=rs
        ta, pa, rs = p0, p1, p2
        return (
            (x - ta) / _ra_bar(x, ta)
            + (_P_CR - pa) / (_zs(rs) + _ZA_BAR)
            - (_T_CR - x) / rs
        )
    if kind == _R_TF_REGION_II_III:  # x=tf, p0=ta, p1=pa, p2=rs, p3=ts_bar
        ta, pa, rs, ts_bar = p0, p1, p2, p3
        return (
            (x - ta) / _ra_bar(x, ta)
            + (_P_CR - pa)
            * (x - ta)
            / ((_zs(rs) + _ZA_BAR) * (x - ta) + _R * _ra_bar(x, ta) * (ts_bar - x))
            - (_T_CR - ts_bar) / rs
        )
    if kind == _R_TS_REGION_IV_V:  # x=ts, p0=ta, p1=pa
        ta, pa = p0, p1
        q_minus_qv = _Q - _qv(ta, pa)
        return (
            (x - ta) / _ra_un(x, ta)
            + (_P_CR - pa) / (_zs((_T_CR - x) / q_minus_qv) + _ZA_UN)
            - q_minus_qv
        )
    if kind == _R_TS_REGION_V:  # x=ts, p0=ta, p1=pa
        ta, pa = p0, p1
        q_minus_qv = _Q - _qv(ta, pa)
        return (
            (x - ta) / _ra_un(x, ta)
            + (_PHI_SALT * _pv_star(x) - pa) / _ZA_UN
            - q_minus_qv
        )
    if kind == _R_T_PHI:  # x=t, p0=eq_var
        _, phi, _, _, _ = _find_eq_var(x, 1.0)
        return phi - p0
    if kind == _R_T_RF:  # x=t, p0=eq_var
        rh_trial = min(1.0, _PA0 / _pv_star(x))
        _, _, rf, _, _ = _find_eq_var(x, rh_trial)
        return rf - p0
    if kind == _R_T_RS:  # x=t, p0=eq_var
        rh_trial = _PA0 / _pv_star(x)
        _, _, _, rs, _ = _find_eq_var(x, rh_trial)
        return rs - p0
    # kind == _R_T_DTCDT: x=t, p0=eq_var
    rh_trial = _PA0 / _pv_star(x)
    _, _, _, _, d_tc_dt = _find_eq_var(x, rh_trial)
    return d_tc_dt - p0


@njit(cache=True)
def _bisect(kind, p0, p1, p2, p3, x1, x2, tol):
    a = x1
    b = x2
    fa = _residual(kind=kind, x=a, p0=p0, p1=p1, p2=p2, p3=p3)
    fb = _residual(kind=kind, x=b, p0=p0, p1=p1, p2=p2, p3=p3)
    if fa * fb > 0.0:
        raise ValueError("wrong initial interval in the root solver")
    c = b
    for i in range(_MAX_ITER):
        c = (a + b) / 2.0
        fc = _residual(kind=kind, x=c, p0=p0, p1=p1, p2=p2, p3=p3)
        if fb * fc > 0.0:
            b = c
            fb = fc
        else:
            a = c
        if abs(a - b) < tol:
            return c
        if i == _MAX_ITER - 1:
            raise ValueError("reaching maximum iteration in the root solver")
    return c


@njit(cache=True)
def _find_t(eq_var_code, eq_var):
    if eq_var_code == _EQ_PHI:
        return _bisect(
            kind=_R_T_PHI,
            p0=eq_var,
            p1=0.0,
            p2=0.0,
            p3=0.0,
            x1=0.0,
            x2=240.0,
            tol=_TOL_T,
        )
    if eq_var_code == _EQ_RF:
        return _bisect(
            kind=_R_T_RF,
            p0=eq_var,
            p1=0.0,
            p2=0.0,
            p3=0.0,
            x1=230.0,
            x2=300.0,
            tol=_TOL_T,
        )
    if eq_var_code == _EQ_RS:
        return _bisect(
            kind=_R_T_RS,
            p0=eq_var,
            p1=0.0,
            p2=0.0,
            p3=0.0,
            x1=295.0,
            x2=350.0,
            tol=_TOL_T,
        )
    return _bisect(
        kind=_R_T_DTCDT,
        p0=eq_var,
        p1=0.0,
        p2=0.0,
        p3=0.0,
        x1=340.0,
        x2=1000.0,
        tol=_TOL_T,
    )


@cast(
    Callable[
        [Callable[[float, float], float]],
        Callable[
            [
                float | NDArray[np.float64],
                float | NDArray[np.float64],
            ],
            np.float64 | NDArray[np.float64],
        ],
    ],
    vectorize(
        [
            float64(float64, float64),
        ],
        cache=True,
    ),
)
def _lu_heat_index_optimized(tdb: float, rh: float) -> float:
    # combining the two functions find_eq_var and find_t
    eq_var_code, phi, rf, rs, d_tc_dt = _find_eq_var(tdb, rh)
    if eq_var_code == _EQ_PHI:
        eq_var = phi
    elif eq_var_code == _EQ_RF:
        eq_var = rf
    elif eq_var_code == _EQ_RS:
        eq_var = rs
    else:
        eq_var = d_tc_dt
    hi = _find_t(eq_var_code, eq_var)
    if tdb == 0.0:
        hi = 0.0
    return hi
