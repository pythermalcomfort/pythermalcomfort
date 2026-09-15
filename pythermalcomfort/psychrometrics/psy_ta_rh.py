from __future__ import annotations

import numpy as np

from pythermalcomfort.classes_return import PsychrometricValues
from pythermalcomfort.utilities import NumericInput

from .dew_point_tmp import dew_point_tmp
from .enthalpy_air import enthalpy_air
from .p_sat import p_sat
from .wet_bulb_tmp import wet_bulb_tmp


def psy_ta_rh(
    tdb: NumericInput,
    rh: NumericInput,
    p_atm: float = 101325,
) -> PsychrometricValues:
    """Calculate psychrometric values of air based on dry bulb air temperature and
    relative humidity.

    For more accurate results we recommend the use of the Python
    package `psychrolib`_.

    .. _psychrolib: https://pypi.org/project/PsychroLib/

    Parameters
    ----------
    tdb: float or list of floats
        air temperature, [°C]
    rh: float or list of floats
        relative humidity, [%]
    p_atm: float or list of floats
        atmospheric pressure, [Pa]

    Returns
    -------
    p_vap: float or list of floats
        partial pressure of water vapor in moist air, [Pa]
    hr: float or list of floats
        humidity ratio, [kg water/kg dry air]
    wet_bulb_tmp: float or list of floats
        wet bulb temperature, [°C]
    dew_point_tmp: float or list of floats
        dew point temperature, [°C]
    h: float or list of floats
        enthalpy_air [J/kg dry air]

    Notes
    -----
    **Elevation and Barometric Sensitivity:**
    Psychrometric properties, particularly humidity ratio (:math:`W`) and
    mixture enthalpy (:math:`h`), are sensitive to atmospheric pressure
    (:math:`p_{atm}`).

    At high elevations (e.g., Denver, CO at 1,609 m / 5,280 ft, where
    :math:`p_{atm} \approx 82,500\text{ Pa}`), the moisture-holding capacity
    of dry air increases. At 25°C and 50% RH:
    - At sea level (101,325 Pa): :math:`W \approx 0.00988\text{ kg}_w/\text{kg}_{da}`
    - At 1,609 m (82,500 Pa): :math:`W \approx 0.01218\text{ kg}_w/\text{kg}_{da}` (+23.3% delta)

    Omitting local atmospheric pressure when calling comfort models that
    derive latent heat exchange or skin wettedness will result in systematic
    deviations in elevated or non-standard barometric conditions.

    For rigorous real-gas calculations incorporating enhancement factors
    (:math:`f_w`), see ASHRAE RP-1485 / Hyland-Wexler formulations. The
    `psychrolib` library provides a standard ideal-gas psychrometric
    implementation.
    """
    tdb = np.asarray(tdb, dtype=np.float64)
    rh = np.asarray(rh, dtype=np.float64)
    p_atm = np.asarray(p_atm, dtype=np.float64)

    p_saturation = p_sat(tdb)
    p_vap = rh / 100 * p_saturation
    hr = 0.62198 * p_vap / (p_atm - p_vap)
    tdp = dew_point_tmp(tdb, rh)
    twb = wet_bulb_tmp(tdb, rh)
    h = enthalpy_air(tdb, hr)

    return PsychrometricValues(
        p_sat=p_saturation,
        p_vap=p_vap,
        hr=hr,
        wet_bulb_tmp=twb,
        dew_point_tmp=tdp,
        h=h,
    )
