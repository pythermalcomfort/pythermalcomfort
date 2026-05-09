from typing import Literal

import numpy as np

from pythermalcomfort.classes_input import ENInputs
from pythermalcomfort.classes_return import AdaptiveEN
from pythermalcomfort.shared_functions import valid_range
from pythermalcomfort.utilities import Units, operative_tmp, units_converter


def adaptive_en(
    tdb: float | list[float],
    tr: float | list[float],
    t_running_mean: float | list[float],
    v: float | list[float],
    units: Literal["SI", "IP"] = Units.SI.value,
    limit_inputs: bool = True,
) -> AdaptiveEN:
    """Calculate the adaptive thermal comfort based on EN 16798-1 2019 [16798EN2019]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, default in [°C] or [°F] if `units` = 'IP'.
    tr : float or list of floats
        Mean radiant temperature, default in [°C] or [°F] if `units` = 'IP'.
    t_running_mean: float or list of floats
        Running mean temperature, default in [°C] or [°F] if `units` = 'IP'.

        .. note::
            The running mean temperature can be calculated using the function :py:meth:`pythermalcomfort.utilities.running_mean_outdoor_temperature`.

    v : float or list of floats
        Air speed, default in [m/s] or [fps] if `units` = 'IP'.

        .. note::
            Indoor operative temperature correction is applicable for buildings equipped
            with fans or personal systems providing building occupants with personal
            control over air speed at occupant level.
            For operative temperatures above 25°C the comfort zone upper limit can be
            increased by 1.2 °C (0.6 < v < 0.9 m/s), 1.8 °C (0.9 < v < 1.2 m/s), 2.2 °C (v
            > 1.2 m/s).

    units : {'SI', 'IP'}
        Select the SI (International System of Units) or the IP (Imperial Units) system.
    limit_inputs : bool, default True
        If True, returns NaN for inputs outside the standard applicability limits.

    Returns
    -------
    AdaptiveEN
        A dataclass containing the results. See :py:class:`~pythermalcomfort.classes_return.AdaptiveEN` for more details.


    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import adaptive_en

        results = adaptive_en(tdb=25, tr=25, t_running_mean=20, v=0.1)
        print(results)
        # AdaptiveEN(tmp_cmf=np.float64(25.4), acceptability_cat_i=np.True_, acceptability_cat_ii=np.True_, ...)

        print(results.acceptability_cat_i)  # or print(results["acceptability_cat_i"])
        # True
        # The conditions you entered are considered to comply with Category I

        # For users who want to use the IP system, units="IP" or "ip" are both valid
        results = adaptive_en(tdb=77, tr=77, t_running_mean=68, v=0.3, units="IP")
        print(results)
        # AdaptiveEN(tmp_cmf=np.float64(77.7), acceptability_cat_i=np.True_, ...)

        results = adaptive_en(tdb=25, tr=25, t_running_mean=9, v=0.1)
        print(results)
        # AdaptiveEN(tmp_cmf=np.float64(nan), acceptability_cat_i=np.False_, ...)
        # The adaptive thermal comfort model can only be used
        # if the running mean temperature is between 10 °C and 30 °C.
    """
    # Validate inputs using the ENInputs class
    ENInputs(tdb=tdb, tr=tr, t_running_mean=t_running_mean, v=v, units=units)

    tdb = np.asarray(tdb)
    tr = np.asarray(tr)
    t_running_mean = np.asarray(t_running_mean)
    v = np.asarray(v)
    standard = "iso"

    if units.upper() == Units.IP.value:
        tdb, tr, t_running_mean, v = units_converter(
            tdb=tdb,
            tr=tr,
            tmp_running_mean=t_running_mean,
            v=v,
        )

    to = operative_tmp(tdb, tr, v, standard=standard)

    # Calculate cooling effect (ce) of elevated air speed when top > 25 degC.
    ce = np.where((v >= 0.6) & (to >= 25.0), 999, 0)
    ce = np.where((v < 0.9) & (ce == 999), 1.2, ce)
    ce = np.where((v < 1.2) & (ce == 999), 1.8, ce)
    ce = np.where(ce == 999, 2.2, ce)

    t_cmf = 0.33 * t_running_mean + 18.8

    if limit_inputs:
        trm_valid = valid_range(t_running_mean, (10.0, 33.5), "t_running_mean")
        all_valid = ~(np.isnan(trm_valid))
        t_cmf = np.where(all_valid, t_cmf, np.nan)

    t_cmf_i_lower = t_cmf - 3.0
    t_cmf_ii_lower = t_cmf - 4.0
    t_cmf_iii_lower = t_cmf - 5.0
    t_cmf_i_upper = t_cmf + 2.0 + ce
    t_cmf_ii_upper = t_cmf + 3.0 + ce
    t_cmf_iii_upper = t_cmf + 4.0 + ce

    acceptability_i = (t_cmf_i_lower <= to) & (to <= t_cmf_i_upper)
    acceptability_ii = (t_cmf_ii_lower <= to) & (to <= t_cmf_ii_upper)
    acceptability_iii = (t_cmf_iii_lower <= to) & (to <= t_cmf_iii_upper)

    if units.upper() == Units.IP.value:
        t_cmf, t_cmf_i_upper, t_cmf_ii_upper, t_cmf_iii_upper = units_converter(
            from_units=Units.SI.value.lower(),
            tmp_cmf=t_cmf,
            tmp_cmf_cat_i_up=t_cmf_i_upper,
            tmp_cmf_cat_ii_up=t_cmf_ii_upper,
            tmp_cmf_cat_iii_up=t_cmf_iii_upper,
        )
        t_cmf_i_lower, t_cmf_ii_lower, t_cmf_iii_lower = units_converter(
            from_units=Units.SI.value.lower(),
            tmp_cmf_cat_i_low=t_cmf_i_lower,
            tmp_cmf_cat_ii_low=t_cmf_ii_lower,
            tmp_cmf_cat_iii_low=t_cmf_iii_lower,
        )

    return AdaptiveEN(
        tmp_cmf=np.around(t_cmf, 1),
        acceptability_cat_i=acceptability_i,
        acceptability_cat_ii=acceptability_ii,
        acceptability_cat_iii=acceptability_iii,
        tmp_cmf_cat_i_up=np.around(t_cmf_i_upper, 1),
        tmp_cmf_cat_ii_up=np.around(t_cmf_ii_upper, 1),
        tmp_cmf_cat_iii_up=np.around(t_cmf_iii_upper, 1),
        tmp_cmf_cat_i_low=np.around(t_cmf_i_lower, 1),
        tmp_cmf_cat_ii_low=np.around(t_cmf_ii_lower, 1),
        tmp_cmf_cat_iii_low=np.around(t_cmf_iii_lower, 1),
    )
