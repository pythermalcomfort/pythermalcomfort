from typing import Literal

import numpy as np

from pythermalcomfort.classes_input import ASHRAEInputs
from pythermalcomfort.classes_return import AdaptiveASHRAE
from pythermalcomfort.shared_functions import valid_range
from pythermalcomfort.utilities import (
    Models,
    Units,
    _check_standard_compliance_array,
    operative_tmp,
    units_converter,
)


def adaptive_ashrae(
    tdb: float | list[float],
    tr: float | list[float],
    t_running_mean: float | list[float],
    v: float | list[float],
    units: Literal["SI", "IP"] = Units.SI.value,
    limit_inputs: bool = True,
) -> AdaptiveASHRAE:
    """Calculate the adaptive thermal comfort based on ASHRAE 55.

    The adaptive
    model relates indoor design temperatures or acceptable temperature ranges
    to outdoor meteorological or climatological parameters. The adaptive model
    can only be used in occupant-controlled naturally conditioned spaces that
    meet all the following criteria:

    * There is no mechanical cooling or heating system in operation.
    * Occupants have a metabolic rate between 1.0 and 1.5 met.
    * Occupants are free to adapt their clothing within a range as wide as 0.5 and 1.0 clo.
    * The prevailing mean (running mean) outdoor temperature is between 10 and 33.5 °C.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, default in [°C] or [°F] if `units` = 'IP'.
    tr : float or list of floats
        Mean radiant temperature, default in [°C] or [°F] if `units` = 'IP'.
    t_running_mean : float or list of floats
        Running mean temperature, default in [°C] or [°F] if `units` = 'IP'.

        .. note::
            The running mean temperature can be calculated using the function :py:meth:`pythermalcomfort.utilities.running_mean_outdoor_temperature`.

    v : float or list of floats
        Air speed, default in [m/s] or [fps] if `units` = 'IP'.
    units : str, optional
        Units system, 'SI' or 'IP'. Defaults to 'SI'.
    limit_inputs : bool, optional
        If True, returns nan for inputs outside standard limits. Defaults to True.

        .. note::
            ASHRAE 55 2020 limits: 10 < tdb [°C] < 40, 10 < tr [°C] < 40, 0 < vr [m/s] < 2, 10 < t_running_mean [°C] < 33.5.

    Returns
    -------
    AdaptiveASHRAE
        A dataclass containing the results. See :py:class:`~pythermalcomfort.classes_return.AdaptiveASHRAE` for more details.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import adaptive_ashrae

        results = adaptive_ashrae(tdb=25, tr=25, t_running_mean=20, v=0.1)
        print(results)
        # AdaptiveASHRAE(tmp_cmf=np.float64(24.0), tmp_cmf_80_low=np.float64(20.5), tmp_cmf_80_up=np.float64(27.5), tmp_cmf_90_low=np.float64(21.5), tmp_cmf_90_up=np.float64(26.5), acceptability_80=array(True), acceptability_90=array(True))

        print(results.acceptability_80)  # or use print(results["acceptability_80"])
        # True
        # The conditions you entered are considered to be comfortable for 80% of the occupants.

        # You can also pass arrays as input to the function
        results = adaptive_ashrae(tdb=[25, 26], tr=25, t_running_mean=20, v=0.1)
        print(results)
        # AdaptiveASHRAE(tmp_cmf=array([24., 24.]), tmp_cmf_80_low=array([20.5, 20.5]), tmp_cmf_80_up=array([27.5, 27.5]), tmp_cmf_90_low=array([21.5, 21.5]), tmp_cmf_90_up=array([26.5, 26.5]), acceptability_80=array([ True,  True]), acceptability_90=array([ True,  True]))

        # For users who want to use the IP system
        results = adaptive_ashrae(tdb=77, tr=77, t_running_mean=68, v=0.3, units="IP")
        print(results)
        # AdaptiveASHRAE(tmp_cmf=np.float64(75.2), tmp_cmf_80_low=np.float64(68.9), tmp_cmf_80_up=np.float64(81.5), tmp_cmf_90_low=np.float64(70.7), tmp_cmf_90_up=np.float64(79.7), acceptability_80=array(True), acceptability_90=array(True))

        adaptive_ashrae(tdb=25, tr=25, t_running_mean=9, v=0.1)
        # AdaptiveASHRAE(tmp_cmf=np.float64(nan), ... acceptability_90=array(False))
        # The adaptive thermal comfort model can only be used if the running mean temperature is higher than 10°C.
    """
    # Validate inputs using the ASHRAEInputs class
    ASHRAEInputs(
        tdb=tdb,
        tr=tr,
        t_running_mean=t_running_mean,
        v=v,
        units=units,
    )

    tdb = np.asarray(tdb)
    tr = np.asarray(tr)
    t_running_mean = np.asarray(t_running_mean)
    v = np.asarray(v)
    standard = "ashrae"

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

    # Relation between comfort and outdoor temperature
    t_cmf = 0.31 * t_running_mean + 17.8

    if limit_inputs:
        tdb_valid, tr_valid, v_valid = _check_standard_compliance_array(
            Models.ashrae_55_2023.value,
            tdb=tdb,
            tr=tr,
            v=v,
        )
        trm_valid = valid_range(t_running_mean, (10.0, 33.5), "t_running_mean")
        all_valid = ~(
            np.isnan(tdb_valid)
            | np.isnan(tr_valid)
            | np.isnan(v_valid)
            | np.isnan(trm_valid)
        )
        t_cmf = np.where(all_valid, t_cmf, np.nan)

    t_cmf = np.around(t_cmf, 1)

    tmp_cmf_80_low = t_cmf - 3.5
    tmp_cmf_90_low = t_cmf - 2.5
    tmp_cmf_80_up = t_cmf + 3.5 + ce
    tmp_cmf_90_up = t_cmf + 2.5 + ce

    acceptability_80 = (tmp_cmf_80_low <= to) & (to <= tmp_cmf_80_up)
    acceptability_90 = (tmp_cmf_90_low <= to) & (to <= tmp_cmf_90_up)

    if units.upper() == Units.IP.value:
        (
            t_cmf,
            tmp_cmf_80_low,
            tmp_cmf_80_up,
            tmp_cmf_90_low,
            tmp_cmf_90_up,
        ) = units_converter(
            from_units=Units.SI.value,
            tmp_cmf=t_cmf,
            tmp_cmf_80_low=tmp_cmf_80_low,
            tmp_cmf_80_up=tmp_cmf_80_up,
            tmp_cmf_90_low=tmp_cmf_90_low,
            tmp_cmf_90_up=tmp_cmf_90_up,
        )

    return AdaptiveASHRAE(
        tmp_cmf=t_cmf,
        tmp_cmf_80_low=tmp_cmf_80_low,
        tmp_cmf_80_up=tmp_cmf_80_up,
        tmp_cmf_90_low=tmp_cmf_90_low,
        tmp_cmf_90_up=tmp_cmf_90_up,
        acceptability_80=acceptability_80,
        acceptability_90=acceptability_90,
    )
