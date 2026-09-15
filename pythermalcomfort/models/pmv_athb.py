from __future__ import annotations

import numpy as np

from pythermalcomfort.classes_input import ATHBInputs, NumericInput
from pythermalcomfort.classes_return import ATHB
from pythermalcomfort.models._pmv_ppd_optimized import _pmv_ppd_optimized
from pythermalcomfort.utilities import met_to_w_m2


def pmv_athb(
    tdb: NumericInput,
    tr: NumericInput,
    vr: NumericInput,
    rh: NumericInput,
    met: NumericInput,
    t_running_mean: NumericInput,
    clo: bool | NumericInput = False,
) -> ATHB:
    """Return the PMV value calculated with the Adaptive Thermal Heat Balance Framework
    [Schweiker2022]_. The adaptive thermal heat balance (ATHB) framework introduced a
    method to account for the three adaptive principals, namely physiological,
    behavioral, and psychological adaptation, individually within existing heat balance
    models. The objective is a predictive model of thermal sensation applicable during
    the design stage or in international standards without knowing characteristics of
    future occupants.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, in [°C].
    tr : float or list of floats
        Mean radiant temperature, in [°C].
    vr : float or list of floats
        Relative air speed, in [m/s].

        .. note::
            vr is the relative air speed caused by body movement and not the air
            speed measured by the air speed sensor. The relative air speed is the sum of the
            average air speed measured by the sensor plus the activity-generated air speed
            (Vag). Where Vag is the activity-generated air speed caused by motion of
            individual body parts. vr can be calculated using the function
            :py:meth:`pythermalcomfort.environment.v_relative`.

    rh : float or list of floats
        Relative humidity, [%].
    met : float or list of floats
        Metabolic rate, [met].
    clo : bool, float or list of floats, optional
        Clothing insulation, in [clo]. If `clo` is set to False, the clothing insulation
        value will be calculated using the equation provided in the paper. Defaults to False.
    t_running_mean: float or list of floats
        Running mean temperature, in [°C].

        .. note::
            The running mean temperature can be calculated using the function
            :py:meth:`pythermalcomfort.environment.running_mean_outdoor_temperature`.

    Returns
    -------
    ATHB
        Dataclass containing the results of the ATHB calculation. See :py:class:`~pythermalcomfort.classes_return.ATHB` for more details.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import pmv_athb

        # calculate the predicted mean vote (PMV) using the Adaptive Thermal Heat Balance model
        results = pmv_athb(tdb=25, tr=25, vr=0.1, rh=50, met=1.2, t_running_mean=20)
        print(results.athb_pmv)  # returns the PMV value

        # for multiple points
        results = pmv_athb(
            tdb=[25, 25, 25],
            tr=[25, 25, 25],
            vr=[0.1, 0.1, 0.1],
            rh=[50, 50, 50],
            met=[1.2, 1.2, 1.2],
            t_running_mean=[20, 20, 20],
        )
        print(results.athb_pmv)
    """
    # Validate inputs using the ATHBInputs class
    ATHBInputs(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, t_running_mean=t_running_mean)

    tdb = np.asarray(tdb)
    tr = np.asarray(tr)
    vr = np.asarray(vr)
    met = np.asarray(met)
    rh = np.asarray(rh)
    t_running_mean = np.asarray(t_running_mean)

    met_adapted = met - (0.234 * t_running_mean) / 58.2

    clo_adapted = clo
    if clo is False:
        # Adapted clothing insulation level through behavioral adaptation
        clo_adapted = np.power(
            10,
            (
                -0.17168
                - 0.000485 * t_running_mean
                + 0.08176 * met_adapted
                - 0.00527 * t_running_mean * met_adapted
            ),
        )

    pmv_res = _pmv_ppd_optimized(tdb, tr, vr, rh, met_adapted, clo_adapted, 0)
    ts = 0.303 * np.exp(-0.036 * met_adapted * met_to_w_m2) + 0.028
    l_adapted = pmv_res / ts

    # Predicted thermal sensation vote
    athb_pmv = np.around(
        1.484
        + 0.0276 * l_adapted
        - 0.9602 * met_adapted
        - 0.0342 * t_running_mean
        + 0.0002264 * l_adapted * t_running_mean
        + 0.018696 * met_adapted * t_running_mean
        - 0.0002909 * l_adapted * met_adapted * t_running_mean,
        3,
    )

    return ATHB(athb_pmv=athb_pmv)
