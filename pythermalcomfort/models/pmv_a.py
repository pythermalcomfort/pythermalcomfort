from __future__ import annotations

from typing import Literal

import numpy as np

from pythermalcomfort.classes_input import APMVInputs, NumericInput
from pythermalcomfort.classes_return import APMV
from pythermalcomfort.models.pmv_ppd_iso import pmv_ppd_iso
from pythermalcomfort.utilities import Models, Units


def pmv_a(
    tdb: NumericInput,
    tr: NumericInput,
    vr: NumericInput,
    rh: NumericInput,
    met: NumericInput,
    clo: NumericInput,
    a_coefficient: float,
    wme: NumericInput = 0,
    units: Literal["SI", "IP"] = Units.SI.value,
    limit_inputs: bool = True,
) -> APMV:
    """Return Adaptive Predicted Mean Vote (aPMV) [Yao2009]_. This index was developed
    by Yao, R. et al. (2009). The model takes into account factors such as culture,
    climate, social, psychological, and behavioral adaptations, which have an impact on
    the senses used to detect thermal comfort. This model uses an adaptive coefficient
    (λ) representing the adaptive factors that affect the sense of thermal comfort.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, default in [°C] or [°F] if `units` = 'IP'.
    tr : float or list of floats
        Mean radiant temperature, default in [°C] or [°F] if `units` = 'IP'.
    vr : float or list of floats
        Relative air speed, default in [m/s] or [fps] if `units` = 'IP'.

        .. note::
            vr is the sum of the average air speed measured by the sensor and the activity-generated air speed (Vag).
            Calculate vr using :py:meth:`pythermalcomfort.environment.v_relative`.

    rh : float or list of floats
        Relative humidity, [%].
    met : float or list of floats
        Metabolic rate, [met].
    clo : float or list of floats
        Clothing insulation, [clo].

        .. note::
            Correct for body movement effects using :py:meth:`pythermalcomfort.clothing.clo_dynamic_iso`.

    a_coefficient : float
        Adaptive coefficient.
    wme : float or list of floats, optional
        External work, [met], default is 0.
    units : str, optional
        Units system, 'SI' or 'IP'. Defaults to 'SI'.
    limit_inputs : bool, optional
        If True, returns nan for inputs outside standard limits. Defaults to True.

        .. note::
            ISO 7730 [7730ISO2005]_ [7730ISO2025]_ limits: 10 < tdb [°C] < 30, 10 < tr [°C] < 40, 0 < vr [m/s] < 1,
            0.8 < met [met] < 4, 0 < clo [clo] < 2, 0 < pa [Pa] < 2700, -2 < PMV < 2.

    Returns
    -------
    APMV
        A dataclass containing the Predicted Mean Vote (a_pmv). See
        :py:class:`~pythermalcomfort.classes_return.AdaptivePMV` for more details.
        To access the `a_pmv` value, use the `a_pmv` attribute of the returned `AdaptivePMV` instance,
        e.g., `result.a_pmv`.

    Examples
    --------
    .. code-block:: python
        :emphasize-lines: 9,12,14

        from pythermalcomfort.models import pmv_a
        from pythermalcomfort.clothing import clo_dynamic_iso
        from pythermalcomfort.environment import v_relative

        v = 0.1
        met = 1.4
        clo = 0.5

        # Calculate relative air speed
        v_r = v_relative(v=v, met=met)

        # Calculate dynamic clothing
        clo_d = clo_dynamic_iso(clo=clo, met=met, v=v)

        results = pmv_a(
            tdb=28,
            tr=28,
            vr=v_r,
            rh=50,
            met=met,
            clo=clo_d,
            a_coefficient=0.293,
        )
        print(results)  # AdaptivePMV(a_pmv=0.74)
        print(results.a_pmv)  # 0.71
    """
    # Validate inputs using the APMVInputs class
    APMVInputs(
        tdb=tdb,
        tr=tr,
        vr=vr,
        rh=rh,
        met=met,
        clo=clo,
        a_coefficient=a_coefficient,
        wme=wme,
        units=units,
    )

    _pmv = pmv_ppd_iso(
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        wme,
        model=Models.iso_7730_2025.value,
        units=units,
        limit_inputs=limit_inputs,
    ).pmv

    pmv_value = np.around(_pmv / (1 + a_coefficient * _pmv), 2)

    return APMV(a_pmv=pmv_value)
