from __future__ import annotations

import numpy as np

from pythermalcomfort.classes_input import EPMVInputs, NumericInput
from pythermalcomfort.classes_return import EPMV
from pythermalcomfort.models.pmv_ppd_iso import pmv_ppd_iso
from pythermalcomfort.utilities import Models, Units


def pmv_e(
    tdb: NumericInput,
    tr: NumericInput,
    vr: NumericInput,
    rh: NumericInput,
    met: NumericInput,
    clo: NumericInput,
    e_coefficient: NumericInput,
    wme: NumericInput = 0,
    units: str = Units.SI.value,
    limit_inputs: bool = True,
) -> EPMV:
    """Return Adjusted Predicted Mean Votes with Expectancy Factor (ePMV). This index
    was developed by Fanger, P. O. et al. (2002). In non-air- conditioned buildings in
    warm climates, occupants may sense the warmth as being less severe than the PMV
    predicts. The main reason is low expectations, but a metabolic rate that is
    estimated too high can also contribute to explaining the difference. An extension of
    the PMV model that includes an expectancy factor is introduced for use in non-air-
    conditioned buildings in warm climates [Fanger2002]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, default in [°C] in [°F] if `units` = 'IP'.
    tr : float or list of floats
        Mean radiant temperature, default in [°C] in [°F] if `units` = 'IP'.
    vr : float or list of floats
        Relative air speed, default in [m/s] in [fps] if `units` = 'IP'.

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
    clo : float or list of floats
        Clothing insulation, [clo].

        .. note::
            this is the basic insulation also known as the intrinsic clothing insulation value of the
            clothing ensemble (`I`:sub:`cl,r`), this is the thermal insulation from the skin
            surface to the outer clothing surface, including enclosed air layers, under actual
            environmental conditions. This value is not the total insulation (`I`:sub:`T,r`).
            The dynamic clothing insulation, clo, can be calculated using the function
            :py:meth:`pythermalcomfort.clothing.clo_dynamic_iso`.

    e_coefficient : float or list of floats
        Expectancy factor.
    wme : float or list of floats, optional
        External work, [met]. Defaults to 0.
    units : {'SI', 'IP'}, optional
        Select the SI (International System of Units) or the IP (Imperial Units) system.
        Supported values are 'SI' and 'IP'. Defaults to 'SI'.
    limit_inputs : bool, optional
        By default, if the inputs are outside the standard applicability limits the
        function returns nan. If False, returns pmv and ppd values even if input values are
        outside the applicability limits of the model. Defaults to True.

        .. note::
            The ISO 7730 [7730ISO2005]_ [7730ISO2025]_ limits are 10 < tdb [°C] < 30, 10 < tr [°C] < 40,
            0 < vr [m/s] < 1, 0.8 < met [met] < 4, 0 < clo [clo] < 2, 0 < pa [Pa] < 2700, and -2 < PMV < 2.

    Returns
    -------
    EPMV
        A dataclass containing the Adjusted Predicted Mean Votes with Expectancy Factor. See :py:class:`~pythermalcomfort.classes_return.EPMV` for more details.
        To access the `e_pmv` value, use the `e_pmv` attribute of the returned `e_pmv` instance, e.g., `result.e_pmv`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import pmv_e
        from pythermalcomfort.clothing import clo_dynamic_iso
        from pythermalcomfort.environment import v_relative

        tdb = 28
        tr = 28
        rh = 50
        v = 0.1
        met = 1.4
        clo = 0.5
        # calculate relative air speed
        v_r = v_relative(v=v, met=met)
        # Calculate dynamic clothing
        clo_d = clo_dynamic_iso(clo=clo, met=met, v=v)
        results = pmv_e(tdb, tr, v_r, rh, met, clo_d, e_coefficient=0.6)
        print(results.e_pmv)  # 0.48
    """
    # Validate inputs using the EPMVInputs class
    EPMVInputs(
        tdb=tdb,
        tr=tr,
        vr=vr,
        rh=rh,
        met=met,
        clo=clo,
        e_coefficient=e_coefficient,
        wme=wme,
        units=units,
    )

    default_kwargs = {"units": units, "limit_inputs": limit_inputs}
    met = np.asarray(met)
    _pmv = pmv_ppd_iso(
        tdb=tdb,
        tr=tr,
        vr=vr,
        rh=rh,
        met=met,
        clo=clo,
        wme=wme,
        model=Models.iso_7730_2025.value,
        **default_kwargs,
    ).pmv
    met = np.where(_pmv > 0, met * (1 + _pmv * (-0.067)), met)
    _pmv = pmv_ppd_iso(
        tdb=tdb,
        tr=tr,
        vr=vr,
        rh=rh,
        met=met,
        clo=clo,
        wme=wme,
        model=Models.iso_7730_2025.value,
        **default_kwargs,
    ).pmv

    e_pmv_value = np.around(_pmv * e_coefficient, 2)

    return EPMV(e_pmv=e_pmv_value)
