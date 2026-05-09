from __future__ import annotations

import numpy as np

from pythermalcomfort.classes_input import PMVPPDInputs
from pythermalcomfort.classes_return import PMVPPD
from pythermalcomfort.models._pmv_ppd_optimized import _pmv_ppd_optimized
from pythermalcomfort.shared_functions import mapping, valid_range
from pythermalcomfort.utilities import (
    Models,
    Units,
    _check_standard_compliance_array,
    units_converter,
)


def pmv_ppd_iso(
    tdb: float | list[float],
    tr: float | list[float],
    vr: float | list[float],
    rh: float | list[float],
    met: float | list[float],
    clo: float | list[float],
    wme: float | list[float] = 0,
    model: str = Models.iso_7730_2005.value,
    units: str = Units.SI.value,
    limit_inputs: bool = True,
    round_output: bool = True,
) -> PMVPPD:
    """Return Predicted Mean Vote (PMV) and Predicted Percentage of Dissatisfied (PPD)
    calculated in accordance with the ISO 7730.

    The ISO uses the same formulation of the PMV as published by Fanger [Fanger1970]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C] in [°F] if `units` = 'IP'
    tr : float or list of floats
        Mean radiant temperature, [°C] in [°F] if `units` = 'IP'
    vr : float or list of floats
        Relative air speed, [m/s] in [fps] if `units` = 'IP'

        .. note::
            vr is the relative air speed caused by body movement and not the air
            speed measured by the air speed sensor. The relative air speed is the sum of the
            average air speed measured by the sensor plus the activity-generated air speed
            (Vag). Where Vag is the activity-generated air speed caused by motion of
            individual body parts. vr can be calculated using the function
            :py:meth:`pythermalcomfort.utilities.v_relative`.

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
            :py:meth:`pythermalcomfort.utilities.clo_dynamic_iso`.

    wme : float or list of floats, optional
        External work, [met]. Defaults to 0.
    model : str, optional
        Select the model you want to use to calculate the PMV. Currently, the only option available is "7730-2005".
    units : str, optional
        Select the SI (International System of Units) or the IP (Imperial Units) system.
        Supported values are 'SI' and 'IP'. Defaults to 'SI'.
    limit_inputs : bool, optional
        If True, limits the inputs to the standard applicability limits. Defaults to True.

        .. note::
            By default, if the inputs are outside the standard applicability limits the
            function returns nan. If False returns pmv and ppd values even if input values are
            outside the applicability limits of the model.

            The ISO 7730 2005 limits are 10 < tdb [°C] < 30, 10 < tr [°C] < 40,
            0 < vr [m/s] < 1, 0.8 < met [met] < 4, 0 < clo [clo] < 2, and -2 < PMV < 2.

    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    PMVPPD
        A dataclass containing the Predicted Mean Vote and Predicted Percentage of
        Dissatisfied. See :py:class:`~pythermalcomfort.classes_return.PMVPPD` for
        more details. To access the `pmv`, `ppd`, `tsv` values, use the corresponding
        attributes of the returned `PMVPPD` instance, e.g., `result.pmv`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import pmv_ppd_iso
        from pythermalcomfort.utilities import v_relative

        tdb = 25
        tr = 25
        rh = 50
        v = 0.1
        met = 1.4
        clo_dynamic = 0.5
        # calculate relative air speed
        v_r = v_relative(v=v, met=met)
        results = pmv_ppd_iso(
            tdb=tdb,
            tr=tr,
            vr=v_r,
            rh=rh,
            met=met,
            clo=clo_dynamic,
            model="7730-2005",
        )
        print(results.pmv)  # 0.17
        print(results.ppd)  # 5.6

        result = pmv_ppd_iso(
            tdb=[22, 25],
            tr=25,
            vr=0.1,
            rh=50,
            met=1.4,
            clo=0.5,
            model="7730-2005",
        )
        print(result.pmv)  # [-0.  0.41]
        print(result.ppd)  # [5.  8.5]
    """
    # Validate inputs using the PMVPPDInputs class
    PMVPPDInputs(
        tdb=tdb,
        tr=tr,
        vr=vr,
        rh=rh,
        met=met,
        clo=clo,
        wme=wme,
        units=units,
        limit_inputs=limit_inputs,
    )

    tdb = np.asarray(tdb)
    tr = np.asarray(tr)
    rh = np.asarray(rh)
    vr = np.asarray(vr)
    met = np.asarray(met)
    clo = np.asarray(clo)
    wme = np.asarray(wme)

    if units.upper() == Units.IP.value:
        tdb, tr, vr = units_converter(tdb=tdb, tr=tr, v=vr)

    model = model.lower()
    if model not in [Models.iso_7730_2005.value]:
        raise ValueError(
            "PMV calculations can only be performed in compliance with ISO 7730-2005",
        )

    pmv_array = _pmv_ppd_optimized(tdb, tr, vr, rh, met, clo, wme)

    ppd_array = 100.0 - 95.0 * np.exp(
        -0.03353 * pmv_array**4.0 - 0.2179 * pmv_array**2.0,
    )

    # Checks that inputs are within the bounds accepted by the model if not return nan
    if limit_inputs:
        (
            tdb_valid,
            tr_valid,
            v_valid,
            met_valid,
            clo_valid,
        ) = _check_standard_compliance_array(
            standard=Models.iso_7730_2005.value,
            tdb=tdb,
            tr=tr,
            v=vr,
            met=met,
            clo=clo,
        )
        pmv_valid = valid_range(pmv_array, (-2, 2))  # this is the ISO limit

        all_valid = ~(
            np.isnan(tdb_valid)
            | np.isnan(tr_valid)
            | np.isnan(v_valid)
            | np.isnan(met_valid)
            | np.isnan(clo_valid)
            | np.isnan(pmv_valid)
        )
        pmv_array = np.where(all_valid, pmv_array, np.nan)
        ppd_array = np.where(all_valid, ppd_array, np.nan)

    if round_output:
        pmv_array = np.round(pmv_array, 2)
        ppd_array = np.round(ppd_array, 1)

    thermal_sensation = {
        -2.5: "Cold",
        -1.5: "Cool",
        -0.5: "Slightly Cool",
        0.5: "Neutral",
        1.5: "Slightly Warm",
        2.5: "Warm",
        10: "Hot",
    }

    return PMVPPD(
        pmv=pmv_array,
        ppd=ppd_array,
        tsv=mapping(pmv_array, thermal_sensation, right=False),
    )
