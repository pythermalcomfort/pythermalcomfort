from __future__ import annotations

import numpy as np

from pythermalcomfort._internal.validation import _mapping, _valid_range
from pythermalcomfort.classes_input import NumericInput, PMVPPDInputs
from pythermalcomfort.classes_return import PMVPPD
from pythermalcomfort.models._pmv_ppd_optimized import _pmv_ppd_optimized
from pythermalcomfort.utilities import (
    Models,
    Units,
    units_converter,
)


def pmv_ppd_iso(
    tdb: NumericInput,
    tr: NumericInput,
    vr: NumericInput,
    rh: NumericInput,
    met: NumericInput,
    clo: NumericInput,
    wme: NumericInput = 0,
    model: str = Models.iso_7730_2025.value,
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

    wme : float or list of floats, optional
        External work, [met]. Defaults to 0.
    model : str, optional
        The ISO 7730 edition to reference. Supported values are "7730-2005"
        and "7730-2025"; any other value raises a ``ValueError``. The two
        editions currently use identical PMV/PPD equations and applicability
        limits, so this parameter does not affect the numerical result. It is
        retained to record the intended edition, reject unsupported edition
        values, and allow future differentiation if the editions diverge. The
        selected edition is not stored on the returned ``PMVPPD`` object.
        Defaults to "7730-2025", the current edition of the standard.
    units : str, optional
        Select the SI (International System of Units) or the IP (Imperial Units) system.
        Supported values are 'SI' and 'IP'. Defaults to 'SI'.
    limit_inputs : bool, optional
        If True, limits the inputs to the standard applicability limits. Defaults to True.

        .. note::
            By default, if the inputs are outside the standard applicability limits the
            function returns nan. If False returns pmv and ppd values even if input values are
            outside the applicability limits of the model.

            The ISO 7730 limits are 10 < tdb [°C] < 30, 10 < tr [°C] < 40,
            0 < vr [m/s] < 1, 0.8 < met [met] < 4, 0 < clo [clo] < 2, 0 < pa [Pa] < 2700,
            and -2 < PMV < 2.

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
        from pythermalcomfort.environment import v_relative

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
            model="7730-2025",
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
            model="7730-2025",
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
    if model not in [Models.iso_7730_2005.value, Models.iso_7730_2025.value]:
        invalid_model_msg = (
            "PMV calculations can only be performed in compliance with "
            f"ISO {Models.iso_7730_2005.value} or ISO {Models.iso_7730_2025.value}"
        )
        raise ValueError(invalid_model_msg)

    pmv = _pmv_ppd_optimized(tdb, tr, vr, rh, met, clo, wme)

    ppd_array = 100.0 - 95.0 * np.exp(
        -0.03353 * pmv**4.0 - 0.2179 * pmv**2.0,
    )

    # Checks that inputs are within the bounds accepted by the model if not return nan
    if limit_inputs:
        # ISO 7730 Clause 4 applicability limits
        pa = rh * 10.0 * np.exp(16.6536 - 4030.183 / (tdb + 235.0))

        tdb_valid = _valid_range(tdb, (10.0, 30.0))
        tr_valid = _valid_range(tr, (10.0, 40.0))
        v_valid = _valid_range(vr, (0.0, 1.0))
        met_valid = _valid_range(met, (0.8, 4.0))
        clo_valid = _valid_range(clo, (0.0, 2.0))
        pa_valid = _valid_range(pa, (0.0, 2700.0))
        pmv_valid = _valid_range(pmv, (-2, 2))

        all_valid = ~(
            np.isnan(tdb_valid)
            | np.isnan(tr_valid)
            | np.isnan(v_valid)
            | np.isnan(met_valid)
            | np.isnan(clo_valid)
            | np.isnan(pa_valid)
            | np.isnan(pmv_valid)
        )
        pmv = np.where(all_valid, pmv, np.nan)
        ppd_array = np.where(all_valid, ppd_array, np.nan)

    if round_output:
        pmv = np.round(pmv, 2)
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
        pmv=pmv,
        ppd=ppd_array,
        tsv=_mapping(pmv, thermal_sensation, right=False),
    )
