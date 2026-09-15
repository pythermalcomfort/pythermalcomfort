from __future__ import annotations

import numpy as np

from pythermalcomfort._internal.ashrae55 import _check_ashrae55_compliance
from pythermalcomfort.classes_input import NumericInput, SETInputs
from pythermalcomfort.classes_return import SET
from pythermalcomfort.models.two_nodes_gagge import two_nodes_gagge
from pythermalcomfort.utilities import Postures


def set_tmp(
    tdb: NumericInput,
    tr: NumericInput,
    v: NumericInput,
    rh: NumericInput,
    met: NumericInput,
    clo: NumericInput,
    wme: NumericInput = 0.0,
    body_surface_area: NumericInput = 1.8258,
    p_atm: NumericInput = 101325,
    position: str = Postures.standing.value,
    limit_inputs: bool = True,
    round_output: bool = True,
    calculate_ce: bool = False,
) -> SET:
    """Calculate the Standard Effective Temperature (SET).

    The SET is the
    temperature of a hypothetical isothermal environment at 50% (rh), <0.1 m/s
    (20 fpm) average air speed (v), and tr = tdb, in which the total heat loss
    from the skin of an imaginary occupant wearing clothing, standardized for
    the activity concerned is the same as that from a person in the actual
    environment with actual clothing and activity level. [Gagge1986]_

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    tr : float or list of floats
        Mean radiant temperature, [°C].
    v : float or list of floats
        Air speed, [m/s].
    rh : float or list of floats
        Relative humidity, [%].
    met : float or list of floats
        Metabolic rate, [met].
    clo : float or list of floats
        Clothing insulation, [clo].
    wme : float or list of floats, optional
        External work, [met]. Defaults to 0.
    body_surface_area : float or list of floats, optional
        Body surface area, default value 1.8258 [m2]

        .. note::
            The body surface area can be calculated using the function
            :py:meth:`pythermalcomfort.utilities.body_surface_area`.

    p_atm : float or list of floats, optional
        Atmospheric pressure, default value 101325 [Pa]
    position : str, optional
        Select either "sitting" or "standing". Defaults to "standing".
    limit_inputs : bool, optional
        If True, limits the inputs to the standard applicability limits. Defaults to True.

        .. note::
            By default, if the inputs are outside the standard applicability limits the
            function returns nan. If False returns values regardless of the input values.
            The limits are 10 < tdb [°C] < 40, 10 < tr [°C] < 40,
            0 < v [m/s] < 2, 1 < met [met] < 4, and 0 < clo [clo] < 1.5.
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    SET
        A dataclass containing the Standard Effective Temperature. See :py:class:`~pythermalcomfort.classes_return.SetTmp` for more details.
        To access the `set` value, use the corresponding attribute of the returned `SetTmp` instance, e.g., `result.set`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import set_tmp

        result = set_tmp(tdb=25, tr=25, v=0.1, rh=50, met=1.2, clo=0.5)
        print(result.set)  # 24.3

        result = set_tmp(tdb=[25, 25], tr=25, v=0.1, rh=50, met=1.2, clo=0.5)
        print(result.set)  # [24.3, 24.3]
    """
    tdb = np.asarray(tdb)
    tr = np.asarray(tr)
    v = np.asarray(v)
    rh = np.asarray(rh)
    met = np.asarray(met)
    clo = np.asarray(clo)
    wme = np.asarray(wme)

    # Validate inputs using the SetTmpInputs class
    SETInputs(
        tdb=tdb,
        tr=tr,
        v=v,
        rh=rh,
        met=met,
        clo=clo,
        wme=wme,
        body_surface_area=body_surface_area,
        p_atm=p_atm,
        position=position,
        limit_inputs=limit_inputs,
    )

    set_array = two_nodes_gagge(
        tdb=tdb,
        tr=tr,
        v=v,
        rh=rh,
        met=met,
        clo=clo,
        wme=wme,
        body_surface_area=body_surface_area,
        p_atm=p_atm,
        position=position,
        calculate_ce=calculate_ce,
        round_output=False,
    ).set

    if limit_inputs:
        (
            tdb_valid,
            tr_valid,
            v_valid,
            met_valid,
            clo_valid,
        ) = _check_ashrae55_compliance(
            tdb=tdb,
            tr=tr,
            v=v,
            met=met,
            clo=clo,
        )
        all_valid = ~(
            np.isnan(tdb_valid)
            | np.isnan(tr_valid)
            | np.isnan(v_valid)
            | np.isnan(met_valid)
            | np.isnan(clo_valid)
        )
        set_array = np.where(all_valid, set_array, np.nan)

    if round_output:
        set_array = np.around(set_array, 1)

    return SET(set=set_array)
