from __future__ import annotations

import numpy as np

from pythermalcomfort.classes_input import HIInputs
from pythermalcomfort.classes_return import HI
from pythermalcomfort.shared_functions import mapping, valid_range


def heat_index_rothfusz(
    tdb: float | list[float],
    rh: float | list[float],
    round_output: bool = True,
    limit_inputs: bool = True,
) -> HI:
    """Calculate the Heat Index (HI) in accordance with the Rothfusz (1990) model
    [Rothfusz1990]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    rh : float or list of floats
        Relative humidity, [%].
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.
    limit_inputs : bool, optional
        If True, limits the inputs to the standard applicability limits. Defaults to True.

        .. note::
            By default, if the inputs are outside the standard applicability limits the
            function returns NaN. If False returns heat index
            values even if input values are outside the applicability limits of the model.

            The applicability limit is tdb >= 27°C.

    Returns
    -------
    HI
        A dataclass containing the Heat Index. See :py:class:`~pythermalcomfort.classes_return.HI` for more details.
        To access the `hi` value, use the `hi` attribute of the returned `HI` instance, e.g., `result.hi`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import heat_index_rothfusz

        result = heat_index_rothfusz(tdb=29, rh=50)
        print(result.hi)  # 29.7
        print(result.stress_category)  # "caution"
    """
    # Validate inputs using the HeatIndexInputs class
    HIInputs(
        tdb=tdb,
        rh=rh,
        round_output=round_output,
        limit_inputs=limit_inputs,
    )

    tdb = np.asarray(tdb)
    rh = np.asarray(rh)

    hi = -8.784695 + 1.61139411 * tdb + 2.338549 * rh - 0.14611605 * tdb * rh
    hi += -1.2308094 * 10**-2 * tdb**2 - 1.6424828 * 10**-2 * rh**2
    hi += 2.211732 * 10**-3 * tdb**2 * rh + 7.2546 * 10**-4 * tdb * rh**2
    hi += -3.582 * 10**-6 * tdb**2 * rh**2

    # heat index should only be calculated for temperatures above 27 °C
    if limit_inputs:
        tdb_valid = valid_range(tdb, (27.0, np.inf), "tdb")
        hi_valid = np.where(~np.isnan(tdb_valid), hi, np.nan)
    else:
        hi_valid = hi

    heat_index_categories = {
        27.0: "no risk",
        32.0: "caution",
        41.0: "extreme caution",
        54.0: "danger",
        1000.0: "extreme danger",
    }

    if round_output:
        hi_valid = np.around(hi_valid, 1)

    return HI(hi=hi_valid, stress_category=mapping(hi_valid, heat_index_categories))
