from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numba import float64, vectorize
from numpy.typing import NDArray

from pythermalcomfort._internal.validation import (
    HEAT_INDEX_STRESS_CATEGORIES,
    _mapping,
    _valid_range,
)
from pythermalcomfort.classes_input import HIInputs, NumericInput
from pythermalcomfort.classes_return import HI


def heat_index_rothfusz(
    tdb: NumericInput,
    rh: NumericInput,
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

    hi = _rothfusz_heat_index_optimized(tdb, rh)

    # heat index should only be calculated for temperatures above 27 °C
    if limit_inputs:
        tdb_valid = _valid_range(tdb, (27.0, np.inf))
        hi_valid = np.where(~np.isnan(tdb_valid), hi, np.nan)
    else:
        hi_valid = hi

    heat_index_categories = {27.0: "no risk", **HEAT_INDEX_STRESS_CATEGORIES}

    if round_output:
        hi_valid = np.around(hi_valid, 1)

    return HI(hi=hi_valid, stress_category=_mapping(hi_valid, heat_index_categories))


@cast(
    Callable[
        [Callable[[float, float], float]],
        Callable[
            [
                float | NDArray[np.float64],
                float | NDArray[np.float64],
            ],
            np.float64 | NDArray[np.float64],
        ],
    ],
    vectorize(
        [
            float64(float64, float64),
        ],
        cache=True,
    ),
)
def _rothfusz_heat_index_optimized(tdb: float, rh: float) -> float:
    return (
        -8.784695
        + 1.61139411 * tdb
        + 2.338549 * rh
        - 0.14611605 * tdb * rh
        - 1.2308094e-2 * tdb**2
        - 1.6424828e-2 * rh**2
        + 2.211732e-3 * tdb**2 * rh
        + 7.2546e-4 * tdb * rh**2
        - 3.582e-6 * tdb**2 * rh**2
    )
