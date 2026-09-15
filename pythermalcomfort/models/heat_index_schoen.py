from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numba import float64, vectorize
from numpy.typing import NDArray

from pythermalcomfort._internal.validation import HEAT_INDEX_STRESS_CATEGORIES, _mapping
from pythermalcomfort.classes_input import HIInputs, NumericInput
from pythermalcomfort.classes_return import HI
from pythermalcomfort.psychrometrics import psy_ta_rh


def heat_index_schoen(
    tdb: NumericInput,
    rh: NumericInput,
    round_output: bool = True,
) -> HI:
    """Calculate the Temperature Humidity Index (THI), also known as Heat Index Schoen,
    in accordance with the Schoen (2005) model [Schoen2005]_.

    The temperature-humidity index (THI) is a simplified scale of apparent
    temperature, considering only dry-bulb temperature and humidity. It is another
    formulation of the heat index.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    rh : float or list of floats
        Relative humidity, [%].
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    HI
        A dataclass containing the Heat Index. See :py:class:`~pythermalcomfort.classes_return.HI` for more details.
        To access the `hi` value, use the `hi` attribute of the returned `HI` instance, e.g., `result.hi`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import heat_index_schoen

        result = heat_index_schoen(tdb=29, rh=50)
        print(result.hi)  # 30.0
        print(result.stress_category)  # caution
    """
    # Validate inputs using the HeatIndexInputs class
    HIInputs(
        tdb=tdb,
        rh=rh,
        round_output=round_output,
        limit_inputs=True,
    )

    tdb = np.asarray(tdb)
    rh = np.asarray(rh)

    # Calculate dew point temperature
    t_dew = psy_ta_rh(tdb, rh, p_atm=101325).dew_point_tmp

    hi = _schoen_heat_index_optimized(tdb, t_dew)

    heat_index_categories = {27.0: "no risk", **HEAT_INDEX_STRESS_CATEGORIES}

    if round_output:
        hi = np.around(hi, 1)

    return HI(hi=hi, stress_category=_mapping(hi, heat_index_categories))


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
def _schoen_heat_index_optimized(tdb: float, t_dew: float) -> float:
    return tdb - 1.0799 * np.exp(0.03755 * tdb) * (1 - np.exp(0.0801 * (t_dew - 14)))
