from __future__ import annotations

import numpy as np

from pythermalcomfort._internal.validation import _mapping
from pythermalcomfort.classes_input import DIInputs, NumericInput
from pythermalcomfort.classes_return import DI


def discomfort_index(
    tdb: NumericInput,
    rh: NumericInput,
) -> DI:
    """Calculate the Discomfort Index (DI).

    The index is essentially an
    effective temperature based on air temperature and humidity. The discomfort
    index is usually divided into 6 discomfort categories and it only applies to
    warm environments: [Polydoros2015]_

    * class 1 - DI < 21 °C - No discomfort
    * class 2 - 21 <= DI < 24 °C - Less than 50% feels discomfort
    * class 3 - 24 <= DI < 27 °C - More than 50% feels discomfort
    * class 4 - 27 <= DI < 29 °C - Most of the population feels discomfort
    * class 5 - 29 <= DI < 32 °C - Everyone feels severe stress
    * class 6 - DI >= 32 °C - State of medical emergency

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    rh : float or list of floats
        Relative humidity, [%].

    Returns
    -------
    DI
        A dataclass containing the Discomfort Index and its classification. See :py:class:`~pythermalcomfort.classes_return.DI` for more details.
        To access the `di` and `discomfort_condition` values, use the respective attributes of the returned `DI` instance, e.g., `result.di`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import discomfort_index

        result = discomfort_index(tdb=25, rh=50)
        print(result.di)  # 22.1
        print(result.discomfort_condition)  # Less than 50% feels discomfort

        result = discomfort_index(tdb=[25, 30], rh=[50, 60])
        print(result.di)  # [22.1, 27.3]
        print(
            result.discomfort_condition
        )  # ['Less than 50% feels discomfort', 'Most of the population feels discomfort']
    """
    # Validate inputs using the DiscomfortIndexInputs class
    DIInputs(
        tdb=tdb,
        rh=rh,
    )

    tdb = np.asarray(tdb)
    rh = np.asarray(rh)

    di = tdb - 0.55 * (1 - 0.01 * rh) * (tdb - 14.5)

    di_categories = {
        21: "No discomfort",
        24: "Less than 50% feels discomfort",
        27: "More than 50% feels discomfort",
        29: "Most of the population feels discomfort",
        32: "Everyone feels severe stress",
        99: "State of medical emergency",
    }

    return DI(
        di=np.around(di, 1),
        discomfort_condition=_mapping(di, di_categories, right=False),
    )
