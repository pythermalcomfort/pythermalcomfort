from __future__ import annotations

import numpy as np

from pythermalcomfort._internal.bfu import _rest_fan_state
from pythermalcomfort.classes_input import BFURestInputs, NumericInput
from pythermalcomfort.classes_return import BFURest
from pythermalcomfort.utilities import BFURestPopulation


def bfu_rest(
    tdb: NumericInput,
    rh: NumericInput,
    population: str | BFURestPopulation = BFURestPopulation.young.value,
    round_output: bool = True,
) -> BFURest:
    """Calculate the change in heat storage from fan use at rest.

    Implement the resting biophysical model of Morris et al. (2021)
    [Morris2021]_. The model uses fixed resting metabolic and physiological
    parameters for healthy young adults, healthy older adults, and older adults
    taking anticholinergic medication.
    It compares fan air speed of 3.5 m/s with still-air speed of 0.2 m/s.
    The metabolic rate is fixed at 65 W/m²; changing it would define a different
    model. The population groups select the source model's fixed sweat
    production and wettedness limits. The output compares modelled heat storage;
    it does not determine an individual's safe-use temperature limit.
    Negative ``delta_storage`` indicates reduced heat storage with the fan.
    Use ``bfu_occupational`` for the separate occupational model with variable
    metabolic rate, clothing, and air speed.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, :math:`°C`.
    rh : float or list of floats
        Relative humidity, :math:`%`.
    population : str or BFURestPopulation, optional
        Population group as defined by Morris et al. (2021) [Morris2021]_.
        ``"YNG"``: healthy young adults aged 18-40 years.
        ``"OLD"``: healthy older adults aged 65 years or older.
        ``"MEDS"``: older adults taking anticholinergic medication.
        Also accepts :py:class:`~pythermalcomfort.utilities.BFURestPopulation`
        members. Defaults to ``"YNG"``.
    round_output : bool, optional
        If ``True`` the returned values are rounded to one decimal place. Defaults to ``True``.

    Returns
    -------
    BFURest
        A dataclass containing fan-minus-no-fan heat storage and evaporative requirements between fans and still air. See :py:class:`~pythermalcomfort.classes_return.BFURest` for details.

    Raises
    ------
    ValueError
        If humidity is outside 0-100 %, the population is unknown, or input arrays
        cannot be broadcast together.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import bfu_rest

        result = bfu_rest(tdb=36, rh=60, population="YNG")
        print(result.delta_storage)

    References
    ----------
    See [Morris2021]_ for the model and its physiological assumptions.
    """

    inputs = BFURestInputs(
        tdb=tdb,
        rh=rh,
        population=population,
        round_output=round_output,
    )
    tdb_arr, rh_arr = np.broadcast_arrays(
        np.asarray(inputs.tdb, dtype=float),
        np.asarray(inputs.rh, dtype=float),
    )
    e_req_fan, e_max_fan = _rest_fan_state(
        tdb=tdb_arr,
        rh=rh_arr,
        population=inputs.population,
        fan_on=True,
    )
    e_req_no_fan, e_max_no_fan = _rest_fan_state(
        tdb=tdb_arr,
        rh=rh_arr,
        population=inputs.population,
        fan_on=False,
    )
    # Fan minus no-fan storage: negative values indicate reduced heat storage.
    delta_storage = (e_req_fan - e_max_fan) - (e_req_no_fan - e_max_no_fan)
    values = dict(
        delta_storage=delta_storage,
        e_req_fan=e_req_fan,
        e_max_fan=e_max_fan,
        e_req_no_fan=e_req_no_fan,
        e_max_no_fan=e_max_no_fan,
    )
    if round_output:
        values = {name: np.around(value, 1) for name, value in values.items()}
    return BFURest(**values)
