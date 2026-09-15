from __future__ import annotations

import numpy as np

from pythermalcomfort.classes_input import ATInputs, NumericInput
from pythermalcomfort.classes_return import AT
from pythermalcomfort.psychrometrics import psy_ta_rh


def at(
    tdb: NumericInput,
    rh: NumericInput,
    v: NumericInput,
    q: NumericInput | None = None,
    round_output: bool = True,
) -> AT:
    """Calculate the Apparent Temperature (AT).

    The AT is defined as the temperature at the reference humidity level producing the same amount of discomfort as that
    experienced under the current ambient temperature, humidity, and solar radiation
    [Steadman1984]_. In other words, the AT is an adjustment to the dry bulb temperature
    based on the relative humidity value. Absolute humidity with a dew point of 14°C is
    chosen as a reference.

    It includes the chilling effect of the wind at lower temperatures. [Blazejczyk2012]_

    .. note::
        Two formulas for AT are in use by the Australian Bureau of Meteorology: one includes
        solar radiation and the other one does not (http://www.bom.gov.au/info/thermal_stress/
        , 29 Sep 2021). Please specify q if you want to estimate AT with solar load.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C]
    rh : float or list of floats
        Relative humidity, [%]
    v : float or list of floats
        Wind speed 10m above ground level, [m/s]
    q : float or list of floats, optional
        Net radiation absorbed per unit area of body surface [W/m2]
    round_output : bool, default True
        If True, rounds the output value; if False, does not round it.

    Returns
    -------
    AT
        Dataclass containing the apparent temperature, [°C]. See :py:class:`~pythermalcomfort.classes_return.AT` for more details.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import at

        at(tdb=25, rh=30, v=0.1)
        # AT(at=24.1)
    """
    # Validate inputs
    ATInputs(tdb=tdb, rh=rh, v=v, q=q, round_output=round_output)

    # Convert lists to numpy arrays if necessary
    tdb = np.asarray(tdb)
    rh = np.asarray(rh)
    v = np.asarray(v)

    # Calculate vapor pressure
    p_vap = psy_ta_rh(tdb, rh).p_vap / 100

    # Calculate apparent temperature
    if q is not None:
        q = np.asarray(q)
        t_at = tdb + 0.348 * p_vap - 0.7 * v + 0.7 * q / (v + 10) - 4.25
    else:
        t_at = tdb + 0.33 * p_vap - 0.7 * v - 4.00

    if round_output:
        t_at = np.around(t_at, 1)

    return AT(at=t_at)
