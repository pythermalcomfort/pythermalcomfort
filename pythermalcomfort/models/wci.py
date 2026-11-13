from __future__ import annotations

import numpy as np

from pythermalcomfort.classes_input import NumericInput, WCIInputs
from pythermalcomfort.classes_return import WCI


def wci(
    tdb: NumericInput,
    v: NumericInput,
    round_output: bool = True,
) -> WCI:
    """Calculate the Wind Chill Index (WCI) in accordance with the ASHRAE 2017 Handbook Fundamentals - Chapter 9 [ashrae2017]_.

    The wind chill index (WCI) is an empirical index based on cooling measurements
    taken on a cylindrical flask partially filled with water in Antarctica
    (Siple and Passel 1945). For a surface temperature of 33°C, the index describes
    the rate of heat loss from the cylinder via radiation and convection as a function
    of ambient temperature and wind velocity.

    This formulation has been met with some valid criticism. WCI is unlikely to be an
    accurate measure of heat loss from exposed flesh, which differs from plastic in terms
    of curvature, roughness, and radiation exchange qualities, and is always below 33°C
    in a cold environment. Furthermore, the equation's values peak at 90 km/h and then
    decline as velocity increases. Nonetheless, this score reliably represents the
    combined effects of temperature and wind on subjective discomfort for velocities
    below 80 km/h [ashrae2017]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    v : float or list of floats
        Wind speed 10m above ground level, [m/s].
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    WCI
        A dataclass containing the Wind Chill Index. See :py:class:`~pythermalcomfort.classes_return.WCI` for more details.
        To access the `wci` value, use the `wci` attribute of the returned `WCI` instance, e.g., `result.wci`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import wci

        result = wci(tdb=-5, v=5.5)
        print(result.wci)  # 1255.2

        result = wci(tdb=[-5, -10], v=[5.5, 10], round_output=True)
        print(result.wci)  # [1255.2 1603.9]

    """
    # Validate inputs using the WCIInputs class
    WCIInputs(
        tdb=tdb,
        v=v,
        round_output=round_output,
    )

    tdb = np.asarray(tdb)
    v = np.asarray(v)

    _wci = (10.45 + 10 * v**0.5 - v) * (33 - tdb)

    # the factor 1.163 is used to convert to W/m^2
    _wci = _wci * 1.163

    if round_output:
        _wci = np.around(_wci, 1)

    return WCI(wci=_wci)
