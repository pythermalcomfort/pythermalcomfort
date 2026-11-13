from __future__ import annotations

import numpy as np

from pythermalcomfort.classes_input import NumericInput, WBGTInputs
from pythermalcomfort.classes_return import WBGT


def wbgt(
    twb: NumericInput,
    tg: NumericInput,
    tdb: NumericInput | None = None,
    with_solar_load: bool = False,
    round_output: bool = True,
) -> WBGT:
    """Calculate the Wet Bulb Globe Temperature (WBGT) index in compliance with the ISO
    7243 Standard [7243ISO2017]_.

    The WBGT is a heat stress index that
    measures the thermal environment to which a person is exposed. In most
    situations, this index is simple to calculate. It should be used as a
    screening tool to determine whether heat stress is present. The PHS model
    allows a more accurate estimation of stress. PHS can be calculated using
    the function :py:meth:`pythermalcomfort.models.phs`.

    The WBGT determines the impact of heat on a person throughout the course of a working
    day (up to 8 hours). It does not apply to very brief heat exposures. It pertains to
    the evaluation of male and female people who are fit for work in both indoor and
    outdoor occupational environments, as well as other sorts of surroundings [7243ISO2017]_.

    The WBGT is defined as a function of only twb and tg if the person is not exposed to
    direct radiant heat from the sun. When a person is exposed to direct radiant heat,
    tdb must also be specified.

    Parameters
    ----------
    twb : float or list of floats
        Natural (no forced air flow) wet bulb temperature, [°C].
    tg : float or list of floats
        Globe temperature, [°C].
    tdb : float or list of floats, optional
        Dry bulb air temperature, [°C]. This value is needed as input if the person is
        exposed to direct solar radiation.
    with_solar_load : bool, optional
        True if the globe sensor is exposed to direct solar radiation. Defaults to False.
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    WBGT
        A dataclass containing the Wet Bulb Globe Temperature Index. See
        :py:class:`~pythermalcomfort.classes_return.WBGT` for more details. To access the
        `wbgt` value, use the `wbgt` attribute of the returned `WBGT` instance, e.g.,
        `result.wbgt`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import wbgt

        result = wbgt(twb=25, tg=32)
        print(result.wbgt)  # 27.1

        result = wbgt(twb=25, tg=32, tdb=20, with_solar_load=True)
        print(result.wbgt)  # 25.9
    """
    # Validate inputs using the WBGTInputs class
    WBGTInputs(
        twb=twb,
        tg=tg,
        tdb=tdb,
        with_solar_load=with_solar_load,
        round_output=round_output,
    )

    twb = np.asarray(twb)
    tg = np.asarray(tg)
    tdb = np.asarray(tdb) if tdb is not None else None

    if with_solar_load and tdb is None:
        raise ValueError("Please enter the dry bulb air temperature")

    if with_solar_load:
        t_wbg = 0.7 * twb + 0.2 * tg + 0.1 * tdb
    else:
        t_wbg = 0.7 * twb + 0.3 * tg

    if round_output:
        t_wbg = np.round(t_wbg, 1)

    return WBGT(wbgt=t_wbg)
