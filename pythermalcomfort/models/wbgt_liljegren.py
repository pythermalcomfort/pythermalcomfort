from __future__ import annotations

import logging

import numpy as np

from pythermalcomfort.classes_input import NumericInput, WBGTLiljegrenInputs
from pythermalcomfort.classes_return import WBGTLiljegren

_logger = logging.getLogger(__name__)


def wbgt_liljegren(
    tdb: NumericInput,
    rh: NumericInput,
    v: NumericInput,
    sol_radiation_global: NumericInput,
    *,
    latitude: NumericInput,
    longitude: NumericInput,
    year: NumericInput,
    month: NumericInput,
    day: NumericInput,
    hour: NumericInput,
    minute: NumericInput | None = None,
    p_atm: NumericInput | None = None,
    gmt_offset_hours: NumericInput | None = None,
    averaging_minutes: NumericInput | None = None,
    wind_height: NumericInput | None = None,
    urban: NumericInput | None = None,
    vertical_temperature_difference: NumericInput | None = None,
    round_output: bool = True,
) -> WBGTLiljegren:
    """Estimate outdoor WBGT from weather using Liljegren's model [Liljegren2008]_.

    Requires the optional ``lwbgt`` backend [Zheng2026LWBGT]_, installed with
    ``pip install 'pythermalcomfort[lwbgt]'``. Use :func:`~pythermalcomfort.models.wbgt.wbgt`
    when natural wet bulb and globe temperatures have already been measured.
    All numeric inputs support NumPy broadcasting.
    Omitted optional weather inputs (or ``None``) use the defaults below and
    generate one INFO logging message per call with evaluated records.

    Parameters
    ----------
    tdb : float or array-like
        Dry bulb air temperature, [°C], greater than -273.15.
    rh : float or array-like
        Relative humidity, [%], between 0 and 100. Zero humidity can cause the
        native wet bulb solver to fail because it starts from the dew point.
    v : float or array-like
        Nonnegative wind speed measured at ``wind_height``, [m/s].
    sol_radiation_global : float or array-like
        Nonnegative global solar irradiance, [W/m²].
    latitude : float or array-like
        Latitude, [degrees north], between -90 and 90.
    longitude : float or array-like
        Longitude, [degrees east], between -180 and 180.
    year : int or array-like
        Gregorian year, 1950 through 2049, as supported by the native solar routine.
    month : int or array-like
        Calendar month, 1 through 12.
    day : int or array-like
        Day of the month. The year, month and day must form a valid date.
    hour : int or array-like
        Hour, 0 through 23, in local standard time at the end of the averaging
        interval. Daylight saving time must be removed by the caller.
    minute : int or array-like, optional
        Minute, 0 through 59. Defaults to 0.
    p_atm : float or array-like, optional
        Positive station pressure at the observation site's elevation, [Pa],
        not pressure corrected to sea level. Defaults to 101325 (1013.25 hPa),
        the standard sea-level reference. Converted to hPa for the backend.
    gmt_offset_hours : int or array-like, optional
        Local standard time minus UTC, [whole hours], between -12 and 14.
        Defaults to 0 (UTC). For fractional-offset time zones, supply UTC dates
        and times with offset 0; offsets are never truncated.
    averaging_minutes : int or array-like, optional
        Input averaging interval, [whole minutes], between 0 and 1440. Defaults
        to 0 (instantaneous). Solar geometry is evaluated half an interval before
        the supplied timestamp. If the shifted day falls outside the native
        calendar window, supply a UTC timestamp or shorten the interval.
    wind_height : float or array-like, optional
        Positive wind measurement height, [m]. Defaults to 2. For other heights,
        ``urban`` is required.
    urban : int or array-like, optional
        Wind scaling setting: 0 for rural, 1 for urban. Ignored by the backend at
        2 m, where an omitted value is replaced with 0.
    vertical_temperature_difference : float or array-like, optional
        Air temperature at wind measurement height minus air temperature at 2 m,
        [°C], used to classify nighttime stability for wind scaling. This is a
        temperature difference, not a gradient per metre. Defaults to -0.052°C
        when wind height differs from 2 m. The backend uses its sign at night
        (zero follows the nonnegative branch); daytime stability uses solar
        radiation and wind instead. Ignored and unlogged at 2 m, where an
        omitted value is replaced with 0.
    round_output : bool, optional
        Round temperatures to one decimal place. Defaults to True. Wind and
        status are not rounded.

    Returns
    -------
    WBGTLiljegren
        A dataclass containing ``wbgt``, ``tg``, ``twb``, ``tpsy``, ``v_2m`` and
        ``status``. See :py:class:`~pythermalcomfort.classes_return.WBGTLiljegren`.
        Scalar inputs produce scalar fields; arrays retain the broadcast shape.

    Raises
    ------
    ImportError
        If the optional lwbgt package is not installed.
    TypeError
        If inputs have invalid types or contain nonnumeric elements.
    ValueError
        If inputs violate the documented bounds, contain infinity, cannot be
        represented as finite float32 values, or cannot be broadcast together.
    RuntimeError
        If the backend cannot load its native library or rejects the batch call.

    Notes
    -----
    The backend estimates natural wet bulb and globe temperatures and computes
    outdoor WBGT as 0.7 * twb + 0.2 * tg + 0.1 * tdb. Its fixed globe diameter is
    50.8 mm. Native solar adjustments, minimum convective wind speed and float32
    numerical behavior are preserved. Returned ``v_2m`` is the supplied or
    height-adjusted wind before the convective minimum is applied.

    Records containing NaN in any input are skipped and all their outputs,
    including status, are NaN. Native -9999 sentinels and nonfinite outputs are
    converted to NaN; valid partial results are retained. Check ``status`` for
    each record: 0 indicates success, native nonzero failure codes are retained,
    and -2 indicates invalid outputs despite native success (or invalid native
    status). Solver failures emit neither Python warnings nor failure logs.

    The ``pythermalcomfort.models.wbgt_liljegren`` logger reports all assumed
    input defaults at INFO level, including minute and averaging interval.
    Explicit values are not reported as defaults. The wind settings ``urban``
    and ``vertical_temperature_difference`` are unused and unlogged at 2 m.
    Other wind heights require explicit ``urban`` and default the temperature
    difference to -0.052°C when omitted. Evaluated records at those heights
    generate an INFO message for wind-height adjustment, including the temperature
    difference if defaulted. All notices are combined into one message per
    call. Enable these messages with ``logging.basicConfig(level=logging.INFO)``
    in the calling application. The library does not configure logging handlers.
    Output rounding is a formatting option and is not logged as an input
    assumption.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import wbgt_liljegren

        site = dict(
            latitude=1.3521,
            longitude=103.8198,
            year=2024,
            month=4,
            day=15,
            minute=30,
            gmt_offset_hours=8,
            averaging_minutes=60,
            wind_height=10,
            urban=1,
        )
        result = wbgt_liljegren(
            32.1,
            68,
            2.8,
            742,
            hour=14,
            p_atm=100840,
            vertical_temperature_difference=-0.4,
            **site,
        )
        print(result.wbgt)  # 32.5

        result = wbgt_liljegren(
            [32.1, 27],
            [68, 88],
            [2.8, 1.1],
            [742, 0],
            hour=[14, 2],
            p_atm=[100840, 100970],
            vertical_temperature_difference=[-0.4, 0.2],
            **site,
        )
        print(result.wbgt)  # [32.5 25.7]
    """
    optional = {
        "minute": (minute, 0, "min"),
        "p_atm": (p_atm, 101325, "Pa (1013.25 hPa)"),
        "gmt_offset_hours": (gmt_offset_hours, 0, "h (UTC)"),
        "averaging_minutes": (averaging_minutes, 0, "min (instantaneous)"),
        "wind_height": (wind_height, 2, "m"),
    }
    resolved = {
        name: default if value is None else value
        for name, (value, default, _) in optional.items()
    }
    assumed = {
        name: f"{name}={default} {unit}"
        for name, (value, default, unit) in optional.items()
        if value is None
    }
    inputs = WBGTLiljegrenInputs(
        tdb=tdb,
        rh=rh,
        v=v,
        sol_radiation_global=sol_radiation_global,
        latitude=latitude,
        longitude=longitude,
        year=year,
        month=month,
        day=day,
        hour=hour,
        **resolved,
        urban=urban,
        vertical_temperature_difference=vertical_temperature_difference,
        round_output=round_output,
    )
    try:
        from lwbgt import Input, calculate_batch
    except ModuleNotFoundError as exc:
        if exc.name != "lwbgt":
            raise
        raise ImportError(
            "wbgt_liljegren requires lwbgt; install it with "
            "`pip install 'pythermalcomfort[lwbgt]'`."
        ) from exc

    arrays = inputs._arrays
    shape = arrays["tdb"].shape
    valid = np.ones(shape, dtype=bool)
    for array in arrays.values():
        valid &= np.isfinite(array)
    values = {name: array[valid] for name, array in arrays.items()}
    count = int(valid.sum())
    adjusted = int(np.count_nonzero(values["wind_height"] != 2))
    if adjusted and vertical_temperature_difference is None:
        assumed["vertical_temperature_difference"] = (
            "vertical_temperature_difference=-0.052 °C (nighttime wind adjustment)"
        )
    notices = []
    if assumed:
        notices.append("assumed defaults: " + "; ".join(assumed.values()))
    if adjusted:
        notices.append(
            "wind-height adjustment uses urban (and vertical_temperature_difference "
            "at night) "
            f"for {adjusted} of {count} evaluated records (wind_height != 2 m)"
        )
    if count and notices:
        _logger.info("wbgt_liljegren %s", "; ".join(notices))
    records = (
        Input(
            year=int(values["year"][i]),
            month=int(values["month"][i]),
            day=int(values["day"][i]),
            hour=int(values["hour"][i]),
            minute=int(values["minute"][i]),
            gmt_offset_hours=int(values["gmt_offset_hours"][i]),
            averaging_minutes=int(values["averaging_minutes"][i]),
            urban=int(values["urban"][i]),
            latitude_deg_north=float(values["latitude"][i]),
            longitude_deg_east=float(values["longitude"][i]),
            solar_w_m2=float(values["sol_radiation_global"][i]),
            pressure_hpa=float(values["p_atm"][i] / 100),
            air_temperature_c=float(values["tdb"][i]),
            relative_humidity_percent=float(values["rh"][i]),
            wind_speed_m_s=float(values["v"][i]),
            wind_height_m=float(values["wind_height"][i]),
            vertical_temperature_difference_c=float(
                values["vertical_temperature_difference"][i]
            ),
        )
        for i in range(count)
    )
    results = calculate_batch(records) if count else []
    fields = {
        "wbgt": "wbgt_c",
        "tg": "globe_temperature_c",
        "twb": "natural_wet_bulb_c",
        "tpsy": "psychrometric_wet_bulb_c",
        "v_2m": "estimated_wind_speed_m_s",
    }
    status = np.array([result.status for result in results], dtype=float)
    native_failed = status != 0
    status = np.where(np.isfinite(status), status, -2)
    outputs = {}
    for name, native_name in fields.items():
        data = np.array(
            [getattr(result, native_name) for result in results], dtype=float
        )
        unavailable = (data == -9999) | ~np.isfinite(data)
        status[(status == 0) & unavailable] = -2
        if name == "wbgt":
            unavailable |= native_failed
        data = np.where(unavailable, np.nan, data)
        if round_output and name != "v_2m":
            data = np.round(data, 1)
        output = np.full(shape, np.nan)
        output[valid] = data
        outputs[name] = output.item() if output.ndim == 0 else output

    output = np.full(shape, np.nan)
    output[valid] = status
    outputs["status"] = output.item() if output.ndim == 0 else output
    return WBGTLiljegren(**outputs)
