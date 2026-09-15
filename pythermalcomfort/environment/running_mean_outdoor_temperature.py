from __future__ import annotations

from pythermalcomfort.utilities import Units, units_converter


def running_mean_outdoor_temperature(
    temp_array: list[float],
    alpha: float = 0.8,
    units: str = Units.SI.value,
) -> float:
    """Estimate the running mean temperature also known as prevailing mean outdoor
    temperature.

    Parameters
    ----------
    temp_array: list
        array containing the mean daily temperature in descending order (i.e. from
        newest/yesterday to oldest) :math:`[t_{day-1}, t_{day-2}, ... ,
        t_{day-n}]`.
        Where :math:`t_{day-1}` is yesterday's daily mean temperature. The EN
        16798-1 2019 [16798EN2019]_ states that n should be equal to 7
    alpha : float
        constant between 0 and 1. The EN 16798-1 2019 [16798EN2019]_ recommends a value of 0.8,
        while the ASHRAE 55 2020 recommends to choose values between 0.9 and 0.6,
        corresponding to a slow- and fast- response running mean, respectively.
        Adaptive comfort theory suggests that a slow-response running mean (alpha =
        0.9) could be more appropriate for climates in which synoptic-scale (day-to-
        day) temperature dynamics are relatively minor, such as the humid tropics.
    units: str default="SI"
        select the SI (International System of Units) or the IP (Imperial Units) system.

    Returns
    -------
    t_rm  : float
        running mean outdoor temperature

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.environment import running_mean_outdoor_temperature

        temp_array = [
            22.0,
            20.5,
            19.0,
            21.0,
            18.5,
            17.0,
            16.5,
        ]  # daily mean temperatures
        t_rm = running_mean_outdoor_temperature(temp_array, alpha=0.8, units="SI")

        # Calculate the mean radiant temperature using the previous 7 days from an array of daily mean temperatures
        temp_array = [
            22.0,
            20.5,
            19.0,
            21.0,
            18.5,
            17.0,
            16.5,
            15.0,
            14.5,
        ]  # daily mean temperatures
        days_to_consider = 7
        results = []
        for i in range(len(temp_array) - days_to_consider + 1):
            subset = temp_array[i : i + days_to_consider]
            print(f"Subset for days {i + 1} to {i + days_to_consider}: {subset}")
            t_rm = running_mean_outdoor_temperature(subset, alpha=0.8, units="SI")
            print(f"Days {i + 1} to {i + days_to_consider}: t_rm = {t_rm}")
            results.append(t_rm)

    Calculate the running mean outdoor temperature from hourly data:

    .. code-block:: python

        import pandas as pd
        import numpy as np
        from pythermalcomfort.environment import running_mean_outdoor_temperature

        # Step 1: Create a DataFrame with hourly dry bulb temperature (tdb) data
        # for 10 days starting from January 1, 2025. We simulate hourly data with
        # a base temperature of 20°C plus sinusoidal daily variation and some noise.
        start_date = pd.Timestamp("2025-01-01")
        hourly_index = pd.date_range(start=start_date, periods=10 * 24, freq="h")
        # Simulate tdb with daily cycle: 20°C base + 10°C amplitude sinusoidal + noise
        t_out_hourly = (
            20
            + 10 * np.sin(2 * np.pi * (hourly_index.hour / 24))
            + np.random.normal(0, 2, len(hourly_index))
        )
        df = pd.DataFrame({"t_out": t_out_hourly}, index=hourly_index)

        # Step 2: Resample the hourly data to daily mean temperatures
        # This gives us the average temperature for each day.
        df_daily = df.resample("D").mean()

        # Step 3: Calculate the running mean outdoor temperature for each day
        # using the previous 7 days' daily means. The function requires the array
        # in descending order (newest to oldest). For the first 6 days, there is
        # insufficient data (less than 7 days), so they remain NaN.
        df_daily["running_mean"] = np.nan
        for i in range(7, len(df_daily)):
            # Get the previous 7 days' means (from i-7 to i-1), reverse to newest first
            prev_7_days = df_daily["t_out"].iloc[i - 7 : i].values[::-1]
            # Calculate the running mean and assign to the current day
            df_daily.loc[df_daily.index[i], "running_mean"] = (
                running_mean_outdoor_temperature(
                    prev_7_days.tolist(), alpha=0.8, units="SI"
                )
            )

        df["date"] = df.index.date
        df_daily["date"] = df_daily.index.date
        df.reset_index().merge(
            df_daily[["date", "running_mean"]], on="date", how="left"
        ).set_index("index").drop(columns=["date"])
    """
    units = units.upper()
    if units == Units.IP.value:
        temp_array = [units_converter(tdb=temp)[0] for temp in temp_array]

    coeff = [alpha**ix for ix, x in enumerate(temp_array)]
    t_rm = sum([a * b for a, b in zip(coeff, temp_array, strict=False)]) / sum(coeff)

    if units == Units.IP.value:
        t_rm = units_converter(tmp=t_rm, from_units=Units.SI.value.lower())[0]

    return round(t_rm, 1)
