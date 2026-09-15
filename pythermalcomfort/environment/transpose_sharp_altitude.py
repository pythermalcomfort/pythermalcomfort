from __future__ import annotations

import math

from numba import njit


@njit(cache=True)
def transpose_sharp_altitude(sharp: float, altitude: float) -> tuple[float, float]:
    """Transpose the solar altitude and solar azimuth angles."""
    altitude_new = math.degrees(
        math.asin(
            math.sin(math.radians(abs(sharp - 90))) * math.cos(math.radians(altitude)),
        ),
    )
    sharp = math.degrees(
        math.atan(
            math.sin(math.radians(sharp)) * math.tan(math.radians(90 - altitude)),
        ),
    )
    sol_altitude = altitude_new
    return round(sharp, 3), round(sol_altitude, 3)
