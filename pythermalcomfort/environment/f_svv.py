from __future__ import annotations

import math


def f_svv(w: float, h: float, d: float) -> float:
    """Calculate the sky-vault view fraction.

    Parameters
    ----------
    w : float
        width of the window, [m]
    h : float
        height of the window, [m]
    d : float
        distance between the occupant and the window, [m]

    Returns
    -------
    f_svv  : float
        sky-vault view fraction ranges between 0 and 1
    """
    return (
        math.degrees(math.atan(h / (2 * d)))
        * math.degrees(math.atan(w / (2 * d)))
        / 16200
    )
