from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pythermalcomfort._internal.validation import _valid_range
from pythermalcomfort.utilities import NumericInput, c_to_k, cp_air, g


def mean_radiant_tmp(
    tg: NumericInput,
    tdb: NumericInput,
    v: NumericInput,
    d: NumericInput = 0.15,
    emissivity: NumericInput = 0.95,
    standard="Mixed Convection",
) -> NDArray[np.float64]:
    """Convert the globe temperature reading into mean radiant temperature in accordance
    with either the Mixed Convection developed by Teitelbaum E. et al. (2022) or the ISO
    7726:1998 Standard [7726ISO1998]_.

    Parameters
    ----------
    tg : float or list of floats
        globe temperature, [°C]
    tdb : float or list of floats
        air temperature, [°C]
    v : float or list of floats
        air speed, [m/s]
    d : float or list of floats
        diameter of the globe, [m] default 0.15 m
    emissivity : float or list of floats
        emissivity of the globe temperature sensor, default 0.95
    standard : str, optional
        Supported values are 'Mixed Convection' and 'ISO'. Defaults to 'Mixed Convection'.
        either choose between the Mixed Convection and ISO formulations.
        The Mixed Convection formulation has been proposed by Teitelbaum E. et al. (2022)
        to better determine the free and forced convection coefficient used in the
        calculation of the mean radiant temperature. They also showed that mean radiant
        temperature measured with ping-pong ball-sized globe thermometers is not reliable
        due to a stochastic convective bias [Teitelbaum2022]_. The Mixed Convection model has only
        been validated for globe sensors with a diameter between 0.04 and 0.15 m.

    Returns
    -------
    tr: float or list of floats
        mean radiant temperature, [°C]

    Raises
    ------
    ValueError
        If the standard is not recognized. Please choose either 'Mixed Convection' or 'ISO'.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.environment import mean_radiant_tmp

        tg = 53.2  # globe temperature in °C
        tdb = 30.0  # dry bulb temperature in °C
        v = 0.3  # air speed in m/s
        d = 0.1  # diameter of the globe in m
        tr = mean_radiant_tmp(tg, tdb, v, d, standard="ISO")
        print(tr)  # 74.8
    """
    standard = standard.lower()

    tdb = np.asarray(tdb)
    tg = np.asarray(tg)
    v = np.asarray(v)
    d = np.asarray(d)

    if standard == "mixed convection":
        mu = 0.0000181  # Pa s
        k_air = 0.02662  # W/m-K
        beta = 0.0034  # 1/K
        nu = 0.0000148  # m2/s
        alpha = 0.00002591  # m2/s
        pr = cp_air * mu / k_air  # Prandtl constants

        o = 0.0000000567
        n = 1.27 * d + 0.57

        ra = g * beta * np.absolute(tg - tdb) * d * d * d / nu / alpha
        re = v * d / nu

        nu_natural = 2 + (0.589 * np.power(ra, (1 / 4))) / (
            np.power(1 + np.power(0.469 / pr, 9 / 16), (4 / 9))
        )
        nu_forced = 2 + (
            0.4 * np.power(re, 0.5) + 0.06 * np.power(re, 2 / 3)
        ) * np.power(pr, 0.4)

        tr = (
            np.power(
                np.power(tg + 273.15, 4)
                - (
                    (
                        (
                            (
                                np.power(
                                    (np.power(nu_forced, n) + np.power(nu_natural, n)),
                                    1 / n,
                                )
                            )
                            * k_air
                            / d
                        )
                        * (-tg + tdb)
                    )
                    / emissivity
                    / o
                ),
                0.25,
            )
            - 273.15
        )

        d_valid = _valid_range(d, (0.04, 0.15))
        return np.where(~np.isnan(d_valid), tr, np.nan)

    if standard == "iso":  # pragma: no branch
        tg = np.add(tg, c_to_k)
        tdb = np.add(tdb, c_to_k)

        # calculate heat transfer coefficient
        h_n = 1.4 * np.power(np.abs(tg - tdb) / d, 0.25)  # natural convection
        h_f = 6.3 * np.power(v, 0.6) / np.power(d, 0.4)  # forced convection

        # get the biggest between the two coefficients
        h = np.maximum(h_f, h_n)

        return (
            np.power(
                np.power(tg, 4) + h * (tg - tdb) / (emissivity * (5.67 * 10**-8)),
                0.25,
            )
            - c_to_k
        )

    raise ValueError(
        "Standard not recognized. Please choose either 'Mixed Convection' or 'ISO'.",
    )
