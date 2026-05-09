from __future__ import annotations

import numpy as np
from numba import float64, vectorize

from pythermalcomfort.classes_input import UTCIInputs
from pythermalcomfort.classes_return import UTCI
from pythermalcomfort.shared_functions import mapping, valid_range
from pythermalcomfort.utilities import Units, units_converter


def utci(
    tdb: float | list[float],
    tr: float | list[float],
    v: float | list[float],
    rh: float | list[float],
    units: str = Units.SI.value,
    limit_inputs: bool = True,
    round_output: bool = True,
) -> UTCI:
    """Calculate the Universal Thermal Climate Index (UTCI).

    The UTCI is the equivalent
    temperature for the environment derived from a reference environment. It is defined
    as the air temperature of the reference environment which produces the same strain
    index value in comparison with the reference individual's response to the real
    environment. It is regarded as one of the most comprehensive indices for calculating
    heat stress in outdoor spaces. The parameters that are taken into account for
    calculating UTCI involve dry bulb temperature, mean radiation temperature, the
    pressure of water vapor or relative humidity, and wind speed (at the elevation of 10
    m above the ground). [Zare2018]_

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature. Default in [°C] in [°F] if `units` = 'IP'.
    tr : float or list of floats
        Mean radiant temperature. Default in [°C] in [°F] if `units` = 'IP'.
    v : float or list of floats
        Wind speed 10m above ground level. Default in [m/s] in [fps] if `units` = 'IP'.
    rh : float or list of floats
        Relative humidity, [%].
    units : str, optional
        Select the SI (International System of Units) or the IP (Imperial Units) system. Defaults to 'SI'.
    limit_inputs : bool, optional
        By default, if the inputs are outside the standard applicability limits the
        function returns nan. If False, returns UTCI values even if input values are
        outside the applicability limits of the model. The valid input ranges are
        -50 < tdb [°C] < 50, tdb - 70 < tr [°C] < tdb + 30, and for 0.5 < v [m/s] < 17.0. Defaults to True.
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    UTCI
        A dataclass containing the Universal Thermal Climate Index and stress category.
        See :py:class:`~pythermalcomfort.classes_return.UTCI` for more details. To access the
        `utci` and `stress_category` values, use the corresponding attributes of the
        returned `Utci` instance, e.g., `result.utci`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import utci

        result = utci(tdb=25, tr=25, v=1.0, rh=50)
        print(result.utci)  # 24.6
        print(result.stress_category)  # "no thermal stress"

        result = utci(tdb=[25, 40], tr=25, v=1.0, rh=50)
        print(result.utci)  # [24.6, 40.6]
    """
    # Validate inputs using the UtciInputs class
    UTCIInputs(
        tdb=tdb,
        tr=tr,
        v=v,
        rh=rh,
        units=units,
        limit_inputs=limit_inputs,
    )

    tdb = np.asarray(tdb)
    tr = np.asarray(tr)
    v = np.asarray(v)
    rh = np.asarray(rh)

    if units.upper() == Units.IP.value:
        tdb, tr, v = units_converter(tdb=tdb, tr=tr, v=v)

    def exponential(t_db):
        g = [
            -2836.5744,
            -6028.076559,
            19.54263612,
            -0.02737830188,
            0.000016261698,
            (7.0229056 * np.power(10.0, -10)),
            (-1.8680009 * np.power(10.0, -13)),
        ]
        tk = t_db + 273.15  # air temp in K
        es = 2.7150305 * np.log1p(tk)
        for count, i in enumerate(g):
            es = es + (i * np.power(tk, count - 2))
        es = np.exp(es) * 0.01  # convert Pa to hPa
        return es

    eh_pa = exponential(tdb) * (rh / 100.0)
    delta_t_tr = tr - tdb
    pa = eh_pa / 10.0  # convert vapour pressure to kPa

    utci_approx = _utci_optimized(tdb, v, delta_t_tr, pa)

    # Checks that inputs are within the bounds accepted by the model if not return nan
    if limit_inputs:
        tdb_valid = valid_range(tdb, (-50.0, 50.0), "tdb")
        diff_valid = valid_range(tr - tdb, (-30.0, 70.0), "tr - tdb")
        v_valid = valid_range(v, (0.5, 17.0), "v")
        all_valid = ~(np.isnan(tdb_valid) | np.isnan(diff_valid) | np.isnan(v_valid))
        utci_approx = np.where(all_valid, utci_approx, np.nan)

    if units.upper() == Units.IP.value:
        utci_approx = units_converter(
            tmp=utci_approx,
            from_units=Units.SI.value.lower(),
        )[0]

    stress_categories = {
        -40.0: "extreme cold stress",
        -27.0: "very strong cold stress",
        -13.0: "strong cold stress",
        0.0: "moderate cold stress",
        9.0: "slight cold stress",
        26.0: "no thermal stress",
        32.0: "moderate heat stress",
        38.0: "strong heat stress",
        46.0: "very strong heat stress",
        1000.0: "extreme heat stress",
    }

    if round_output:
        utci_approx = np.round(utci_approx, 1)

    return UTCI(
        utci=utci_approx,
        stress_category=mapping(utci_approx, stress_categories),
    )


@vectorize(
    [
        float64(
            float64,
            float64,
            float64,
            float64,
        ),
    ],
    cache=True,
)
def _utci_optimized(
    tdb: float64, v: float64, delta_t_tr: float64, pa: float64
) -> float64:
    return (
        tdb
        + 0.607562052
        + (-0.0227712343) * tdb
        + (8.06470249 * (10 ** (-4))) * tdb * tdb
        + (-1.54271372 * (10 ** (-4))) * tdb * tdb * tdb
        + (-3.24651735 * (10 ** (-6))) * tdb * tdb * tdb * tdb
        + (7.32602852 * (10 ** (-8))) * tdb * tdb * tdb * tdb * tdb
        + (1.35959073 * (10 ** (-9))) * tdb * tdb * tdb * tdb * tdb * tdb
        + (-2.25836520) * v
        + 0.0880326035 * tdb * v
        + 0.00216844454 * tdb * tdb * v
        + (-1.53347087 * (10 ** (-5))) * tdb * tdb * tdb * v
        + (-5.72983704 * (10 ** (-7))) * tdb * tdb * tdb * tdb * v
        + (-2.55090145 * (10 ** (-9))) * tdb * tdb * tdb * tdb * tdb * v
        + (-0.751269505) * v * v
        + (-0.00408350271) * tdb * v * v
        + (-5.21670675 * (10 ** (-5))) * tdb * tdb * v * v
        + (1.94544667 * (10 ** (-6))) * tdb * tdb * tdb * v * v
        + (1.14099531 * (10 ** (-8))) * tdb * tdb * tdb * tdb * v * v
        + 0.158137256 * v * v * v
        + (-6.57263143 * (10 ** (-5))) * tdb * v * v * v
        + (2.22697524 * (10 ** (-7))) * tdb * tdb * v * v * v
        + (-4.16117031 * (10 ** (-8))) * tdb * tdb * tdb * v * v * v
        + (-0.0127762753) * v * v * v * v
        + (9.66891875 * (10 ** (-6))) * tdb * v * v * v * v
        + (2.52785852 * (10 ** (-9))) * tdb * tdb * v * v * v * v
        + (4.56306672 * (10 ** (-4))) * v * v * v * v * v
        + (-1.74202546 * (10 ** (-7))) * tdb * v * v * v * v * v
        + (-5.91491269 * (10 ** (-6))) * v * v * v * v * v * v
        + 0.398374029 * delta_t_tr
        + (1.83945314 * (10 ** (-4))) * tdb * delta_t_tr
        + (-1.73754510 * (10 ** (-4))) * tdb * tdb * delta_t_tr
        + (-7.60781159 * (10 ** (-7))) * tdb * tdb * tdb * delta_t_tr
        + (3.77830287 * (10 ** (-8))) * tdb * tdb * tdb * tdb * delta_t_tr
        + (5.43079673 * (10 ** (-10))) * tdb * tdb * tdb * tdb * tdb * delta_t_tr
        + (-0.0200518269) * v * delta_t_tr
        + (8.92859837 * (10 ** (-4))) * tdb * v * delta_t_tr
        + (3.45433048 * (10 ** (-6))) * tdb * tdb * v * delta_t_tr
        + (-3.77925774 * (10 ** (-7))) * tdb * tdb * tdb * v * delta_t_tr
        + (-1.69699377 * (10 ** (-9))) * tdb * tdb * tdb * tdb * v * delta_t_tr
        + (1.69992415 * (10 ** (-4))) * v * v * delta_t_tr
        + (-4.99204314 * (10 ** (-5))) * tdb * v * v * delta_t_tr
        + (2.47417178 * (10 ** (-7))) * tdb * tdb * v * v * delta_t_tr
        + (1.07596466 * (10 ** (-8))) * tdb * tdb * tdb * v * v * delta_t_tr
        + (8.49242932 * (10 ** (-5))) * v * v * v * delta_t_tr
        + (1.35191328 * (10 ** (-6))) * tdb * v * v * v * delta_t_tr
        + (-6.21531254 * (10 ** (-9))) * tdb * tdb * v * v * v * delta_t_tr
        + (-4.99410301 * (10 ** (-6))) * v * v * v * v * delta_t_tr
        + (-1.89489258 * (10 ** (-8))) * tdb * v * v * v * v * delta_t_tr
        + (8.15300114 * (10 ** (-8))) * v * v * v * v * v * delta_t_tr
        + (7.55043090 * (10 ** (-4))) * delta_t_tr * delta_t_tr
        + (-5.65095215 * (10 ** (-5))) * tdb * delta_t_tr * delta_t_tr
        + (-4.52166564 * (10 ** (-7))) * tdb * tdb * delta_t_tr * delta_t_tr
        + (2.46688878 * (10 ** (-8))) * tdb * tdb * tdb * delta_t_tr * delta_t_tr
        + (2.42674348 * (10 ** (-10))) * tdb * tdb * tdb * tdb * delta_t_tr * delta_t_tr
        + (1.54547250 * (10 ** (-4))) * v * delta_t_tr * delta_t_tr
        + (5.24110970 * (10 ** (-6))) * tdb * v * delta_t_tr * delta_t_tr
        + (-8.75874982 * (10 ** (-8))) * tdb * tdb * v * delta_t_tr * delta_t_tr
        + (-1.50743064 * (10 ** (-9))) * tdb * tdb * tdb * v * delta_t_tr * delta_t_tr
        + (-1.56236307 * (10 ** (-5))) * v * v * delta_t_tr * delta_t_tr
        + (-1.33895614 * (10 ** (-7))) * tdb * v * v * delta_t_tr * delta_t_tr
        + (2.49709824 * (10 ** (-9))) * tdb * tdb * v * v * delta_t_tr * delta_t_tr
        + (6.51711721 * (10 ** (-7))) * v * v * v * delta_t_tr * delta_t_tr
        + (1.94960053 * (10 ** (-9))) * tdb * v * v * v * delta_t_tr * delta_t_tr
        + (-1.00361113 * (10 ** (-8))) * v * v * v * v * delta_t_tr * delta_t_tr
        + (-1.21206673 * (10 ** (-5))) * delta_t_tr * delta_t_tr * delta_t_tr
        + (-2.18203660 * (10 ** (-7))) * tdb * delta_t_tr * delta_t_tr * delta_t_tr
        + (7.51269482 * (10 ** (-9))) * tdb * tdb * delta_t_tr * delta_t_tr * delta_t_tr
        + (9.79063848 * (10 ** (-11)))
        * tdb
        * tdb
        * tdb
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        + (1.25006734 * (10 ** (-6))) * v * delta_t_tr * delta_t_tr * delta_t_tr
        + (-1.81584736 * (10 ** (-9))) * tdb * v * delta_t_tr * delta_t_tr * delta_t_tr
        + (-3.52197671 * (10 ** (-10)))
        * tdb
        * tdb
        * v
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        + (-3.36514630 * (10 ** (-8))) * v * v * delta_t_tr * delta_t_tr * delta_t_tr
        + (1.35908359 * (10 ** (-10)))
        * tdb
        * v
        * v
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        + (4.17032620 * (10 ** (-10)))
        * v
        * v
        * v
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        + (-1.30369025 * (10 ** (-9)))
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        + (4.13908461 * (10 ** (-10)))
        * tdb
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        + (9.22652254 * (10 ** (-12)))
        * tdb
        * tdb
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        + (-5.08220384 * (10 ** (-9)))
        * v
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        + (-2.24730961 * (10 ** (-11)))
        * tdb
        * v
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        + (1.17139133 * (10 ** (-10)))
        * v
        * v
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        + (6.62154879 * (10 ** (-10)))
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        + (4.03863260 * (10 ** (-13)))
        * tdb
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        + (1.95087203 * (10 ** (-12)))
        * v
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        + (-4.73602469 * (10 ** (-12)))
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        + 5.12733497 * pa
        + (-0.312788561) * tdb * pa
        + (-0.0196701861) * tdb * tdb * pa
        + (9.99690870 * (10 ** (-4))) * tdb * tdb * tdb * pa
        + (9.51738512 * (10 ** (-6))) * tdb * tdb * tdb * tdb * pa
        + (-4.66426341 * (10 ** (-7))) * tdb * tdb * tdb * tdb * tdb * pa
        + 0.548050612 * v * pa
        + (-0.00330552823) * tdb * v * pa
        + (-0.00164119440) * tdb * tdb * v * pa
        + (-5.16670694 * (10 ** (-6))) * tdb * tdb * tdb * v * pa
        + (9.52692432 * (10 ** (-7))) * tdb * tdb * tdb * tdb * v * pa
        + (-0.0429223622) * v * v * pa
        + 0.00500845667 * tdb * v * v * pa
        + (1.00601257 * (10 ** (-6))) * tdb * tdb * v * v * pa
        + (-1.81748644 * (10 ** (-6))) * tdb * tdb * tdb * v * v * pa
        + (-1.25813502 * (10 ** (-3))) * v * v * v * pa
        + (-1.79330391 * (10 ** (-4))) * tdb * v * v * v * pa
        + (2.34994441 * (10 ** (-6))) * tdb * tdb * v * v * v * pa
        + (1.29735808 * (10 ** (-4))) * v * v * v * v * pa
        + (1.29064870 * (10 ** (-6))) * tdb * v * v * v * v * pa
        + (-2.28558686 * (10 ** (-6))) * v * v * v * v * v * pa
        + (-0.0369476348) * delta_t_tr * pa
        + 0.00162325322 * tdb * delta_t_tr * pa
        + (-3.14279680 * (10 ** (-5))) * tdb * tdb * delta_t_tr * pa
        + (2.59835559 * (10 ** (-6))) * tdb * tdb * tdb * delta_t_tr * pa
        + (-4.77136523 * (10 ** (-8))) * tdb * tdb * tdb * tdb * delta_t_tr * pa
        + (8.64203390 * (10 ** (-3))) * v * delta_t_tr * pa
        + (-6.87405181 * (10 ** (-4))) * tdb * v * delta_t_tr * pa
        + (-9.13863872 * (10 ** (-6))) * tdb * tdb * v * delta_t_tr * pa
        + (5.15916806 * (10 ** (-7))) * tdb * tdb * tdb * v * delta_t_tr * pa
        + (-3.59217476 * (10 ** (-5))) * v * v * delta_t_tr * pa
        + (3.28696511 * (10 ** (-5))) * tdb * v * v * delta_t_tr * pa
        + (-7.10542454 * (10 ** (-7))) * tdb * tdb * v * v * delta_t_tr * pa
        + (-1.24382300 * (10 ** (-5))) * v * v * v * delta_t_tr * pa
        + (-7.38584400 * (10 ** (-9))) * tdb * v * v * v * delta_t_tr * pa
        + (2.20609296 * (10 ** (-7))) * v * v * v * v * delta_t_tr * pa
        + (-7.32469180 * (10 ** (-4))) * delta_t_tr * delta_t_tr * pa
        + (-1.87381964 * (10 ** (-5))) * tdb * delta_t_tr * delta_t_tr * pa
        + (4.80925239 * (10 ** (-6))) * tdb * tdb * delta_t_tr * delta_t_tr * pa
        + (-8.75492040 * (10 ** (-8))) * tdb * tdb * tdb * delta_t_tr * delta_t_tr * pa
        + (2.77862930 * (10 ** (-5))) * v * delta_t_tr * delta_t_tr * pa
        + (-5.06004592 * (10 ** (-6))) * tdb * v * delta_t_tr * delta_t_tr * pa
        + (1.14325367 * (10 ** (-7))) * tdb * tdb * v * delta_t_tr * delta_t_tr * pa
        + (2.53016723 * (10 ** (-6))) * v * v * delta_t_tr * delta_t_tr * pa
        + (-1.72857035 * (10 ** (-8))) * tdb * v * v * delta_t_tr * delta_t_tr * pa
        + (-3.95079398 * (10 ** (-8))) * v * v * v * delta_t_tr * delta_t_tr * pa
        + (-3.59413173 * (10 ** (-7))) * delta_t_tr * delta_t_tr * delta_t_tr * pa
        + (7.04388046 * (10 ** (-7))) * tdb * delta_t_tr * delta_t_tr * delta_t_tr * pa
        + (-1.89309167 * (10 ** (-8)))
        * tdb
        * tdb
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * pa
        + (-4.79768731 * (10 ** (-7))) * v * delta_t_tr * delta_t_tr * delta_t_tr * pa
        + (7.96079978 * (10 ** (-9)))
        * tdb
        * v
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * pa
        + (1.62897058 * (10 ** (-9)))
        * v
        * v
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * pa
        + (3.94367674 * (10 ** (-8)))
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * pa
        + (-1.18566247 * (10 ** (-9)))
        * tdb
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * pa
        + (3.34678041 * (10 ** (-10)))
        * v
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * pa
        + (-1.15606447 * (10 ** (-10)))
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * pa
        + (-2.80626406) * pa * pa
        + 0.548712484 * tdb * pa * pa
        + (-0.00399428410) * tdb * tdb * pa * pa
        + (-9.54009191 * (10 ** (-4))) * tdb * tdb * tdb * pa * pa
        + (1.93090978 * (10 ** (-5))) * tdb * tdb * tdb * tdb * pa * pa
        + (-0.308806365) * v * pa * pa
        + 0.0116952364 * tdb * v * pa * pa
        + (4.95271903 * (10 ** (-4))) * tdb * tdb * v * pa * pa
        + (-1.90710882 * (10 ** (-5))) * tdb * tdb * tdb * v * pa * pa
        + 0.00210787756 * v * v * pa * pa
        + (-6.98445738 * (10 ** (-4))) * tdb * v * v * pa * pa
        + (2.30109073 * (10 ** (-5))) * tdb * tdb * v * v * pa * pa
        + (4.17856590 * (10 ** (-4))) * v * v * v * pa * pa
        + (-1.27043871 * (10 ** (-5))) * tdb * v * v * v * pa * pa
        + (-3.04620472 * (10 ** (-6))) * v * v * v * v * pa * pa
        + 0.0514507424 * delta_t_tr * pa * pa
        + (-0.00432510997) * tdb * delta_t_tr * pa * pa
        + (8.99281156 * (10 ** (-5))) * tdb * tdb * delta_t_tr * pa * pa
        + (-7.14663943 * (10 ** (-7))) * tdb * tdb * tdb * delta_t_tr * pa * pa
        + (-2.66016305 * (10 ** (-4))) * v * delta_t_tr * pa * pa
        + (2.63789586 * (10 ** (-4))) * tdb * v * delta_t_tr * pa * pa
        + (-7.01199003 * (10 ** (-6))) * tdb * tdb * v * delta_t_tr * pa * pa
        + (-1.06823306 * (10 ** (-4))) * v * v * delta_t_tr * pa * pa
        + (3.61341136 * (10 ** (-6))) * tdb * v * v * delta_t_tr * pa * pa
        + (2.29748967 * (10 ** (-7))) * v * v * v * delta_t_tr * pa * pa
        + (3.04788893 * (10 ** (-4))) * delta_t_tr * delta_t_tr * pa * pa
        + (-6.42070836 * (10 ** (-5))) * tdb * delta_t_tr * delta_t_tr * pa * pa
        + (1.16257971 * (10 ** (-6))) * tdb * tdb * delta_t_tr * delta_t_tr * pa * pa
        + (7.68023384 * (10 ** (-6))) * v * delta_t_tr * delta_t_tr * pa * pa
        + (-5.47446896 * (10 ** (-7))) * tdb * v * delta_t_tr * delta_t_tr * pa * pa
        + (-3.59937910 * (10 ** (-8))) * v * v * delta_t_tr * delta_t_tr * pa * pa
        + (-4.36497725 * (10 ** (-6))) * delta_t_tr * delta_t_tr * delta_t_tr * pa * pa
        + (1.68737969 * (10 ** (-7)))
        * tdb
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * pa
        * pa
        + (2.67489271 * (10 ** (-8)))
        * v
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * pa
        * pa
        + (3.23926897 * (10 ** (-9)))
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * pa
        * pa
        + (-0.0353874123) * pa * pa * pa
        + (-0.221201190) * tdb * pa * pa * pa
        + 0.0155126038 * tdb * tdb * pa * pa * pa
        + (-2.63917279 * (10 ** (-4))) * tdb * tdb * tdb * pa * pa * pa
        + 0.0453433455 * v * pa * pa * pa
        + (-0.00432943862) * tdb * v * pa * pa * pa
        + (1.45389826 * (10 ** (-4))) * tdb * tdb * v * pa * pa * pa
        + (2.17508610 * (10 ** (-4))) * v * v * pa * pa * pa
        + (-6.66724702 * (10 ** (-5))) * tdb * v * v * pa * pa * pa
        + (3.33217140 * (10 ** (-5))) * v * v * v * pa * pa * pa
        + (-0.00226921615) * delta_t_tr * pa * pa * pa
        + (3.80261982 * (10 ** (-4))) * tdb * delta_t_tr * pa * pa * pa
        + (-5.45314314 * (10 ** (-9))) * tdb * tdb * delta_t_tr * pa * pa * pa
        + (-7.96355448 * (10 ** (-4))) * v * delta_t_tr * pa * pa * pa
        + (2.53458034 * (10 ** (-5))) * tdb * v * delta_t_tr * pa * pa * pa
        + (-6.31223658 * (10 ** (-6))) * v * v * delta_t_tr * pa * pa * pa
        + (3.02122035 * (10 ** (-4))) * delta_t_tr * delta_t_tr * pa * pa * pa
        + (-4.77403547 * (10 ** (-6))) * tdb * delta_t_tr * delta_t_tr * pa * pa * pa
        + (1.73825715 * (10 ** (-6))) * v * delta_t_tr * delta_t_tr * pa * pa * pa
        + (-4.09087898 * (10 ** (-7)))
        * delta_t_tr
        * delta_t_tr
        * delta_t_tr
        * pa
        * pa
        * pa
        + 0.614155345 * pa * pa * pa * pa
        + (-0.0616755931) * tdb * pa * pa * pa * pa
        + 0.00133374846 * tdb * tdb * pa * pa * pa * pa
        + 0.00355375387 * v * pa * pa * pa * pa
        + (-5.13027851 * (10 ** (-4))) * tdb * v * pa * pa * pa * pa
        + (1.02449757 * (10 ** (-4))) * v * v * pa * pa * pa * pa
        + (-0.00148526421) * delta_t_tr * pa * pa * pa * pa
        + (-4.11469183 * (10 ** (-5))) * tdb * delta_t_tr * pa * pa * pa * pa
        + (-6.80434415 * (10 ** (-6))) * v * delta_t_tr * pa * pa * pa * pa
        + (-9.77675906 * (10 ** (-6))) * delta_t_tr * delta_t_tr * pa * pa * pa * pa
        + 0.0882773108 * pa * pa * pa * pa * pa
        + (-0.00301859306) * tdb * pa * pa * pa * pa * pa
        + 0.00104452989 * v * pa * pa * pa * pa * pa
        + (2.47090539 * (10 ** (-4))) * delta_t_tr * pa * pa * pa * pa * pa
        + 0.00148348065 * pa * pa * pa * pa * pa * pa
    )
