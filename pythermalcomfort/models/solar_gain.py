from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from pythermalcomfort._internal.validation import _valid_range
from pythermalcomfort.classes_input import NumericInput, SolarGainInputs
from pythermalcomfort.classes_return import SolarGain
from pythermalcomfort.environment import transpose_sharp_altitude
from pythermalcomfort.utilities import Postures

# integer codes for posture, since numba nopython mode can't dispatch on
# Python string/enum comparisons the way the rest of this module's helpers do
_POSTURE_STANDING = 0
_POSTURE_SITTING = 1
_POSTURE_SUPINE = 2

_POSTURE_CODES = {
    Postures.standing.value: _POSTURE_STANDING,
    Postures.sitting.value: _POSTURE_SITTING,
    Postures.supine.value: _POSTURE_SUPINE,
}

_ALT_RANGE = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
_AZ_RANGE = np.array(
    [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 105.0, 120.0, 135.0, 150.0, 165.0, 180.0]
)

# fp is the projected area factor, standing/supine
_FP_TABLE_STANDING = np.array(
    [
        [0.35, 0.35, 0.314, 0.258, 0.206, 0.144, 0.082],
        [0.342, 0.342, 0.31, 0.252, 0.2, 0.14, 0.082],
        [0.33, 0.33, 0.3, 0.244, 0.19, 0.132, 0.082],
        [0.31, 0.31, 0.275, 0.228, 0.175, 0.124, 0.082],
        [0.283, 0.283, 0.251, 0.208, 0.16, 0.114, 0.082],
        [0.252, 0.252, 0.228, 0.188, 0.15, 0.108, 0.082],
        [0.23, 0.23, 0.214, 0.18, 0.148, 0.108, 0.082],
        [0.242, 0.242, 0.222, 0.18, 0.153, 0.112, 0.082],
        [0.274, 0.274, 0.245, 0.203, 0.165, 0.116, 0.082],
        [0.304, 0.304, 0.27, 0.22, 0.174, 0.121, 0.082],
        [0.328, 0.328, 0.29, 0.234, 0.183, 0.125, 0.082],
        [0.344, 0.344, 0.304, 0.244, 0.19, 0.128, 0.082],
        [0.347, 0.347, 0.308, 0.246, 0.191, 0.128, 0.082],
    ]
)

# fp table, sitting
_FP_TABLE_SITTING = np.array(
    [
        [0.29, 0.324, 0.305, 0.303, 0.262, 0.224, 0.177],
        [0.292, 0.328, 0.294, 0.288, 0.268, 0.227, 0.177],
        [0.288, 0.332, 0.298, 0.29, 0.264, 0.222, 0.177],
        [0.274, 0.326, 0.294, 0.289, 0.252, 0.214, 0.177],
        [0.254, 0.308, 0.28, 0.276, 0.241, 0.202, 0.177],
        [0.23, 0.282, 0.262, 0.26, 0.233, 0.193, 0.177],
        [0.216, 0.26, 0.248, 0.244, 0.22, 0.186, 0.177],
        [0.234, 0.258, 0.236, 0.227, 0.208, 0.18, 0.177],
        [0.262, 0.26, 0.224, 0.208, 0.196, 0.176, 0.177],
        [0.28, 0.26, 0.21, 0.192, 0.184, 0.17, 0.177],
        [0.298, 0.256, 0.194, 0.174, 0.168, 0.168, 0.177],
        [0.306, 0.25, 0.18, 0.156, 0.156, 0.166, 0.177],
        [0.3, 0.24, 0.168, 0.152, 0.152, 0.164, 0.177],
    ]
)


def solar_gain(
    sol_altitude: NumericInput,
    sharp: NumericInput,
    sol_radiation_dir: NumericInput,
    sol_transmittance: NumericInput,
    f_svv: NumericInput,
    f_bes: NumericInput,
    asw: NumericInput = 0.7,
    posture: str = Postures.sitting.value,
    floor_reflectance: NumericInput = 0.6,
    round_output: bool = True,
) -> SolarGain:
    """Calculate the solar gain to the human body using the Effective Radiant Field
    (ERF) [55ASHRAE2023]_. The ERF is a measure of the net energy flux to or from the
    human body. ERF is expressed in W over human body surface area [W/m2]. In addition,
    it calculates the delta mean radiant temperature. Which is the amount by which the
    mean radiant temperature of the space should be increased if no solar radiation is
    present.

    Parameters
    ----------
    sol_altitude : float or list of floats
        Solar altitude, degrees from horizontal [deg]. Ranges between 0 and 90.
    sharp : float or list of floats
        Solar horizontal angle relative to the front of the person (SHARP) [deg].
        Ranges between 0 and 180 and is symmetrical on either side. Zero (0) degrees
        represents direct-beam radiation from the front, 90 degrees represents
        direct-beam radiation from the side, and 180 degrees represents direct-beam
        radiation from the back. SHARP is the angle between the sun and the person
        only. Orientation relative to compass or to room is not included in SHARP.
    sol_radiation_dir : float or list of floats
        Direct-beam solar radiation, [W/m2]. Ranges between 200 and 1000. See Table
        C2-3 of ASHRAE 55 2020 [55ASHRAE2023]_.
    sol_transmittance : float or list of floats
        Total solar transmittance, ranges from 0 to 1. The total solar
        transmittance of window systems, including glazing unit, blinds, and other
        façade treatments, shall be determined using one of the following methods:
        i) Provided by manufacturer or from the National Fenestration Rating
        Council approved Lawrence Berkeley National Lab International Glazing
        Database.
        ii) Glazing unit plus venetian blinds or other complex or unique shades
        shall be calculated using National Fenestration Rating Council approved
        software or Lawrence Berkeley National Lab Complex Glazing Database.
    f_svv : float or list of floats
        Fraction of sky-vault view fraction exposed to body, ranges from 0 to 1.
        It can be calculated using the function
        :py:meth:`pythermalcomfort.environment.f_svv`.
    f_bes : float or list of floats
        Fraction of the possible body surface exposed to sun, ranges from 0 to 1.
        See Table C2-2 and equation C-7 ASHRAE 55 2020 [55ASHRAE2023]_.
    asw : float or list of floats, optional
        The average short-wave absorptivity of the occupant. It will range widely,
        depending on the color of the skin of the occupant as well as the color and
        amount of clothing covering the body. Defaults to 0.7.

        .. note::
            Short-wave absorptivity typically ranges from 0.57 to 0.84, depending
            on skin and clothing color. More information is available in Blum (1945).

    posture : str, optional
        Default 'sitting' list of available options 'standing', 'supine' or 'sitting'.
    floor_reflectance : float or list of floats, optional
        Floor reflectance. It is assumed to be constant and equal to 0.6. Defaults to 0.6.
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    SolarGain
        A dataclass containing the solar gain to the human body and delta mean radiant temperature.
        See :py:class:`~pythermalcomfort.classes_return.SolarGain` for more details.
        To access the `erf` and `delta_mrt` values, use the corresponding attributes of the returned `SolarGain` instance, e.g., `result.erf`.

    Raises
    ------
    ValueError
        If the posture is not one of 'standing', 'supine', or 'sitting'.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import solar_gain

        result = solar_gain(
            sol_altitude=0,
            sharp=120,
            sol_radiation_dir=800,
            sol_transmittance=0.5,
            f_svv=0.5,
            f_bes=0.5,
            asw=0.7,
            posture="sitting",
        )
        print(result.erf)  # 42.9
        print(result.delta_mrt)  # 10.3

    Applicability
    -------------
    This model is applicable under the following conditions:

    - Solar altitude (sol_altitude) must be between 0° and 90°.
    - Solar horizontal angle (sharp) must be between 0° and 180°.
    - Direct-beam solar radiation (sol_radiation_dir) should typically range from 200 to 1000 W/m², as per ASHRAE 55 Table C2-3.
    - Solar transmittance (sol_transmittance) must be between 0 and 1.
    - Sky-vault view fraction (f_svv) and body surface exposure fraction (f_bes) must be between 0 and 1.
    - Average short-wave absorptivity (asw) typically ranges from 0.57 to 0.84, depending on skin and clothing color.
    - Posture must be one of 'standing', 'supine', or 'sitting'.
    - Floor reflectance (floor_reflectance) is assumed constant at 0.6, but can vary.
    - All inputs are in SI units (e.g., angles in degrees, radiation in W/m²).
    """
    # Validate inputs using the SolarGainInputs class
    SolarGainInputs(
        sol_altitude=sol_altitude,
        sharp=sharp,
        sol_radiation_dir=sol_radiation_dir,
        sol_transmittance=sol_transmittance,
        f_svv=f_svv,
        f_bes=f_bes,
        asw=asw,
        floor_reflectance=floor_reflectance,
    )

    sol_altitude = np.asarray(sol_altitude)
    sharp = np.asarray(sharp)
    # the fp lookup table only covers 0-90/0-180; outside that, _find_span
    # can't return a valid span at all, so clip to NaN (with a warning) here
    # rather than let the numba kernel silently wrap a -1 index
    sol_altitude = _valid_range(sol_altitude, (0.0, 90.0))
    sharp = _valid_range(sharp, (0.0, 180.0))
    sol_radiation_dir = np.asarray(sol_radiation_dir)
    sol_transmittance = np.asarray(sol_transmittance)
    f_svv = np.asarray(f_svv)
    f_bes = np.asarray(f_bes)
    asw = np.asarray(asw)
    floor_reflectance = np.asarray(floor_reflectance)

    posture = posture.lower()
    if posture not in _POSTURE_CODES:
        error_msg_posture = (
            "Posture has to be either 'standing', 'supine' or 'sitting'."
        )
        raise ValueError(error_msg_posture)
    posture_code = np.asarray(_POSTURE_CODES[posture])

    (
        sol_altitude_b,
        sharp_b,
        sol_radiation_dir_b,
        sol_transmittance_b,
        f_svv_b,
        f_bes_b,
        asw_b,
        floor_reflectance_b,
        posture_code_b,
    ) = np.broadcast_arrays(
        sol_altitude,
        sharp,
        sol_radiation_dir,
        sol_transmittance,
        f_svv,
        f_bes,
        asw,
        floor_reflectance,
        posture_code,
    )
    output_shape = sol_altitude_b.shape

    erf, d_mrt = _solar_gain_array(
        sol_altitude=np.ravel(sol_altitude_b).astype(np.float64),
        sharp=np.ravel(sharp_b).astype(np.float64),
        sol_radiation_dir=np.ravel(sol_radiation_dir_b).astype(np.float64),
        sol_transmittance=np.ravel(sol_transmittance_b).astype(np.float64),
        f_svv=np.ravel(f_svv_b).astype(np.float64),
        f_bes=np.ravel(f_bes_b).astype(np.float64),
        asw=np.ravel(asw_b).astype(np.float64),
        floor_reflectance=np.ravel(floor_reflectance_b).astype(np.float64),
        posture_code=np.ravel(posture_code_b).astype(np.int64),
    )
    erf = erf.reshape(output_shape)
    d_mrt = d_mrt.reshape(output_shape)

    if round_output:
        erf = np.round(erf, 1)
        d_mrt = np.round(d_mrt, 1)

    return SolarGain(erf=erf, delta_mrt=d_mrt)


@njit(cache=True)
def _find_span(arr, x):
    # range(n - 1), not range(n): the original pure Python loop iterated
    # range(len(arr)) and relied on valid (in-range) inputs always matching
    # before the last iteration read arr[i + 1] one past the end - out of
    # range inputs would raise IndexError there. In the numba version this
    # is called from a prange-parallelized loop (_solar_gain_array), and
    # numba does not propagate exceptions raised inside prange loops, so an
    # out-of-bounds read there would silently return garbage rather than a
    # clean error. Stopping one iteration early avoids the out-of-bounds
    # read entirely; out-of-range inputs now return -1 (same as any other
    # unmatched span) instead of raising.
    n = arr.shape[0]
    for i in range(n - 1):
        if arr[i + 1] >= x >= arr[i]:
            return i
    return -1


@njit(cache=True)
def _solar_gain_scalar(
    sol_altitude,
    sharp,
    sol_radiation_dir,
    sol_transmittance,
    f_svv,
    f_bes,
    asw,
    floor_reflectance,
    posture_code,
):
    deg_to_rad = 0.0174532925
    hr = 6
    i_diff = 0.2 * sol_radiation_dir

    if posture_code == _POSTURE_SITTING:
        fp_table = _FP_TABLE_SITTING
    else:
        fp_table = _FP_TABLE_STANDING

    if posture_code == _POSTURE_SUPINE:
        sharp, sol_altitude = transpose_sharp_altitude(sharp, sol_altitude)

    alt_i = _find_span(_ALT_RANGE, sol_altitude)
    az_i = _find_span(_AZ_RANGE, sharp)
    if alt_i == -1 or az_i == -1:
        # sol_altitude/sharp out of the table's domain (0-90/0-180), or NaN
        # (e.g. from _valid_range clipping upstream): -1 would otherwise wrap
        # around to the last row/column instead of failing, so bail out
        # explicitly rather than returning a plausible-looking wrong value.
        return np.nan, np.nan
    fp11 = fp_table[az_i, alt_i]
    fp12 = fp_table[az_i, alt_i + 1]
    fp21 = fp_table[az_i + 1, alt_i]
    fp22 = fp_table[az_i + 1, alt_i + 1]
    az1 = _AZ_RANGE[az_i]
    az2 = _AZ_RANGE[az_i + 1]
    alt1 = _ALT_RANGE[alt_i]
    alt2 = _ALT_RANGE[alt_i + 1]
    fp = fp11 * (az2 - sharp) * (alt2 - sol_altitude)
    fp += fp21 * (sharp - az1) * (alt2 - sol_altitude)
    fp += fp12 * (az2 - sharp) * (sol_altitude - alt1)
    fp += fp22 * (sharp - az1) * (sol_altitude - alt1)
    fp /= (az2 - az1) * (alt2 - alt1)

    f_eff = 0.725  # fraction of the body surface exposed to environmental radiation
    if posture_code == _POSTURE_SITTING:
        f_eff = 0.696

    sw_abs = asw
    lw_abs = 0.95

    e_diff = f_eff * f_svv * 0.5 * sol_transmittance * i_diff
    e_direct = f_eff * fp * sol_transmittance * f_bes * sol_radiation_dir
    e_reflected = (
        f_eff
        * f_svv
        * 0.5
        * sol_transmittance
        * (sol_radiation_dir * math.sin(sol_altitude * deg_to_rad) + i_diff)
        * floor_reflectance
    )

    e_solar = e_diff + e_direct + e_reflected
    erf = e_solar * (sw_abs / lw_abs)
    d_mrt = erf / (hr * f_eff)

    return erf, d_mrt


@njit(cache=True, parallel=True)
def _solar_gain_array(
    sol_altitude,
    sharp,
    sol_radiation_dir,
    sol_transmittance,
    f_svv,
    f_bes,
    asw,
    floor_reflectance,
    posture_code,
):
    n = sol_altitude.shape[0]
    erf = np.empty(n, dtype=np.float64)
    d_mrt = np.empty(n, dtype=np.float64)
    for i in prange(n):
        erf[i], d_mrt[i] = _solar_gain_scalar(
            sol_altitude[i],
            sharp[i],
            sol_radiation_dir[i],
            sol_transmittance[i],
            f_svv[i],
            f_bes[i],
            asw[i],
            floor_reflectance[i],
            posture_code[i],
        )
    return erf, d_mrt
