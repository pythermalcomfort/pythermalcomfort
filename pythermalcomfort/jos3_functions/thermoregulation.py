"""This code includes some models of jos-3 model to calculate human thermoregulation.

The values of a NumPy array containing 17 body parts
"""

import numpy as np

from pythermalcomfort.classes_return import JOS3BodyParts
from pythermalcomfort.jos3_functions import construction as cons
from pythermalcomfort.jos3_functions.matrix import IDICT
from pythermalcomfort.jos3_functions.parameters import Default
from pythermalcomfort.psychrometrics import antoine
from pythermalcomfort.utilities import Postures, Sex


def natural_convection(posture: str, tdb: float, t_skin: float) -> np.ndarray:
    """Calculate the natural convection heat transfer coefficient based on posture.

    Parameters
    ----------
    posture : str
        The posture of the subject. Valid options are "standing", "sitting", "lying", "sedentary", "supine".
    tdb : float
        Dry bulb temperature [°C].
    t_skin : float
        Skin temperature [°C].

    Returns
    -------
    hc_natural : numpy.ndarray
        Natural convection heat transfer coefficient body segments.

    Raises
    ------
    ValueError
        If the posture is not one of the valid options.
    """
    if posture.lower() == Postures.standing.value:
        # Ichihara et al., 1997, https://doi.org/10.3130/aija.62.45_5
        hc_natural = np.asarray(
            [
                4.48,
                4.48,
                2.97,
                2.91,
                2.85,
                3.61,
                3.55,
                3.67,
                3.61,
                3.55,
                3.67,
                2.80,
                2.04,
                2.04,
                2.80,
                2.04,
                2.04,
            ],
        )
    elif posture.lower() in [Postures.sitting.value, Postures.sedentary.value]:
        # Ichihara et al., 1997, https://doi.org/10.3130/aija.62.45_5
        hc_natural = np.asarray(
            [
                4.75,
                4.75,
                3.12,
                2.48,
                1.84,
                3.76,
                3.62,
                2.06,
                3.76,
                3.62,
                2.06,
                2.98,
                2.98,
                2.62,
                2.98,
                2.98,
                2.62,
            ],
        )
    elif posture.lower() in [Postures.lying.value, Postures.supine.value]:
        # Kurazumi et al., 2008, https://doi.org/10.20718/jjpa.13.1_17
        # The values are applied under cold environment.
        hc_a = np.asarray(
            [
                1.105,
                1.105,
                1.211,
                1.211,
                1.211,
                0.913,
                2.081,
                2.178,
                0.913,
                2.081,
                2.178,
                0.945,
                0.385,
                0.200,
                0.945,
                0.385,
                0.200,
            ],
        )
        hc_b = np.asarray(
            [
                0.345,
                0.345,
                0.046,
                0.046,
                0.046,
                0.373,
                0.850,
                0.297,
                0.373,
                0.850,
                0.297,
                0.447,
                0.580,
                0.966,
                0.447,
                0.580,
                0.966,
            ],
        )
        hc_natural = hc_a * (abs(tdb - t_skin) ** hc_b)
    else:
        valid_postures = [
            Postures.standing.value,
            Postures.sitting.value,
            Postures.lying.value,
            Postures.sedentary.value,
            Postures.supine.value,
        ]
        error_msg = f"Invalid posture '{posture}'. Must be one of {valid_postures}."
        raise ValueError(error_msg)
    return hc_natural


def forced_convection(v: float) -> np.ndarray:
    """Calculate the forced convection heat transfer coefficient.

    Ichihara et al., 1997, https://doi.org/10.3130/aija.62.45_5

    Parameters
    ----------
    v : float
        Air velocity [m/s].

    Returns
    -------
    np.ndarray
        Forced convection heat transfer coefficient body segments.
    """
    hc_a = np.asarray(
        [
            15.0,
            15.0,
            11.0,
            17.0,
            13.0,
            17.0,
            17.0,
            20.0,
            17.0,
            17.0,
            20.0,
            14.0,
            15.8,
            15.1,
            14.0,
            15.8,
            15.1,
        ],
    )
    hc_b = np.asarray(
        [
            0.62,
            0.62,
            0.67,
            0.49,
            0.60,
            0.59,
            0.61,
            0.60,
            0.59,
            0.61,
            0.60,
            0.61,
            0.74,
            0.62,
            0.61,
            0.74,
            0.62,
        ],
    )
    hc_forced = hc_a * (v**hc_b)
    return hc_forced


def conv_coef(
    posture: str,
    v: float | np.ndarray,
    tdb: float | np.ndarray,
    t_skin: float | np.ndarray,
) -> np.ndarray:
    """Calculate convective heat transfer coefficient (hc) [W/(m2*K)].

    Ichihara et al., 1997, https://doi.org/10.3130/aija.62.45_5
    Kurazumi et al., 2008, https://doi.org/10.20718/jjpa.13.1_17

    Parameters
    ----------
    posture : str, optional
        Select posture from standing, sitting, lying, sedentary or supine.
    v : float or iter, optional
        Air velocity [m/s]. If iter is input, its length should be 17.
        The default is 0.1.
    tdb : float or iter, optional
        Air temperature [°C]. If iter is input, its length should be 17.
        The default is 28.8.
    t_skin : float or iter, optional
        Skin temperature [°C]. If iter is input, its length should be 17.
        The default is 34.0.

    Returns
    -------
    np.ndarray
        Convective heat transfer coefficient (hc) [W/(m2*K)].
    """
    hc_natural = natural_convection(posture=posture, tdb=tdb, t_skin=t_skin)
    hc_forced = forced_convection(v=v)
    hc = np.where(v < 0.2, hc_natural, hc_forced)  # hc [W/(m2*K))]
    return hc


def rad_coef(posture: str) -> np.ndarray:
    """Calculate radiative heat transfer coefficient (hr) [W/(m2*K)].

    Parameters
    ----------
    posture : str, optional
        Select posture from standing, sitting, lying, sedentary or supine.

    Returns
    -------
    np.ndarray
        Radiative heat transfer coefficient (hr) [W/(m2*K)].
    """
    if posture.lower() == Postures.standing.value:
        # Ichihara et al., 1997, https://doi.org/10.3130/aija.62.45_5
        hr = np.asarray(
            [
                4.89,
                4.89,
                4.32,
                4.09,
                4.32,
                4.55,
                4.43,
                4.21,
                4.55,
                4.43,
                4.21,
                4.77,
                5.34,
                6.14,
                4.77,
                5.34,
                6.14,
            ],
        )
    elif posture.lower() in [Postures.sitting.value, Postures.sedentary.value]:
        # Ichihara et al., 1997, https://doi.org/10.3130/aija.62.45_5
        hr = np.asarray(
            [
                4.96,
                4.96,
                3.99,
                4.64,
                4.21,
                4.96,
                4.21,
                4.74,
                4.96,
                4.21,
                4.74,
                4.10,
                4.74,
                6.36,
                4.10,
                4.74,
                6.36,
            ],
        )
    elif posture.lower() in [Postures.lying.value, Postures.supine.value]:
        # Kurazumi et al., 2008, https://doi.org/10.20718/jjpa.13.1_17
        hr = np.asarray(
            [
                5.475,
                5.475,
                3.463,
                3.463,
                3.463,
                4.249,
                4.835,
                4.119,
                4.249,
                4.835,
                4.119,
                4.440,
                5.547,
                6.085,
                4.440,
                5.547,
                6.085,
            ],
        )
    else:
        valid_postures = [
            Postures.standing.value,
            Postures.sitting.value,
            Postures.lying.value,
            Postures.sedentary.value,
            Postures.supine.value,
        ]
        error_msg = f"Invalid posture '{posture}'. Must be one of {valid_postures}."
        raise ValueError(
            error_msg,
        )
    return hr


def fixed_hc(hc: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Fix hc values to fit two-node-model's values.

    Parameters
    ----------
    hc : np.ndarray
        Convective heat transfer coefficient (hc) [W/(m2*K)].
    v : float
        Air velocity [m/s].

    Returns
    -------
    np.ndarray
        Fixed convective heat transfer coefficient (hc) [W/(m2*K)].
    """
    mean_hc = np.average(hc, weights=cons.Default.local_bsa)
    mean_va = np.average(v, weights=cons.Default.local_bsa)
    mean_hc_whole = max(3, 8.600001 * (mean_va**0.53))
    _fixed_hc = hc * mean_hc_whole / mean_hc
    return _fixed_hc


def fixed_hr(hr: np.ndarray) -> np.ndarray:
    """Fix hr values to fit two-node-model's values.

    Parameters
    ----------
    hr : np.ndarray
        Radiative heat transfer coefficient (hr) [W/(m2*K)].

    Returns
    -------
    np.ndarray
        Fixed radiative heat transfer coefficient (hr) [W/(m2*K)].
    """
    mean_hr = np.average(hr, weights=cons.Default.local_bsa)
    _fixed_hr = hr * 4.7 / mean_hr
    return _fixed_hr


def operative_temp(tdb, tr, hc, hr):
    """Calculate operative temperature [°C].

    Parameters
    ----------
    tdb : float or array
        Air temperature [°C]
    tr : float or array
        Mean radiant temperature [°C]
    hc : float or array
        Convective heat transfer coefficient [W/(m2*K)]
    hr : float or array
        Radiative heat transfer coefficient [W/(m2*K)]

    Returns
    -------
    to : float or array
        Operative temperature [°C]
    """
    to = (hc * tdb + hr * tr) / (hc + hr)
    return to


def clo_area_factor(clo):
    """Calculate clothing area factor.

    Parameters
    ----------
    clo : float or array
        Clothing insulation [clo]

    Returns
    -------
    fcl : float or array
        clothing area factor [-]
    """
    fcl = np.where(clo < 0.5, clo * 0.2 + 1, clo * 0.1 + 1.05)
    return fcl


# TODO this function is different from ISO 9920
def dry_r(hc, hr, clo):
    """Calculate total sensible thermal resistance (between the skin and ambient air).

    Parameters
    ----------
    hc : float or array
        Convective heat transfer coefficient (hc) [W/(m2*K)].
    hr : float or array
        Radiative heat transfer coefficient (hr) [W/(m2*K)].
    clo : float or array
        Clothing insulation [clo].

    Returns
    -------
    np.ndarray
        Total sensible thermal resistance between skin and ambient.

    Raises
    ------
    ValueError
        If any of the input parameters are negative.
    """
    if (np.asarray(hc) < 0).any() or (np.asarray(hr) < 0).any():
        raise ValueError("Input parameters hc and hr must be non-negative.")

    fcl = clo_area_factor(clo)
    r_a = 1 / (hc + hr)
    r_cl = 0.155 * clo
    r_t = r_a / fcl + r_cl
    return r_t


def wet_r(
    hc,
    clo,
    i_clo,
    lewis_rate,
):
    """Calculate total evaporative thermal resistance (between the skin and ambient
    air).

    Parameters
    ----------
    hc : float or array
        Convective heat transfer coefficient (hc) [W/(m2*K)].
    clo : float or array
        Clothing insulation [clo].
    i_clo : float or array, optional
        Clothing vapor permeation efficiency [-]. The default is 0.45.
    lewis_rate : float, optional
        Lewis rate [K/kPa]. The default is 16.5.

    Returns
    -------
    np.ndarray
        Total evaporative thermal resistance.

    Raises
    ------
    ValueError
        If any of the input parameters are negative.
    """
    if (np.asarray(hc) < 0).any():
        raise ValueError("Input parameters hc must be non-negative.")

    fcl = clo_area_factor(clo)
    r_cl = 0.155 * clo
    r_ea = 1 / (lewis_rate * hc)
    r_ecl = r_cl / (lewis_rate * i_clo)
    r_et = r_ea / fcl + r_ecl
    return r_et


def error_signals(err_sk=0.0):
    """Calculate WRMS and CLDS signals of thermoregulation.

    Parameters
    ----------
    err_sk : float or np.ndarray, optional
        Difference between set-point and skin temperatures [°C]. The default is 0.

    Returns
    -------
    tuple
        Warm signal (WRMS) [°C] and Cold signal (CLDS) [°C].
    """
    err_sk = np.asarray(err_sk, dtype=float)

    # SKINR (Distribution coefficients of thermal receptor) [-]
    receptor = np.asarray(
        [
            0.0549,
            0.0146,
            0.1492,
            0.1321,
            0.2122,
            0.0227,
            0.0117,
            0.0923,
            0.0227,
            0.0117,
            0.0923,
            0.0501,
            0.0251,
            0.0167,
            0.0501,
            0.0251,
            0.0167,
        ],
    )

    # wrms signal
    wrm = np.maximum(err_sk, 0) * receptor
    warm_signal_sum = wrm.sum()

    # clds signal
    cld = np.minimum(err_sk, 0) * -receptor
    cold_signal_sum = cld.sum()
    return warm_signal_sum, cold_signal_sum


def tetens(x: float) -> float:
    """Calculate saturated vapor pressure using Tetens equation [kPa].

    Parameters
    ----------
    x : float
        Temperature [°C].

    Returns
    -------
    float
        Saturated vapor pressure [kPa].
    """
    return 0.61078 * 10 ** (7.5 * x / (x + 237.3))


def evaporation(
    err_cr,
    err_sk,
    t_skin,
    tdb,
    rh,
    ret,
    height,
    weight,
    bsa_equation,
    age,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate evaporative heat loss.

    Parameters
    ----------
    err_cr, err_sk : array
        Difference between set-point and body temperatures [°C].
    t_skin : array
        Skin temperatures [°C].
    tdb : array
        Air temperatures at local body segments [°C].
    rh : array
        Relative humidity at local body segments [%].
    ret : array
        Total evaporative thermal resistances [m2.K/W].
    height : float, optional
        Body height [m].
    weight : float, optional
        Body weight [kg].
    bsa_equation : str, optional
        The equation name (str) of bsa calculation. Choose one from BodySurfaceAreaEquations.
    age : float, optional
        age [years].

    Returns
    -------
    wet : array
        Local skin wettedness [-].
    e_sk : array
        Evaporative heat loss at the skin by sweating and diffuse [W].
    e_max : array
        Maximum evaporative heat loss at the skin [W].
    e_sweat : TYPE
        Evaporative heat loss at the skin by only sweating [W].
    """
    wrms, clds = error_signals(
        err_sk,
    )  # Thermoregulation signals
    bsar = cons.bsa_rate(
        height,
        weight,
        bsa_equation,
    )  # bsa rate
    bsa = Default.local_bsa * bsar  # bsa
    p_a = antoine(tdb) * rh / 100  # Saturated vapor pressure of ambient [kPa]
    p_sk_s = antoine(t_skin)  # Saturated vapor pressure at the skin [kPa]

    e_max = (p_sk_s - p_a) / ret * bsa  # Maximum evaporative heat loss

    # Replace any zero values in the e_max array with 0.001 to avoid causing a divide by 0 error
    e_max = np.where(e_max == 0, 0.001, e_max)

    # SKINS
    skin_sweat = np.asarray(
        [
            0.064,
            0.017,
            0.146,
            0.129,
            0.206,
            0.051,
            0.026,
            0.0155,
            0.051,
            0.026,
            0.0155,
            0.073,
            0.036,
            0.0175,
            0.073,
            0.036,
            0.0175,
        ],
    )

    sig_sweat = (371.2 * err_cr[0]) + (33.64 * (wrms - clds))
    sig_sweat = max(sig_sweat, 0)
    sig_sweat *= bsar

    # Signal decrement by aging
    if age < 60:
        sd_sweat = np.ones(Default.num_body_parts)
    else:  # age >= 60
        sd_sweat = np.asarray(
            [
                0.69,
                0.69,
                0.59,
                0.52,
                0.40,
                0.75,
                0.75,
                0.75,
                0.75,
                0.75,
                0.75,
                0.40,
                0.40,
                0.40,
                0.40,
                0.40,
                0.40,
            ],
        )
    e_sweat = skin_sweat * sig_sweat * sd_sweat * 2 ** (err_sk / 10)
    wet = 0.06 + 0.94 * (e_sweat / e_max)
    wet = np.minimum(wet, 1)  # Wettedness' upper limit
    e_sk = wet * e_max
    e_sweat = (wet - 0.06) / 0.94 * e_max  # Effective sweating
    return wet, e_sk, e_max, e_sweat


def skin_blood_flow(
    err_cr,
    err_sk,
    height,
    weight,
    bsa_equation,
    age,
    ci,
) -> np.ndarray:
    """Calculate skin blood flow rate (bf_skin) [L/h].

    Parameters
    ----------
    err_cr, err_sk : array
        Difference between set-point and body temperatures [°C].
    height : float, optional
        Body height [m].
    weight : float, optional
        Body weight [kg].
    bsa_equation : str, optional
        The equation name (str) of bsa calculation. Choose one from BodySurfaceAreaEquations.
    age : float, optional
        age [years].
    ci : float, optional
        Cardiac index [L/min/㎡].

    Returns
    -------
    bf_skin : array
        Skin blood flow rate [L/h].
    """
    wrms, clds = error_signals(err_sk)

    # BFBsk
    bfb_sk = np.asarray(
        [
            1.754,
            0.325,
            1.967,
            1.475,
            2.272,
            0.91,
            0.508,
            1.114,
            0.91,
            0.508,
            1.114,
            1.456,
            0.651,
            0.934,
            1.456,
            0.651,
            0.934,
        ],
    )
    # SKIND
    skin_dilat = np.asarray(
        [
            0.0692,
            0.0992,
            0.0580,
            0.0679,
            0.0707,
            0.0400,
            0.0373,
            0.0632,
            0.0400,
            0.0373,
            0.0632,
            0.0736,
            0.0411,
            0.0623,
            0.0736,
            0.0411,
            0.0623,
        ],
    )
    # SKINC
    skin_stric = np.asarray(
        [
            0.0213,
            0.0213,
            0.0638,
            0.0638,
            0.0638,
            0.0213,
            0.0213,
            0.1489,
            0.0213,
            0.0213,
            0.1489,
            0.0213,
            0.0213,
            0.1489,
            0.0213,
            0.0213,
            0.1489,
        ],
    )

    sig_dilat = (100.5 * err_cr[0]) + (6.4 * (wrms - clds))
    sig_stric = (-10.8 * err_cr[0]) + (-10.8 * (wrms - clds))
    sig_dilat = max(sig_dilat, 0)
    sig_stric = max(sig_stric, 0)

    # Signal decrement by aging
    if age < 60:
        sd_dilat = np.ones(Default.num_body_parts)
        sd_stric = np.ones(Default.num_body_parts)
    else:  # age >= 60
        sd_dilat = np.asarray(
            [
                0.91,
                0.91,
                0.47,
                0.47,
                0.31,
                0.47,
                0.47,
                0.47,
                0.47,
                0.47,
                0.47,
                0.31,
                0.31,
                0.31,
                0.31,
                0.31,
                0.31,
            ],
        )
        sd_stric = np.ones(Default.num_body_parts)

    # Skin blood flow [L/h]
    bf_skin = (
        (1 + skin_dilat * sd_dilat * sig_dilat)
        / (1 + skin_stric * sd_stric * sig_stric)
        * bfb_sk
        * 2 ** (err_sk / 6)
    )
    # Basal blood flow rate to the standard body [-]
    bfb_rate = cons.bfb_rate(
        height,
        weight,
        bsa_equation,
        age,
        ci,
    )
    bf_skin *= bfb_rate
    return bf_skin


def ava_blood_flow(
    err_cr,
    err_sk,
    height,
    weight,
    bsa_equation,
    age,
    ci,
):
    """Calculate areteriovenous anastmoses (AVA) blood flow rate [L/h] based on
    Takemori's model, 1995.

    Parameters
    ----------
    err_cr, err_sk : array
        Difference between set-point and body temperatures [°C].
    height : float, optional
        Body height [m].
    weight : float, optional
        Body weight [kg].
    bsa_equation : str, optional
        The equation name (str) of bsa calculation. Choose one from BodySurfaceAreaEquations.
    age : float, optional
        age [years].
    ci : float, optional
        Cardiac index [L/min/m2].

    Returns
    -------
    bf_ava_hand, bf_ava_foot : array
        AVA blood flow rate at hand and foot [L/h].
    """
    # Cal. mean error body core temp.
    cap_bcr = [10.2975, 9.3935, 4.488]  # Thermal capacity at chest, back and pelvis
    err_bcr = np.average(err_cr[2:5], weights=cap_bcr)

    # Cal. mean error skin temp.
    bsa = cons.Default.local_bsa
    err_msk = np.average(err_sk, weights=bsa)

    # Openness of AVA [-]
    sig_ava_hand = 0.265 * (err_msk + 0.43) + 0.953 * (err_bcr + 0.1905) + 0.9126
    sig_ava_foot = 0.265 * (err_msk - 0.997) + 0.953 * (err_bcr + 0.0095) + 0.9126

    sig_ava_hand = min(sig_ava_hand, 1)
    sig_ava_hand = max(sig_ava_hand, 0)
    sig_ava_foot = min(sig_ava_foot, 1)
    sig_ava_foot = max(sig_ava_foot, 0)

    # Basal blood flow rate to the standard body [-]
    bfb_rate = cons.bfb_rate(
        height,
        weight,
        bsa_equation,
        age,
        ci,
    )
    # AVA blood flow rate [L/h]
    bf_ava_hand = 1.71 * bfb_rate * sig_ava_hand  # Hand
    bf_ava_foot = 2.16 * bfb_rate * sig_ava_foot  # Foot
    return bf_ava_hand, bf_ava_foot


def basal_met(
    height,
    weight,
    age,
    sex,
    bmr_equation,
):
    """Calculate basal metabolic rate [W].

    Parameters
    ----------
    height : float, optional
        Body height [m]. The default is 1.72.
    weight : float, optional
        Body weight [kg]. The default is 74.43.
    age : float, optional
        age [years]. The default is 20.
    sex : str, optional
        Choose male or female. The default is "male".
    bmr_equation : str, optional
        Choose harris-benedict or ganpule. The default is "harris-benedict".

    Returns
    -------
     bmr: float
        Basal metabolic rate [W].
    """
    if bmr_equation == "harris-benedict":
        if sex == Sex.male.value:
            bmr = 88.362 + 13.397 * weight + 500.3 * height - 5.677 * age
        else:
            bmr = 447.593 + 9.247 * weight + 479.9 * height - 4.330 * age

    elif bmr_equation == "harris-benedict_origin":
        if sex == Sex.male.value:
            bmr = 66.4730 + 13.7516 * weight + 500.33 * height - 6.7550 * age
        else:
            bmr = 655.0955 + 9.5634 * weight + 184.96 * height - 4.6756 * age

    elif bmr_equation == "japanese" or bmr_equation == "ganpule":
        # Ganpule et al., 2007, https://doi.org/10.1038/sj.ejcn.1602645
        if sex == Sex.male.value:
            bmr = 0.0481 * weight + 2.34 * height - 0.0138 * age - 0.4235
        else:
            bmr = 0.0481 * weight + 2.34 * height - 0.0138 * age - 0.9708
        bmr *= 1000 / 4.186
    else:
        # TODO why we cannot use dubois?
        valid_equations = [
            "harris-benedict",
            "harris-benedict_origin",
            "japanese",
            "ganpule",
        ]
        error_msg = (
            f"Invalid equation: '{bmr_equation}'. Must be one of {valid_equations}."
        )
        raise ValueError(error_msg)

    bmr *= 0.048  # [kcal/day] to [W]

    # Set minimum BMR value in W
    min_bmr_in_w = 68
    bmr = max(bmr, min_bmr_in_w)

    return bmr


def local_mbase(
    height: float = Default.height,
    weight: float = Default.weight,
    age: int = Default.age,
    sex: str = Default.sex,
    bmr_equation: str = Default.bmr_equation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate local basal metabolic rate [W].

    Parameters
    ----------
    height : float, optional
        Body height [m]. The default is 1.72.
    weight : float, optional
        Body weight [kg]. The default is 74.43.
    age : int, optional
        age [years]. The default is 20.
    sex : str, optional
        Choose male or female.
    bmr_equation : str, optional
        Choose harris-benedict or ganpule. The default is "harris-benedict".

    Returns
    -------
    mbase : array
        Local basal metabolic rate (Mbase) [W].
    """
    mbase_all = basal_met(height, weight, age, sex, bmr_equation)
    # Distribution coefficient of basal metabolic rate
    mbf_cr = np.asarray(
        [
            0.19551,
            0.00324,
            0.28689,
            0.25677,
            0.09509,
            0.01435,
            0.00409,
            0.00106,
            0.01435,
            0.00409,
            0.00106,
            0.01557,
            0.00422,
            0.00250,
            0.01557,
            0.00422,
            0.00250,
        ],
    )
    mbf_ms = np.asarray(
        [
            0.00252,
            0.0,
            0.0,
            0.0,
            0.04804,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )
    mbf_fat = np.asarray(
        [
            0.00127,
            0.0,
            0.0,
            0.0,
            0.00950,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )
    mbf_sk = np.asarray(
        [
            0.00152,
            0.00033,
            0.00211,
            0.00187,
            0.00300,
            0.00059,
            0.00031,
            0.00059,
            0.00059,
            0.00031,
            0.00059,
            0.00144,
            0.00027,
            0.00118,
            0.00144,
            0.00027,
            0.00118,
        ],
    )

    mbase_cr = mbf_cr * mbase_all
    mbase_ms = mbf_ms * mbase_all
    mbase_fat = mbf_fat * mbase_all
    mbase_sk = mbf_sk * mbase_all
    return mbase_cr, mbase_ms, mbase_fat, mbase_sk


def local_q_work(bmr, par):
    """Calculate local thermogenesis by work [W].

    Parameters
    ----------
    bmr : float
        Basal metabolic rate [W].
    par : float
        Physical activity ratio [-].

    Returns
    -------
    q_work : array
        Local thermogenesis by work [W].

    Raises
    ------
    ValueError
        If par is less than 1.
    """
    if par < 1:
        raise ValueError("par must be 1 or more")

    q_work_all = (par - 1) * bmr

    # Distribution coefficient of thermogenesis by work
    workf = np.asarray(
        [
            0,
            0,
            0.091,
            0.08,
            0.129,
            0.0262,
            0.0139,
            0.005,
            0.0262,
            0.0139,
            0.005,
            0.2010,
            0.0990,
            0.005,
            0.2010,
            0.0990,
            0.005,
        ],
    )
    q_work = q_work_all * workf
    return q_work


PRE_SHIV = 0


def shivering(
    err_cr: np.ndarray,
    err_sk: np.ndarray,
    t_core: np.ndarray,
    t_skin: np.ndarray,
    height,
    weight,
    bsa_equation,
    age,
    sex,
    dtime=60,
    options=None,
):
    """Calculate local thermogenesis by shivering [W].

    Parameters
    ----------
    err_cr, err_sk : array
        Difference between set-point and body temperatures [°C].
    t_core, t_skin : array
        Core and skin temperatures [°C].
    height : float, optional
        Body height [m].
    weight : float, optional
        Body weight [kg].
    bsa_equation : str, optional
        The equation name (str) of bsa calculation. Choose one from BodySurfaceAreaEquations.
    age : float, optional
        age [years].
    sex : str, optional
        Choose male or female.
    dtime : float, optional
        Interval of analysis time.

    Returns
    -------
    q_shiv : array
        Local thermogenesis by shivering [W].
    """
    # Integrated error signal in the warm and cold receptors
    wrms, clds = error_signals(err_sk)

    # Distribution coefficient of thermogenesis by shivering
    shivf = np.asarray(
        [
            0.0339,
            0.0436,
            0.27394,
            0.24102,
            0.38754,
            0.00243,
            0.00137,
            0.0002,
            0.00243,
            0.00137,
            0.0002,
            0.0039,
            0.00175,
            0.00035,
            0.0039,
            0.00175,
            0.00035,
        ],
    )
    # integrated error signal of shivering
    sig_shiv = 24.36 * clds * (-err_cr[0])
    sig_shiv = max(sig_shiv, 0)

    if options:
        if options["shivering_threshold"]:
            # Asaka, 2016
            # Threshold of starting shivering
            tskm = np.average(t_skin, weights=cons.Default.local_bsa)  # Mean skin temp.
            if tskm < 31:
                thres = 36.6
            elif sex == Sex.male.value:
                thres = -0.2436 * tskm + 44.10
            else:  # sex == "female":
                thres = -0.2250 * tskm + 43.05
            # Second threshold of starting shivering
            if thres < t_core[0]:
                sig_shiv = 0

    # TODO I do not think this is well written code
    global PRE_SHIV  # Previous shivering thermogenesis [W]
    if options:
        if options["limit_dshiv/dt"]:
            dshiv = sig_shiv - PRE_SHIV  # Asaka, 2016 dshiv < 0.0077 [W/s]
            if options["limit_dshiv/dt"] is True:  # default is 0.0077 [W/s]
                limit_dshiv = 0.0077 * dtime
            else:
                limit_dshiv = options["limit_dshiv/dt"] * dtime
            if dshiv > limit_dshiv:
                sig_shiv = limit_dshiv + PRE_SHIV
            elif dshiv < -limit_dshiv:
                sig_shiv = -limit_dshiv + PRE_SHIV
        PRE_SHIV = sig_shiv

    # Signal sd_shiv by aging
    if age < 30:
        sd_shiv = np.ones(Default.num_body_parts)
    elif age < 40:
        sd_shiv = np.ones(Default.num_body_parts) * 0.97514
    elif age < 50:
        sd_shiv = np.ones(Default.num_body_parts) * 0.95028
    elif age < 60:
        sd_shiv = np.ones(Default.num_body_parts) * 0.92818
    elif age < 70:
        sd_shiv = np.ones(Default.num_body_parts) * 0.90055
    elif age < 80:
        sd_shiv = np.ones(Default.num_body_parts) * 0.86188
    else:  # age >= 80
        sd_shiv = np.ones(Default.num_body_parts) * 0.82597

    # Ratio of body surface area to the standard body [-]
    bsar = cons.bsa_rate(height, weight, bsa_equation)

    # Local thermogenesis by shivering [W]
    q_shiv = shivf * bsar * sd_shiv * sig_shiv
    return q_shiv


def nonshivering(
    err_sk,
    height,
    weight,
    bsa_equation,
    age,
    cold_acclimation=False,
    batpositive=True,
):
    """Calculate local metabolic rate by non-shivering [W].

    Parameters
    ----------
    err_sk : array
        Difference between set-point and body temperatures [°C].
    height : float, optional
        Body height [m].
    weight : float, optional
        Body weight [kg].
    bsa_equation : str, optional
        The equation name (str) of bsa calculation. Choose one from BodySurfaceAreaEquations.
    age : float, optional
        age [years].
    cold_acclimation : bool, optional
        Whether the subject acclimates cold environment or not.
        The default is False.
    batpositive : bool, optional
        Whether BAT activity is positive or not.
        The default is True.

    Returns
    -------
    q_nst : array
        Local metabolic rate by non-shivering [W].
    """
    # NST (Non-Shivering Thermogenesis) model, Asaka, 2016
    wrms, clds = error_signals(err_sk)

    # BMI (Body Mass Index)
    bmi = weight / height**2

    # BAT: brown adipose tissue [SUV]
    bat = 10 ** (-0.10502 * bmi + 2.7708)

    # age factor
    if age < 30:
        bat *= 1.61
    elif age < 40:
        bat *= 1.00
    else:  # age >= 40
        bat *= 0.80

    if cold_acclimation:
        bat += 3.46

    if not batpositive:
        # incidence age factor: T.Yoneshiro 2011
        if age < 30:  # age = 20s or younger
            bat *= 44 / 83
        elif age < 40:  # age = 30s
            bat *= 15 / 38
        elif age < 50:  # age = 40s
            bat *= 7 / 26
        elif age < 60:  # age = 50s
            bat *= 1 / 8
        else:  # age > 60
            bat *= 0

    # NST limit
    thres = (1.80 * bat + 2.43) + 5.62  # [W]

    sig_nst = 2.8 * clds  # [W]
    sig_nst = min(sig_nst, thres)

    # Distribution coefficient of thermogenesis by non-shivering
    nstf = np.asarray(
        [
            0.000,
            0.190,
            0.000,
            0.190,
            0.190,
            0.215,
            0.000,
            0.000,
            0.215,
            0.000,
            0.000,
            0.000,
            0.000,
            0.000,
            0.000,
            0.000,
            0.000,
        ],
    )

    # Ratio of body surface area to the standard body [-]
    bsar = cons.bsa_rate(height, weight, bsa_equation)

    # Local thermogenesis by non-shivering [W]
    q_nst = bsar * nstf * sig_nst
    return q_nst


def sum_m(mbase: np.ndarray, q_work, q_shiv, q_nst):
    """Calculate total thermogenesis in each layer [W].

    Parameters
    ----------
    mbase : array
        Local basal metabolic rate (Mbase) [W].
    q_work : array
        Local thermogenesis by work [W].
    q_shiv : array
        Local thermogenesis by shivering [W].
    q_nst : array
        Local thermogenesis by non-shivering [W].

    Returns
    -------
    q_thermogenesis_core, q_thermogenesis_muscle, q_thermogenesis_fat, q_thermogenesis_skin : array
        Total thermogenesis in core, muscle, fat, skin layers [W].
    """
    q_thermogenesis_core = mbase[0].copy()
    q_thermogenesis_muscle = mbase[1].copy()
    q_thermogenesis_fat = mbase[2].copy()
    q_thermogenesis_skin = mbase[3].copy()

    for i, bn in enumerate(JOS3BodyParts.get_attribute_names()):
        # If the segment has a muscle layer, muscle thermogenesis increases by the activity.
        if IDICT[bn]["muscle"] is not None:
            q_thermogenesis_muscle[i] += q_work[i] + q_shiv[i]
        # In other segments, core thermogenesis increase, instead of muscle.
        else:
            q_thermogenesis_core[i] += q_work[i] + q_shiv[i]
    q_thermogenesis_core += q_nst  # Non-shivering thermogenesis occurs in core layers
    return (
        q_thermogenesis_core,
        q_thermogenesis_muscle,
        q_thermogenesis_fat,
        q_thermogenesis_skin,
    )


def cr_ms_fat_blood_flow(
    q_work,
    q_shiv,
    height,
    weight,
    bsa_equation,
    age,
    ci,
):
    """Calculate core, muscle and fat blood flow rate [L/h].

    Parameters
    ----------
    q_work : array
        Heat production by work [W].
    q_shiv : array
        Heat production by shivering [W].
    height : float, optional
        Body height [m].
    weight : float, optional
        Body weight [kg].
    bsa_equation : str, optional
        The equation name (str) of bsa calculation. Choose one from BodySurfaceAreaEquations.
    age : float, optional
        age [years].
    ci : float, optional
        Cardiac index [L/min/㎡].

    Returns
    -------
    bf_core, bf_muscle, bf_fat : array
        Core, muscle and fat blood flow rate [L/h].
    """
    # Basal blood flow rate [L/h]
    # core, CBFB
    bfb_core = np.asarray(
        [
            35.251,
            15.240,
            89.214,
            87.663,
            18.686,
            1.808,
            0.940,
            0.217,
            1.808,
            0.940,
            0.217,
            1.406,
            0.164,
            0.080,
            1.406,
            0.164,
            0.080,
        ],
    )
    # muscle, MSBFB
    bfb_muscle = np.asarray(
        [
            0.682,
            0.0,
            0.0,
            0.0,
            12.614,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )
    # fat, FTBFB
    bfb_fat = np.asarray(
        [
            0.265,
            0.0,
            0.0,
            0.0,
            2.219,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    )

    bfb_rate = cons.bfb_rate(height, weight, bsa_equation, age, ci)
    bf_core = bfb_core * bfb_rate
    bf_muscle = bfb_muscle * bfb_rate
    bf_fat = bfb_fat * bfb_rate

    for i, bn in enumerate(JOS3BodyParts.get_attribute_names()):
        # If the segment has a muscle layer, muscle blood flow increases.
        if IDICT[bn]["muscle"] is not None:
            bf_muscle[i] += (q_work[i] + q_shiv[i]) / 1.163
        # In other segments, core blood flow increase, instead of muscle blood flow.
        else:
            bf_core[i] += (q_work[i] + q_shiv[i]) / 1.163
    return bf_core, bf_muscle, bf_fat


def sum_bf(bf_core, bf_muscle, bf_fat, bf_skin, bf_ava_hand, bf_ava_foot):
    """Sum the total blood flow in various body parts.

    Parameters
    ----------
    bf_core : array
        Blood flow rate in the core region [L/h].
    bf_muscle : array
        Blood flow rate in the muscle region [L/h].
    bf_fat : array
        Blood flow rate in the fat region [L/h].
    bf_skin : array
        Blood flow rate in the skin region [L/h].
    bf_ava_hand : array
        AVA blood flow rate in one hand [L/h].
    bf_ava_foot : array
        AVA blood flow rate in one foot [L/h].

    Returns
    -------
    co : float
        Cardiac output (the sum of the whole blood flow rate) [L/h].
    """
    # Cardiac output (CO)
    co = 0
    co += bf_core.sum()
    co += bf_muscle.sum()
    co += bf_fat.sum()
    co += bf_skin.sum()
    co += 2 * bf_ava_hand
    co += 2 * bf_ava_foot
    return co


def resp_heat_loss(tdb: float, p_a: float, q_thermogenesis_total: float):
    """Calculate heat loss by respiration [W].

    Parameters
    ----------
    tdb : float
        Dry bulb air temperature [°C].
    p_a : float
        Water vapor pressure in the ambient air [kPa].
    q_thermogenesis_total : float
        Total thermogenesis [W].

    Returns
    -------
    res_sh : float
        Sensible heat loss by respiration [W].
    res_lh : float
        Latent heat loss by respiration [W].
    """
    res_sh = 0.0014 * q_thermogenesis_total * (34 - tdb)  # Sensible heat loss
    res_lh = 0.0173 * q_thermogenesis_total * (5.87 - p_a)  # Latent heat loss
    return res_sh, res_lh
