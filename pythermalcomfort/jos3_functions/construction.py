"""This module provides models for calculating various body parameters including surface
area, weight ratio, basal blood flow ratio, thermal conductance, and thermal capacity.

The values of a NumPy array containing 17 elements correspond to the body parts defined
in JOS3BodyParts.
"""

import numpy as np

from pythermalcomfort.classes_return import JOS3BodyParts
from pythermalcomfort.jos3_functions.matrix import IDICT, NUM_NODES
from pythermalcomfort.jos3_functions.parameters import Default
from pythermalcomfort.utilities import body_surface_area


def validate_body_parameters(
    height: float,
    weight: float,
    age: int,
    body_fat: float,
):
    """Validate the body parameters: height, weight, age, and body fat percentage.

    Parameters
    ----------
    height : float, optional
        The height of the person in meters.
    weight : float, optional
        The weight of the person in kilograms.
    age : int, optional
        The age of the person in years.
    body_fat : float, optional
        The body fat percentage as a fraction of total body mass.

    Raises
    ------
    ValueError
        If any of the parameters are out of the specified range.

    Examples
    --------
    Validate parameters for a person with height 1.80 meters, weight 75 kilograms, age 30 years, and body fat 20%:

    .. code-block:: python

        validate_body_parameters(height=1.80, weight=75, age=30, body_fat=20)

    """
    if not (0.5 <= height <= 3.0):
        raise ValueError("Height must be in the range [0.5, 3.0] meters.")

    if not (20.0 <= weight <= 200.0):
        raise ValueError("Weight must be in the range [20.0, 200.0] kilograms.")

    if not (5 <= age <= 100):
        raise ValueError("Age must be in the range [5, 100] years.")

    if not (1 <= body_fat <= 90):
        raise ValueError(
            "Body fat percentage must be in the range [1, 90] (1% to 90%).",
        )


def to_array_body_parts(inp) -> np.ndarray:
    """Create a NumPy array of shape (17,) with the given input.

    Parameters
    ----------
    inp : int, float, dict, list, ndarray
        The value(s) to use when creating the 17-element array.
        Supports int, float, dict (with BODY_NAMES as keys), list, and ndarray types.

    Returns
    -------
    ndarray
        A NumPy array of shape (17,) with the specified values.

    Raises
    ------
    ValueError
        If the input type is not supported or if the input list or ndarray is not of length 17.
    """
    if isinstance(inp, int | float):
        return np.full(Default.num_body_parts, inp)
    if isinstance(inp, dict):
        return np.asarray([inp[key] for key in JOS3BodyParts.get_attribute_names()])
    if isinstance(inp, list | np.ndarray):
        inp = np.asarray(inp)
        if inp.shape == (Default.num_body_parts,):
            return inp.copy()
        error_msg = (
            f"The input list or ndarray is not of length {Default.num_body_parts}"
        )
        raise ValueError(error_msg)
    error_msg = f"Unsupported input type: {type(inp)}. Supported types: int, float, list, dict, ndarray"
    raise ValueError(error_msg)


def bsa_rate(
    height: float,
    weight: float,
    bsa_equation: str,
) -> float | np.ndarray:
    """Calculate the ratio of body surface area (BSA) to the standard body.

    This function computes the ratio of an individual's body surface area to that of a standard body,
    based on the given height, weight, and BSA calculation equation.

    Parameters
    ----------
    height : float, optional
        The height of the person in meters.
    weight : float, optional
        The weight of the person in kilograms.
    bsa_equation : str, optional
        The equation name for BSA calculation. Choose one from BodySurfaceAreaEquations

    Returns
    -------
    float, np.ndarray
        The ratio of the individual's BSA to the standard body BSA.

    Raises
    ------
    ValueError
        If the parameters are out of the specified range.

    Examples
    --------
    Calculate the BSA ratio for a person with height 1.80 meters and weight 75 kilograms:

    .. code-block:: python

        bsa_ratio = bsa_rate(height=1.80, weight=75, bsa_equation="dubois")
        print(bsa_ratio)
    """
    bsa_all = body_surface_area(
        height=height,
        weight=weight,
        formula=bsa_equation,
    )
    return (
        bsa_all / Default.local_bsa.sum()
    )  # The bsa ratio to the standard body (1.87m2)


def local_bsa(
    height: float,
    weight: float,
    bsa_equation: str,
) -> np.ndarray:
    """Calculate local body surface area (BSA) in square meters.

    The local body surface area has been derived from 65MN.
    The head has been divided into head and neck based on Smith's model.
        head = 0.1396 * 0.1117 / 0.1414 (65MN_Head * Smith_Head / Smith_Head+neck)
        neck = 0.1396 * 0.0297 / 0.1414 (65MN_Head * Smith_Neck / Smith_Head+neck)

    Parameters
    ----------
    height : float, optional
        Body height in meters.
    weight : float, optional
        Body weight in kilograms.
    bsa_equation : str, optional
        The equation name for BSA calculation. Choose one from BodySurfaceAreaEquations.

    Returns
    -------
    np.ndarray
        Local body surface area (BSA) in square meters.

    Raises
    ------
    ValueError
        If any of the parameters are out of the specified range.

    Examples
    --------
    Calculate the local BSA for a person with height 1.80 meters and weight 75 kilograms:

    .. code-block:: python

        local_bsa_values = local_bsa(height=1.80, weight=75, bsa_equation="dubois")
        print(local_bsa_values)
    """
    bsa_ratio = bsa_rate(height=height, weight=weight, bsa_equation=bsa_equation)
    return Default.local_bsa * bsa_ratio


def weight_rate(weight: float) -> float:
    """Calculate the ratio of the body weight to the standard body.

    Parameters
    ----------
    weight : float, optional
        The body weight in kilograms.

    Returns
    -------
    float
        The ratio of the body weight to the standard body.
    """
    return weight / Default.weight


def bfb_rate(
    height: float,
    weight: float,
    bsa_equation: str,
    age: int,
    ci: float,
) -> float:
    """Calculate the ratio of basal blood flow (BFB) to the standard body.

    This function computes the basal blood flow rate based on the given height, weight,
    BSA calculation equation, age, and cardiac index.

    Parameters
    ----------
    height : float, optional
        Body height in meters.
    weight : float, optional
        Body weight in kilograms.
    bsa_equation : str, optional
        The equation name for BSA calculation. Choose one from BodySurfaceAreaEquations.
    age : int, optional
        Age in years.
    ci : float, optional
        Cardiac index in L/min/m².

    Returns
    -------
    float
        Basal blood flow rate.

    Raises
    ------
    ValueError
        If any of the parameters are out of the specified range.

    Examples
    --------
    Calculate the BFB rate for a person with height 1.80 meters, weight 75 kilograms, age 30 years, and cardiac index 2.59:

    .. code-block:: python

        bfb_ratio = bfb_rate(height=1.80, weight=75, age=30, ci=2.59)
        print(bfb_ratio)
    """
    ci *= 60  # Convert unit from L/min/m² to L/h/m²

    # Adjust cardiac index based on age
    if age < 50:
        ci *= 1
    elif age < 60:
        ci *= 0.85
    elif age < 70:
        ci *= 0.75
    else:  # age >= 70
        ci *= 0.7

    bsa_ratio = bsa_rate(height=height, weight=weight, bsa_equation=bsa_equation)
    bfb_all = ci * bsa_ratio * Default.local_bsa.sum()  # Total BFB in L/h
    return bfb_all / Default.blood_flow_rate  # Ratio to the standard body (290 L/h)


def conductance(
    height: float,
    weight: float,
    bsa_equation: str,
    fat: float,
):
    """Calculate thermal conductance between layers [W/K].

    Parameters
    ----------
    height : float, optional
        Body height [m].
    weight : float, optional
        Body weight [kg].
    bsa_equation : str, optional
        The equation name (str) of bsa calculation. Choose one from BodySurfaceAreaEquations.
    fat : float, optional
        Body fat rate [%].

    Returns
    -------
    conductance : numpy.ndarray
        Thermal conductance between layers [W/K].
        The shape is (NUM_NODES, NUM_NODES).
    """
    if fat < 12.5:
        cdt_cr_sk = np.asarray(
            [
                1.341,
                0.930,
                1.879,
                1.729,
                2.370,
                1.557,
                1.018,
                2.210,
                1.557,
                1.018,
                2.210,
                2.565,
                1.378,
                3.404,
                2.565,
                1.378,
                3.404,
            ],
        )
    elif fat < 17.5:
        cdt_cr_sk = np.asarray(
            [
                1.311,
                0.909,
                1.785,
                1.643,
                2.251,
                1.501,
                0.982,
                2.183,
                1.501,
                0.982,
                2.183,
                2.468,
                1.326,
                3.370,
                2.468,
                1.326,
                3.370,
            ],
        )
    elif fat < 22.5:
        cdt_cr_sk = np.asarray(
            [
                1.282,
                0.889,
                1.698,
                1.563,
                2.142,
                1.448,
                0.947,
                2.156,
                1.448,
                0.947,
                2.156,
                2.375,
                1.276,
                3.337,
                2.375,
                1.276,
                3.337,
            ],
        )
    elif fat < 27.5:
        cdt_cr_sk = np.asarray(
            [
                1.255,
                0.870,
                1.618,
                1.488,
                2.040,
                1.396,
                0.913,
                2.130,
                1.396,
                0.913,
                2.130,
                2.285,
                1.227,
                3.304,
                2.285,
                1.227,
                3.304,
            ],
        )
    else:  # fat >= 27.5
        cdt_cr_sk = np.asarray(
            [
                1.227,
                0.852,
                1.542,
                1.419,
                1.945,
                1.346,
                0.880,
                1.945,
                1.346,
                0.880,
                1.945,
                2.198,
                1.181,
                3.271,
                2.198,
                1.181,
                3.271,
            ],
        )

    cdt_cr_ms = np.zeros(Default.num_body_parts)  # core to muscle [W/K]
    cdt_ms_fat = np.zeros(Default.num_body_parts)  # muscle to fat [W/K]
    cdt_fat_sk = np.zeros(Default.num_body_parts)  # fat to skin [W/K]

    # head and pelvis consists of 65MN's conductances
    cdt_cr_ms[0] = 1.601  # head
    cdt_ms_fat[0] = 13.222
    cdt_fat_sk[0] = 16.008
    cdt_cr_ms[4] = 3.0813  # pelvis
    cdt_ms_fat[4] = 10.3738
    cdt_fat_sk[4] = 41.4954

    # vessel to core
    # The shape is a cylinder.
    # It is assumed that the inner is vascular radius, 2.5mm and the outer is
    # stolwijk's core radius.
    # The heat transfer coefficient of the core is assumed as the Michel's
    # counter-flow model 0.66816 [W/(m･K)].
    cdt_ves_cr = np.asarray(
        [
            0,
            0,
            0,
            0,
            0,
            0.586,
            0.383,
            1.534,
            0.586,
            0.383,
            1.534,
            0.810,
            0.435,
            1.816,
            0.810,
            0.435,
            1.816,
        ],
    )
    # superficial vein to skin
    cdt_sfv_sk = np.asarray(
        [
            0,
            0,
            0,
            0,
            0,
            57.735,
            37.768,
            16.634,
            57.735,
            37.768,
            16.634,
            102.012,
            54.784,
            24.277,
            102.012,
            54.784,
            24.277,
        ],
    )

    # art to vein (counter-flow) [W/K]
    # The data has been derived Mitchell's model.
    # The values = 15.869 [W/(m･K)] * the segment length [m]
    cdt_art_vein = np.asarray(
        [
            0,
            0,
            0,
            0,
            0,
            0.537,
            0.351,
            0.762,
            0.537,
            0.351,
            0.762,
            0.826,
            0.444,
            0.992,
            0.826,
            0.444,
            0.992,
        ],
    )

    # Changes values by body size based on the standard body.
    wr = weight_rate(weight)
    bsar = bsa_rate(height=height, weight=weight, bsa_equation=bsa_equation)
    # head, neck (Sphere shape)
    # TODO we are multiplying zeros by a value
    cdt_cr_sk[:2] *= wr / bsar
    cdt_cr_ms[:2] *= wr / bsar
    cdt_ms_fat[:2] *= wr / bsar
    cdt_fat_sk[:2] *= wr / bsar
    cdt_ves_cr[:2] *= wr / bsar
    cdt_sfv_sk[:2] *= wr / bsar
    cdt_art_vein[:2] *= wr / bsar
    # Others (Cylinder shape)
    cdt_cr_sk[2:] *= bsar**2 / wr
    cdt_cr_ms[2:] *= bsar**2 / wr
    cdt_ms_fat[2:] *= bsar**2 / wr
    cdt_fat_sk[2:] *= bsar**2 / wr
    cdt_ves_cr[2:] *= bsar**2 / wr
    cdt_sfv_sk[2:] *= bsar**2 / wr
    cdt_art_vein[2:] *= bsar**2 / wr

    cdt_whole = np.zeros((NUM_NODES, NUM_NODES))
    for i, bn in enumerate(JOS3BodyParts.get_attribute_names()):
        # Dictionary of indices in each body segment
        # key = layer name, value = index of matrix
        index_of = IDICT[bn]

        # Common
        cdt_whole[index_of["artery"], index_of["vein"]] = cdt_art_vein[i]  # art to vein
        cdt_whole[index_of["artery"], index_of["core"]] = cdt_ves_cr[i]  # art to cr
        cdt_whole[index_of["vein"], index_of["core"]] = cdt_ves_cr[i]  # vein to cr

        # Only limbs
        if i >= 5:
            cdt_whole[index_of["sfvein"], index_of["skin"]] = cdt_sfv_sk[i]  # sfv to sk

        # If the segment has a muscle or fat layer
        if index_of["muscle"] is not None:  # or not indexof["fat"] is None
            cdt_whole[index_of["core"], index_of["muscle"]] = cdt_cr_ms[i]  # cr to ms
            cdt_whole[index_of["muscle"], index_of["fat"]] = cdt_ms_fat[i]  # ms to fat
            cdt_whole[index_of["fat"], index_of["skin"]] = cdt_fat_sk[i]  # fat to sk

        else:
            cdt_whole[index_of["core"], index_of["skin"]] = cdt_cr_sk[i]  # cr to sk

    # Creates a symmetrical matrix
    cdt_whole = cdt_whole + cdt_whole.T

    return cdt_whole.copy()


def capacity(
    height: float,
    weight: float,
    bsa_equation: str,
    age: int,
    ci: float,
) -> np.ndarray:
    """Calculate the thermal capacity [J/K].

    The values of vascular and central blood capacity have been derived from
    Yokoyama's model.
    The specific heat of blood is assumed as 1.0 [kcal/L.K].

    Parameters
    ----------
    height : float, optional
        Body height [m].
    weight : float, optional
        Body weight [kg].
    bsa_equation : str, optional
        The equation name (str) of bsa calculation. Choose one from BodySurfaceAreaEquations.
    age : int, optional
        Age [years].
    ci : float, optional
        Cardiac index [L/min/㎡].

    Returns
    -------
    np.ndarray
        Thermal capacity [J/K].
        The shape is (NUM_NODES).
    """
    # Define capacities [Wh/K]

    # artery [Wh/K]
    cap_art = np.asarray(
        [
            0.096,
            0.025,
            0.12,
            0.111,
            0.265,
            0.0186,
            0.0091,
            0.0044,
            0.0186,
            0.0091,
            0.0044,
            0.0813,
            0.04,
            0.0103,
            0.0813,
            0.04,
            0.0103,
        ],
    )

    # vein [Wh/K]
    cap_vein = np.asarray(
        [
            0.321,
            0.085,
            0.424,
            0.39,
            0.832,
            0.046,
            0.024,
            0.01,
            0.046,
            0.024,
            0.01,
            0.207,
            0.1,
            0.024,
            0.207,
            0.1,
            0.024,
        ],
    )

    # superficial vein [Wh/K]
    cap_sfv = np.asarray(
        [
            0,
            0,
            0,
            0,
            0,
            0.025,
            0.015,
            0.011,
            0.025,
            0.015,
            0.011,
            0.074,
            0.05,
            0.021,
            0.074,
            0.05,
            0.021,
        ],
    )

    # central blood [Wh/K]
    cap_cb = 1.999

    # core [Wh/K]
    cap_cr = np.asarray(
        [
            1.7229,
            0.564,
            10.2975,
            9.3935,
            4.488,
            1.6994,
            1.1209,
            0.1536,
            1.6994,
            1.1209,
            0.1536,
            5.3117,
            2.867,
            0.2097,
            5.3117,
            2.867,
            0.2097,
        ],
    )

    # muscle [Wh/K]
    cap_ms = np.asarray(
        [
            0.305,
            0.0,
            0.0,
            0.0,
            7.409,
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

    # fat [Wh/K]
    cap_fat = np.asarray(
        [
            0.203,
            0.0,
            0.0,
            0.0,
            1.947,
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

    # skin [Wh/K]
    cap_sk = np.asarray(
        [
            0.1885,
            0.058,
            0.441,
            0.406,
            0.556,
            0.126,
            0.084,
            0.088,
            0.126,
            0.084,
            0.088,
            0.334,
            0.169,
            0.107,
            0.334,
            0.169,
            0.107,
        ],
    )

    # Adjust capacities based on body parameters
    bfbr = bfb_rate(
        height=height, weight=weight, bsa_equation=bsa_equation, age=age, ci=ci
    )
    wr = weight_rate(weight)
    cap_art *= bfbr
    cap_vein *= bfbr
    cap_sfv *= bfbr
    cap_cb *= bfbr
    cap_cr *= wr
    cap_ms *= wr
    cap_fat *= wr
    cap_sk *= wr

    # Initialize capacity array
    cap_whole = np.zeros(NUM_NODES)
    cap_whole[0] = cap_cb

    for i, bn in enumerate(JOS3BodyParts.get_attribute_names()):
        # Dictionary of indices in each body segment key = layer name, value = index of matrix
        index_of = IDICT[bn]

        # Common
        cap_whole[index_of["artery"]] = cap_art[i]
        cap_whole[index_of["vein"]] = cap_vein[i]
        cap_whole[index_of["core"]] = cap_cr[i]
        cap_whole[index_of["skin"]] = cap_sk[i]

        # Only limbs
        if i >= 5:
            cap_whole[index_of["sfvein"]] = cap_sfv[i]

        # If the segment has a muscle or fat layer
        if index_of["muscle"] is not None:  # or not indexof["fat"] is None
            cap_whole[index_of["muscle"]] = cap_ms[i]
            cap_whole[index_of["fat"]] = cap_fat[i]

    cap_whole *= 3600  # Convert [Wh/K] to [J/K]
    return cap_whole


def pass_values_to_jos3_body_parts(values, round_digits=2, body_parts=None):
    if body_parts is None:
        body_parts = JOS3BodyParts.get_attribute_names()
    return JOS3BodyParts(
        **{
            name: value
            for name, value in zip(
                body_parts, np.round(values, round_digits), strict=False
            )
        },
    )
