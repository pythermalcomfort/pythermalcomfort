from __future__ import annotations

import importlib
import warnings
from enum import Enum
from functools import wraps
from typing import NamedTuple

import numpy as np

NumericInput = float | int | np.ndarray | list

c_to_k = 273.15
cp_vapour = 1805.0
cp_water = 4186
cp_air = 1004
h_fg = 2501000
r_air = 287.055
g = 9.81  # m/s2
met_to_w_m2 = 58.15


class Models(Enum):
    """Models options."""

    ashrae_55_2023 = "55-2023"
    iso_7730_2005 = "7730-2005"
    iso_7730_2025 = "7730-2025"
    iso_9920_2007 = "9920-2007"
    iso_7933_2004 = "7933-2004"
    iso_7933_2023 = "7933-2023"


class Units(Enum):
    """Units options."""

    SI = "SI"
    IP = "IP"


class Sex(Enum):
    """Sex options."""

    male = "male"
    female = "female"


class Postures(Enum):
    """Postures options."""

    standing = "standing"
    sitting = "sitting"
    sedentary = "sedentary"
    reclining = "reclining"
    lying = "lying"
    supine = "supine"
    crouching = "crouching"


class BodySurfaceAreaEquations(Enum):
    """Body Surface Area Equations."""

    dubois = "dubois"
    takahira = "takahira"
    fujimoto = "fujimoto"
    kurazumi = "kurazumi"


def body_surface_area(
    weight: float,
    height: float,
    formula: str = BodySurfaceAreaEquations.dubois.value,
) -> float:
    """Calculate the body surface area (BSA) in square meters.

    Parameters
    ----------
    weight : float
        Body weight in kilograms.
    height : float
        Body height in meters.
    formula : str, optional
        Formula used to calculate the body surface area. Default is "dubois".
        Choose one from BodySurfaceAreaEquations.

    Returns
    -------
    float
        Body surface area in square meters.

    Raises
    ------
    ValueError
        If the specified formula is not recognized.

    Examples
    --------
    Calculate BSA using the DuBois formula:

    .. code-block:: python

        bsa = body_surface_area(weight=70, height=1.75, formula="dubois")
        print(bsa)
    """
    if formula == BodySurfaceAreaEquations.dubois.value:
        return 0.202 * (weight**0.425) * (height**0.725)
    if formula == BodySurfaceAreaEquations.takahira.value:
        return 0.2042 * (weight**0.425) * (height**0.725)
    if formula == BodySurfaceAreaEquations.fujimoto.value:
        return 0.1882 * (weight**0.444) * (height**0.663)
    if formula == BodySurfaceAreaEquations.kurazumi.value:
        return 0.2440 * (weight**0.383) * (height**0.693)
    invalid_formula_msg = (
        f"Formula '{formula}' for calculating body surface area is not recognized."
    )
    raise ValueError(invalid_formula_msg)


def units_converter(from_units=Units.IP.value, **kwargs) -> list[float]:
    """Convert IP values to SI units.

    Parameters
    ----------
    from_units: str
        specify system to convert from
    **kwargs : [t, v]

    Returns
    -------
    converted values in SI units
    """
    results = []
    from_units = from_units.upper()
    if from_units == Units.IP.value:
        for key, value in kwargs.items():
            if "tmp" in key or key == "tr" or key == "tdb":
                results.append((value - 32) * 5 / 9)
            if key in ["v", "vr", "vel"]:
                results.append(value / 3.281)
            if key == "area":
                results.append(value / 10.764)
            if key == "pressure":
                results.append(value * 101325)

    elif from_units == Units.SI.value:
        for key, value in kwargs.items():
            if "tmp" in key or key == "tr" or key == "tdb":
                results.append((value * 9 / 5) + 32)
            if key in ["v", "vr", "vel"]:
                results.append(value * 3.281)
            if key == "area":
                results.append(value * 10.764)
            if key == "pressure":
                results.append(value / 101325)

    return results


#: Met values of typical tasks.
met_typical_tasks = {
    "Sleeping": 0.7,
    "Reclining": 0.8,
    "Seated, quiet": 1.0,
    "Reading, seated": 1.0,
    "Writing": 1.0,
    "Typing": 1.1,
    "Standing, relaxed": 1.2,
    "Filing, seated": 1.2,
    "Flying aircraft, routine": 1.2,
    "Filing, standing": 1.4,
    "Driving a car": 1.5,
    "Walking about": 1.7,
    "Cooking": 1.8,
    "Table sawing": 1.8,
    "Walking 2mph (3.2kmh)": 2.0,
    "Lifting/packing": 2.1,
    "Seated, heavy limb movement": 2.2,
    "Light machine work": 2.2,
    "Flying aircraft, combat": 2.4,
    "Walking 3mph (4.8kmh)": 2.6,
    "House cleaning": 2.7,
    "Driving, heavy vehicle": 3.2,
    "Dancing": 3.4,
    "Calisthenics": 3.5,
    "Walking 4mph (6.4kmh)": 3.8,
    "Tennis": 3.8,
    "Heavy machine work": 4.0,
    "Handling 100lb (45 kg) bags": 4.0,
    "Pick and shovel work": 4.4,
    "Basketball": 6.3,
    "Wrestling": 7.8,
}

#: Total clothing insulation of typical ensembles.
clo_typical_ensembles = {
    "Walking shorts, short-sleeve shirt": 0.36,
    "Typical summer indoor clothing": 0.5,
    "Knee-length skirt, short-sleeve shirt, sandals, underwear": 0.54,
    "Trousers, short-sleeve shirt, socks, shoes, underwear": 0.57,
    "Trousers, long-sleeve shirt": 0.61,
    "Knee-length skirt, long-sleeve shirt, full slip": 0.67,
    "Sweat pants, long-sleeve sweatshirt": 0.74,
    "Jacket, Trousers, long-sleeve shirt": 0.96,
    "Typical winter indoor clothing": 1.0,
}

#: Clo values of individual clothing elements. To calculate the total
#: clothing insulation you need to add these values together.
clo_individual_garments = {
    "Metal chair": 0.00,
    "Bra": 0.01,
    "Wooden stool": 0.01,
    "Ankle socks": 0.02,
    "Shoes or sandals": 0.02,
    "Slippers": 0.03,
    "Panty hose": 0.02,
    "Calf length socks": 0.03,
    "Women's underwear": 0.03,
    "Men's underwear": 0.04,
    "Knee socks (thick)": 0.06,
    "Short shorts": 0.06,
    "Walking shorts": 0.08,
    "T-shirt": 0.08,
    "Standard office chair": 0.10,
    "Executive chair": 0.15,
    "Boots": 0.1,
    "Sleeveless scoop-neck blouse": 0.12,
    "Half slip": 0.14,
    "Long underwear bottoms": 0.15,
    "Full slip": 0.16,
    "Short-sleeve knit shirt": 0.17,
    "Sleeveless vest (thin)": 0.1,
    "Sleeveless vest (thick)": 0.17,
    "Sleeveless short gown (thin)": 0.18,
    "Short-sleeve dress shirt": 0.19,
    "Sleeveless long gown (thin)": 0.2,
    "Long underwear top": 0.2,
    "Thick skirt": 0.23,
    "Long-sleeve dress shirt": 0.25,
    "Long-sleeve flannel shirt": 0.34,
    "Long-sleeve sweat shirt": 0.34,
    "Short-sleeve hospital gown": 0.31,
    "Short-sleeve short robe (thin)": 0.34,
    "Short-sleeve pajamas": 0.42,
    "Long-sleeve long gown": 0.46,
    "Long-sleeve short wrap robe (thick)": 0.48,
    "Long-sleeve pajamas (thick)": 0.57,
    "Long-sleeve long wrap robe (thick)": 0.69,
    "Thin trousers": 0.15,
    "Thick trousers": 0.24,
    "Sweatpants": 0.28,
    "Overalls": 0.30,
    "Coveralls": 0.49,
    "Thin skirt": 0.14,
    "Long-sleeve shirt dress (thin)": 0.33,
    "Long-sleeve shirt dress (thick)": 0.47,
    "Short-sleeve shirt dress": 0.29,
    "Sleeveless, scoop-neck shirt (thin)": 0.23,
    "Sleeveless, scoop-neck shirt (thick)": 0.27,
    "Long sleeve shirt (thin)": 0.25,
    "Long sleeve shirt (thick)": 0.36,
    "Single-breasted coat (thin)": 0.36,
    "Single-breasted coat (thick)": 0.44,
    "Double-breasted coat (thin)": 0.42,
    "Double-breasted coat (thick)": 0.48,
}

#: This dictionary contains the reflection coefficients, Fr, for different
#: special materials
f_r_garments = {
    "Cotton with aluminium paint": 0.42,
    "Viscose with glossy aluminium foil": 0.19,
    "Aramid (Kevlar) with glossy aluminium foil": 0.14,
    "Wool with glossy aluminium foil": 0.12,
    "Cotton with glossy aluminium foil": 0.04,
    "Viscose vacuum metallized with aluminium": 0.06,
    "Aramid vacuum metallized with aluminium": 0.04,
    "Wool vacuum metallized with aluminium": 0.05,
    "Cotton vacuum metallized with aluminium": 0.05,
    "Glass fiber vacuum metallized with aluminium": 0.07,
}


class DefaultSkinTemperature(NamedTuple):
    """Default skin temperature in degree Celsius for 17 local body parts.

    The data comes from Hui Zhang's experiments
    https://escholarship.org/uc/item/3f4599hx
    """

    head: float = 35.3
    neck: float = 35.6
    chest: float = 35.1
    back: float = 35.3
    pelvis: float = 35.3
    left_shoulder: float = 34.2
    left_arm: float = 34.6
    left_hand: float = 34.4
    right_shoulder: float = 34.2
    right_arm: float = 34.6
    right_hand: float = 34.4
    left_thigh: float = 34.3
    left_leg: float = 32.8
    left_foot: float = 33.3
    right_thigh: float = 34.3
    right_leg: float = 32.8
    right_foot: float = 33.3


def _deprecated_utility(function_name: str, new_module: str):
    """Create a compatibility wrapper for a utility moved to a public package."""
    module = importlib.import_module(f"pythermalcomfort.{new_module}")
    target = getattr(module, function_name)

    @wraps(target)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"pythermalcomfort.utilities.{function_name} is deprecated; "
            f"import it from pythermalcomfort.{new_module} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return target(*args, **kwargs)

    wrapper.__doc__ = (
        f"Deprecated alias for pythermalcomfort.{new_module}.{function_name}(). "
        "Import from there instead; this path will be removed after two minor releases."
    )
    return wrapper


_MOVED_PUBLIC_FUNCTIONS = {
    "mean_radiant_tmp": "environment",
    "operative_tmp": "environment",
    "running_mean_outdoor_temperature": "environment",
    "transpose_sharp_altitude": "environment",
    "f_svv": "environment",
    "v_relative": "environment",
    "p_sat": "psychrometrics",
    "p_sat_torr": "psychrometrics",
    "antoine": "psychrometrics",
    "psy_ta_rh": "psychrometrics",
    "hr_to_rh": "psychrometrics",
    "wet_bulb_tmp": "psychrometrics",
    "dew_point_tmp": "psychrometrics",
    "enthalpy_air": "psychrometrics",
    "clo_dynamic_ashrae": "clothing",
    "clo_dynamic_iso": "clothing",
    "clo_intrinsic_insulation_ensemble": "clothing",
    "clo_area_factor": "clothing",
    "clo_insulation_air_layer": "clothing",
    "clo_total_insulation": "clothing",
    "clo_correction_factor_environment": "clothing",
}


def __getattr__(name: str):
    """Create moved-public-function shims lazily to avoid circular imports."""
    try:
        new_module = _MOVED_PUBLIC_FUNCTIONS[name]
    except KeyError as error:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from error

    wrapper = _deprecated_utility(name, new_module)
    globals()[name] = wrapper
    return wrapper


def __dir__() -> list[str]:
    """Include lazily created compatibility names in module introspection."""
    return sorted(set(globals()) | set(_MOVED_PUBLIC_FUNCTIONS))
