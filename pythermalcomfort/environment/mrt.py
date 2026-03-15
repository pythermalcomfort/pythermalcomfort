from numpy._typing import NDArray
from numpy import float64
import numpy as np
from pythermalcomfort.classes_return import MRT
from pythermalcomfort.classes_input import ForthPowerInputs, AreaWeightedInputs
from pythermalcomfort.utilities import units_converter

def _forth_power(p: ForthPowerInputs) -> MRT:
    """Fourth power method detail here..."""
    surface_temps: NDArray[np.float64] = np.asarray(p.surface_temps, dtype=np.float64)
    angle_factors: NDArray[np.float64] = np.asarray(p.angle_factors, dtype=np.float64)

    if p.units == "IP":
        surface_temps: NDArray[float64] = np.asarray(units_converter(tdb=surface_temps), dtype=np.float64)

    mrt: float64 = np.sum((surface_temps + 273.15) ** 4 * angle_factors) ** 0.25 - 273.15

    if p.units == "IP":
        mrt: float = units_converter(from_units="SI", tmp=mrt)[0]

    return MRT(mrt=mrt)

def _area_weighted(p: AreaWeightedInputs) -> MRT:
    surface_temps: NDArray[np.float64] = np.asarray(p.surface_temps, dtype=np.float64)
    surface_areas: NDArray[np.float64] = np.asarray(p.surface_areas, dtype=np.float64)

    if p.units == "IP":
        surface_temps: NDArray[float64] = np.asarray(units_converter(tdb=surface_temps), dtype=np.float64)

    angle_factors = surface_areas / surface_areas.sum()

    mrt: float = np.sum((surface_temps + 273.15) ** 4 * angle_factors) ** 0.25 - 273.15

    if p.units == "IP":
        mrt: float = units_converter(from_units="SI", tmp=mrt)[0]

    return MRT(mrt=mrt)

def _ray_men() -> None: # Established Name
    ...

def _nwp_based() -> None: #EMWPF Implementtion, from thermofeel
    ...


def mrt(params: ForthPowerInputs | AreaWeightedInputs) -> MRT:
    """Main docstring."""
    match params:
        case ForthPowerInputs():
            return _forth_power(p=params)
        case AreaWeightedInputs():
            return _area_weighted(p=params)
        case _:
            raise ValueError(f"Unknown method '{type(params)}'")