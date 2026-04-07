from numpy._typing import NDArray
from numpy import float64
import numpy as np
from pythermalcomfort.classes_return import MRT
from pythermalcomfort.classes_input import ForthPowerInputs, AreaWeightedInputs, RayMenInputs
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
    """Area weighetr method detail here..."""
    surface_temps: NDArray[np.float64] = np.asarray(p.surface_temps, dtype=np.float64)
    surface_areas: NDArray[np.float64] = np.asarray(p.surface_areas, dtype=np.float64)

    if p.units == "IP":
        surface_temps: NDArray[float64] = np.asarray(units_converter(tdb=surface_temps), dtype=np.float64)

    angle_factors = surface_areas / surface_areas.sum()

    mrt: float = np.sum((surface_temps + 273.15) ** 4 * angle_factors) ** 0.25 - 273.15

    if p.units == "IP":
        mrt: float = units_converter(from_units="SI", tmp=mrt)[0]

    return MRT(mrt=mrt)

def _ray_men(p: RayMenInputs) -> MRT:
    """Ray Men method detail here..."""
    boltzman = 5.67e-8 # do we have this stefan Boltzman constant defined somewhere?

    # 1. Geometry: We use sol_altitude directly now
    alt_rad = np.radians(p.sol_altitude)
    sin_alt = np.sin(alt_rad) # sin(alt) is mathematically cos(zenith)

    # 2. Projected area factor (Fanger approximation)
    fp = 0.308 * np.cos(alt_rad * (0.998 - p.sol_altitude**2 / 50000))

    # 3. Shortwave fluxes 
    K_direct = fp * p.dni
    K_diffuse = p.svf * p.dhi
    # Using sin_alt here because that's the horizontal projection component
    K_reflected = p.albedo_ground * (1 - p.svf) * (p.dni * sin_alt + p.dhi)

    K_total = K_direct + K_diffuse + K_reflected

    # 4. Longwave fluxes
    T_surr_K = p.t_surr + 273.15
    L_total = boltzman * T_surr_K**4 

    # 5. MRT inversion
    mrt_K = ((1 / boltzman) * (
        (p.alpha_k / p.emissivity) * K_total + L_total
    )) ** 0.25

    mrt_C = mrt_K - 273.15
    return MRT(mrt=mrt_C)

def _nwp_based() -> None: #EMWPF Implementtion, from thermofeel
    ...


def mrt(params: ForthPowerInputs | AreaWeightedInputs | RayMenInputs) -> MRT:
    """Main docstring."""
    match params:
        case ForthPowerInputs():
            return _forth_power(p=params)
        case AreaWeightedInputs():
            return _area_weighted(p=params)
        case RayMenInputs():
            return _ray_men(p=params)
        case _:
            raise ValueError(f"Unknown method '{type(params)}'")