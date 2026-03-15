from numpy._typing import NDArray
from numpy import float64
import numpy as np
from pythermalcomfort.classes_return import MRT
from pythermalcomfort.classes_input import ForthPowerInputs, NoGeometryInputs
from pythermalcomfort.utilities import units_converter

def _forth_power(p: ForthPowerInputs) -> MRT:
    surface_temps: NDArray[np.float64] = np.asarray(p.surface_temps, dtype=np.float64)
    angle_factors: NDArray[np.float64] = np.asarray(p.angle_factors, dtype=np.float64)

    if p.units == "IP":
        surface_temps: NDArray[float64] = np.asarray(units_converter(tdb=surface_temps), dtype=np.float64)

    mrt: float64 = np.sum((surface_temps + 273.15) ** 4 * angle_factors) ** 0.25 - 273.15

    if p.units == "IP":
        mrt: float = units_converter(from_units="SI", tmp=mrt)[0]

    return MRT(mrt=mrt)

def _no_geometry(p: NoGeometryInputs) -> MRT:
    tdb: NDArray[np.float64] = np.asarray(p.tdb, dtype=np.float64)
    tr: NDArray[np.float64] = np.asarray(p.tr, dtype=np.float64)

    if p.units == "IP":
        tdb: NDArray[float64] = np.asarray(units_converter(tdb=tdb), dtype=np.float64)
        tr: NDArray[float64] = np.asarray(units_converter(tr=tr), dtype=np.float64)

    mrt: float = p.asw * tdb + (1 - p.asw) * tr

    if p.units == "IP":
        mrt: float = units_converter(from_units="SI", tmp=mrt)[0]

    return MRT(mrt=mrt)

def mrt(params: ForthPowerInputs | NoGeometryInputs) -> MRT:
    match params:
        case ForthPowerInputs():
            return _forth_power(p=params)
        case NoGeometryInputs():
            return _no_geometry(p=params)
        case _:
            raise ValueError(f"Unknown method '{type(params)}'")