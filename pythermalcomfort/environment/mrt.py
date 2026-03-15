from numpy._typing import NDArray
import numpy as np
from pythermalcomfort.classes_return import MRT
from pythermalcomfort.classes_input import ForthPowerInputs, NoGeometryInputs

# ── Helpers ─────────────────────────────────────────────────────────

def _f_to_c(value: NDArray[np.float64]) -> NDArray[np.float64]:
    return (value - 32) * 5 / 9

def _c_to_f(value: NDArray[np.float64]) -> NDArray[np.float64]:
    return value * 9 / 5 + 32

# ── Method implementations ───────────────────────────────────────────

def _forth_power(p: ForthPowerInputs) -> MRT:
    surface_temps: NDArray[np.float64] = np.asarray(p.surface_temps, dtype=np.float64)
    angle_factors: NDArray[np.float64] = np.asarray(p.angle_factors, dtype=np.float64)
    if p.units == "IP":
        surface_temps = _f_to_c(surface_temps)
    mrt = np.sum((surface_temps + 273.15) ** 4 * angle_factors) ** 0.25 - 273.15
    if p.units == "IP":
        mrt = _c_to_f(mrt)
    return MRT(mrt=mrt)

def _no_geometry(p: NoGeometryInputs) -> MRT:
    tdb: NDArray[np.float64] = np.asarray(p.tdb, dtype=np.float64)
    tr: NDArray[np.float64] = np.asarray(p.tr, dtype=np.float64)
    if p.units == "IP":
        tdb = _f_to_c(tdb)
        tr = _f_to_c(tr)
    mrt = p.asw * tdb + (1 - p.asw) * tr
    if p.units == "IP":
        mrt = _c_to_f(mrt)
    return MRT(mrt=mrt)

# ── Dispatcher ───────────────────────────────────────────────────────

def mrt(params: ForthPowerInputs | NoGeometryInputs) -> MRT:
    match params:
        case ForthPowerInputs():
            return _forth_power(params)
        case NoGeometryInputs():
            return _no_geometry(params)
        case _:
            raise ValueError(f"Unknown method '{type(params)}'")