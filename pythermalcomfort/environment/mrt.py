from numpy import float64
from numpy._typing._array_like import NDArray
from dataclasses import dataclass
from typing import Any, Literal
import numpy as np

from pythermalcomfort.classes_return import MRT

@dataclass
class ForthPower:
    surface_temps: list[float]  # °C or °F
    angle_factors: list[float]
    units: Literal["SI", "IP"] = "SI"

    def __post_init__(self) -> None:
        self._check_equal_lengths()
        self._check_angle_factors_sum()

    def _check_equal_lengths(self) -> None:
        if len(self.surface_temps) != len(self.angle_factors):
            raise ValueError(
                f"surface_temps and angle_factors must be the same length, "
                f"got {len(self.surface_temps)} and {len(self.angle_factors)}"
            )

    def _check_angle_factors_sum(self) -> None:
        if abs(sum(self.angle_factors) - 1.0) > 1e-9:
            raise ValueError(
                f"angle_factors must sum to 1, got {sum(self.angle_factors)}"
            )

@dataclass
class NoGeometry:
    tdb: float
    tr: float
    asw: float
    units: Literal["SI", "IP"] = "SI"

# ── Helpers ─────────────────────────────────────────────────────────

def _f_to_c(value: np.ndarray) -> np.ndarray:
    return (value - 32) * 5 / 9

def _c_to_f(value: np.ndarray) -> np.ndarray:
    return value * 9 / 5 + 32

def _forth_power(p: ForthPower) -> MRT:
    temps: NDArray[np.float64] = np.asarray(p.surface_temps)
    factors: NDArray[np.float64] = np.asarray(p.angle_factors)
    if p.units == "IP":
        temps: NDArray[float64] = _f_to_c(value=temps)
    mrt = np.sum((temps + 273.15) ** 4 * factors) ** 0.25 - 273.15
    if p.units == "IP":
        mrt = _c_to_f(value=mrt)
    return MRT(mrt=mrt)

def _no_geometry(p: NoGeometry) -> MRT:
    tdb = np.asarray(p.tdb)
    tr = np.asarray(p.tr)
    if p.units == "IP":
        tdb = _f_to_c(tdb)
        tr = _f_to_c(tr)
    mrt = p.asw * tdb + (1 - p.asw) * tr
    if p.units == "IP":
        mrt = _c_to_f(mrt)
    return MRT(mrt=mrt)

# ── Dispatcher ───────────────────────────────────────────────────────

def calculate_mrt(params: ForthPower | NoGeometry) -> MRT:
    match params:
        case ForthPower():
            return _forth_power(params)
        case NoGeometry():
            return _no_geometry(params)

# Testing:
print(calculate_mrt(ForthPower(surface_temps=[27.0, 17.0, 37.0], angle_factors=[0.4, 0.3, 0.3])))
print(calculate_mrt(ForthPower(surface_temps=[80.6, 62.6, 98.6], angle_factors=[0.4, 0.3, 0.3], units="IP")))
print(calculate_mrt(NoGeometry(tdb=25, tr=30, asw=0.6)))