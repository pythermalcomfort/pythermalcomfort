from dataclasses import dataclass

@dataclass
class ForthPower:
    surface_temps: list[float]   # Celsius
    angle_factors: list[float]   

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

def _forth_power(p: ForthPower) -> float:
    return sum((t + 273.15)**4 * f for t, f in zip(p.surface_temps, p.angle_factors)) ** 0.25 - 273.15

def _no_geometry(i: NoGeometry) -> float:
    # lots of complex logic can live here cleanly
    weighted = i.asw * i.tdb + (1 - i.asw) * i.tr
    return weighted

# ── Dispatcher ──────────────────────────────────────────────────────

def calculate_mrt(inputs: ForthPower | NoGeometry) -> float:
    match inputs:
        case ForthPower():
            return _forth_power(inputs)
        case NoGeometry():
            return _no_geometry(inputs)
        case _:
            raise ValueError(f"Unknown input type '{type(inputs)}'")


# Testing:
print(calculate_mrt(ForthPower(surface_temps=[27.0, 17.0, 37.0], angle_factors=[0.4, 0.3, 0.3])))
print(calculate_mrt(NoGeometry(tdb=25, tr=30, asw=0.6)))
