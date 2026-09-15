from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pythermalcomfort.models.sports_heat_stress_risk import _SportsValues

import numpy as np

from pythermalcomfort._internal.validation import validate_type
from pythermalcomfort.utilities import NumericInput, Postures, Sex, Units

_NUMERIC_TYPES = NumericInput.__args__


def numeric_field(default=None):
    return field(default=default, metadata={"types": _NUMERIC_TYPES})


class WorkIntensity(str, Enum):
    """Enumeration for work intensity levels."""

    HEAVY = "heavy"
    MODERATE = "moderate"
    LIGHT = "light"


@dataclass
class BaseInputs:
    """Base inputs with metadata-driven validation."""

    a_coefficient: float | int = field(default=None, metadata={"types": (float, int)})
    age: NumericInput = numeric_field()
    airspeed_control: bool = field(default=True, metadata={"is_bool": True})
    asw: NumericInput = numeric_field()
    body_surface_area: NumericInput = numeric_field(1.8258)
    clo: NumericInput = numeric_field()
    d: NumericInput = numeric_field(0)
    duration: int = field(default=None, metadata={"types": (int, np.ndarray)})
    e_coefficient: float | int = field(default=None, metadata={"types": (float, int)})
    f_bes: NumericInput = numeric_field()
    f_svv: NumericInput = numeric_field()
    floor_reflectance: NumericInput = numeric_field()
    height: NumericInput = numeric_field()
    limit_inputs: bool = field(default=True, metadata={"is_bool": True})
    max_skin_blood_flow: NumericInput = numeric_field(80)
    max_sweating: NumericInput = numeric_field(500)
    met: NumericInput = numeric_field()
    p_atm: NumericInput = numeric_field(101325)
    position: str | np.ndarray | list = field(
        default=None,
        metadata={
            "allowed": [
                Postures.sitting.value,
                Postures.standing.value,
                "standing, forced convection",
            ]
        },
    )
    posture: str | np.ndarray | list = field(
        default=None,
        metadata={
            "allowed": [
                Postures.sitting.value,
                Postures.standing.value,
                Postures.crouching.value,
            ]
        },
    )
    q: NumericInput = numeric_field()
    rh: NumericInput = numeric_field()
    round_output: bool = field(default=True, metadata={"is_bool": True})
    sex: str | np.ndarray | list = field(
        default=None, metadata={"allowed": [Sex.male.value, Sex.female.value]}
    )
    sharp: NumericInput = numeric_field()
    sol_altitude: NumericInput = numeric_field()
    sol_radiation_dir: NumericInput = numeric_field()
    sol_radiation_global: NumericInput = numeric_field()
    sol_transmittance: NumericInput = numeric_field()
    t_re: NumericInput = numeric_field()
    t_running_mean: NumericInput = numeric_field()
    t_sk: NumericInput = numeric_field()
    tdb: NumericInput = numeric_field()
    tg: NumericInput = numeric_field()
    thickness_quilt: NumericInput = numeric_field()
    tout: NumericInput = numeric_field()
    tr: NumericInput = numeric_field()
    twb: NumericInput = numeric_field()
    units: str = field(default=Units.SI.value)
    v: NumericInput = numeric_field()
    v_ankle: NumericInput = numeric_field()
    v_z1: NumericInput = numeric_field()
    vapor_pressure: NumericInput = numeric_field()
    vertical_tmp_grad: NumericInput = numeric_field()
    vr: NumericInput = numeric_field()
    w_max: NumericInput = numeric_field()
    wbgt: NumericInput = numeric_field()
    weight: NumericInput = numeric_field()
    with_solar_load: bool = field(default=False, metadata={"is_bool": True})
    work_intensity: str | Enum = field(
        default=None, metadata={"allowed": [i.value for i in WorkIntensity]}
    )
    z0: NumericInput = numeric_field()
    z1: NumericInput = numeric_field()
    z2: NumericInput = numeric_field()
    wme: NumericInput = numeric_field(0)

    def __post_init__(self) -> None:
        """Validate and normalize fields using metadata declared on each field."""
        # Validate and normalise units
        units_str = (
            self.units.value if isinstance(self.units, Units) else str(self.units)
        )
        units_up = units_str.upper()
        if units_up not in (Units.SI.value, Units.IP.value):
            raise ValueError("Units must be either 'SI' or 'IP'")
        self.units = units_up

        # Validate booleans from metadata
        for f in dataclass_fields(self):
            if f.metadata.get("is_bool"):
                val = getattr(self, f.name)
                if not isinstance(val, bool):
                    msg = f"{f.name} must be a boolean (True or False)."
                    raise TypeError(msg)

        # Process fields with validation metadata (types / allowed)
        for f in dataclass_fields(self):
            meta = f.metadata
            if not meta:
                continue

            value = getattr(self, f.name)
            if value is None:
                continue

            # Convert pandas Series to list if needed
            value = self._convert_series_to_list(value)

            # Type validation
            expected_types = meta.get("types")
            if expected_types:
                value = validate_type(value, f.name, expected_types)
                # Store the validated and possibly normalized value.
                setattr(self, f.name, value)
                continue

            # Allowed string values validation (supports arrays/lists)
            allowed = meta.get("allowed")
            if allowed:
                self._validate_str_values(f.name, value, allowed)
                setattr(self, f.name, value)
                continue

    # ----- helper methods -----
    @staticmethod
    def _is_pandas_series(obj: Any) -> bool:
        return type(obj).__name__ in ("Series", "Index")

    def _convert_series_to_list(self, obj: Any) -> Any:
        return obj.tolist() if self._is_pandas_series(obj) else obj

    @staticmethod
    def _validate_str_values(name: str, value: Any, allowed: list[str]) -> None:
        arr = np.atleast_1d(value)
        # Coerce Enums to their .value, then to str
        coerced = [v.value if isinstance(v, Enum) else str(v) for v in arr.tolist()]
        allowed_lower = {str(a).lower() for a in allowed}
        for v in coerced:
            if v.lower() not in allowed_lower:
                msg = f"{name} must be one of {allowed!r}"
                raise ValueError(msg)


@dataclass
class APMVInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        a_coefficient,
        wme=0,
        units=Units.SI.value,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            clo=clo,
            a_coefficient=a_coefficient,
            wme=wme,
            units=units,
        )


@dataclass
class ASHRAEInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        t_running_mean,
        v,
        units,
        round_output=True,
    ):
        super().__init__(
            tdb=tdb,
            tr=tr,
            v=v,
            units=units,
            t_running_mean=t_running_mean,
            round_output=round_output,
        )


@dataclass
class ENInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        t_running_mean,
        v,
        units,
        round_output=True,
    ):
        super().__init__(
            tdb=tdb,
            tr=tr,
            v=v,
            units=units,
            t_running_mean=t_running_mean,
            round_output=round_output,
        )


@dataclass
class AnkleDraftInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        v_ankle,
        units=Units.SI.value,
        limit_inputs=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            clo=clo,
            v_ankle=v_ankle,
            units=units,
            limit_inputs=limit_inputs,
        )


@dataclass
class ATInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        rh,
        v,
        q=None,
        round_output=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            vr=v,
            rh=rh,
            q=q,
            round_output=round_output,
        )


@dataclass
class ATHBInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        vr,
        rh,
        met,
        t_running_mean,
    ):
        super().__init__(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            t_running_mean=t_running_mean,
        )


@dataclass
class CloTOutInputs(BaseInputs):
    def __init__(
        self,
        tout,
        units: str = Units.SI.value,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tout=tout,
            units=units,
        )


@dataclass
class CEInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        wme,
        units,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            clo=clo,
            wme=wme,
            units=units,
        )


@dataclass
class DIInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        rh,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            rh=rh,
        )


@dataclass
class EPMVInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        e_coefficient,
        wme,
        units,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            clo=clo,
            e_coefficient=e_coefficient,
            wme=wme,
            units=units,
        )


@dataclass
class ESIInputs(BaseInputs):
    """Input class for the Environmental Stress Index (ESI) calculation.

    This class validates and processes inputs required for calculating the ESI, which
    evaluates heat stress based on temperature, humidity, and solar radiation.
    """

    def __init__(self, tdb, rh, sol_radiation_global, round_output=True):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            rh=rh,
            sol_radiation_global=sol_radiation_global,
            round_output=round_output,
        )

    def __post_init__(self):
        super().__post_init__()

        rh = np.asarray(self.rh, dtype=float)
        if np.any(rh < 0) or np.any(rh > 100):
            raise ValueError("Relative humidity must be between 0 and 100 %")

        sol_radiation_global = np.asarray(self.sol_radiation_global, dtype=float)
        if np.any(sol_radiation_global < 0):
            raise ValueError("Solar radiation must be greater than or equal to 0 W/m2")


class HIModels(Enum):
    rothfusz = "rothfusz"
    lu_romps = "lu-romps"


@dataclass
class HIInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        rh,
        round_output,
        limit_inputs,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            rh=rh,
            round_output=round_output,
            limit_inputs=limit_inputs,
        )


class HumidexModels(Enum):
    rana = "rana"
    masterson = "masterson"


@dataclass
class HumidexInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        rh,
        round_output,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            rh=rh,
            round_output=round_output,
        )


@dataclass
class NETInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        rh,
        v,
        round_output=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            rh=rh,
            v=v,
            round_output=round_output,
        )


@dataclass
class PETSteadyInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        v,
        rh,
        met,
        clo,
        p_atm,
        position,
        age,
        sex,
        weight,
        height,
        wme,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            v=v,
            rh=rh,
            met=met,
            clo=clo,
            p_atm=p_atm,
            position=position,
            age=age,
            sex=sex,
            weight=weight,
            height=height,
            wme=wme,
        )


@dataclass
class PHSInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        v,
        rh,
        met,
        clo,
        round_output,
        wme,
        posture,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            v=v,
            rh=rh,
            met=met,
            clo=clo,
            round_output=round_output,
            wme=wme,
            posture=posture,
        )


@dataclass
class PMVInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        wme=0,
        units=Units.SI.value,
        limit_inputs=True,
        airspeed_control=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            clo=clo,
            wme=wme,
            units=units,
            limit_inputs=limit_inputs,
            airspeed_control=airspeed_control,
        )


@dataclass
class PMVPPDInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        wme=0,
        units=Units.SI.value,
        limit_inputs=True,
        airspeed_control=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            clo=clo,
            wme=wme,
            units=units,
            limit_inputs=limit_inputs,
            airspeed_control=airspeed_control,
        )


@dataclass
class SETInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        v,
        rh,
        met,
        clo,
        wme=0,
        body_surface_area=1.8258,
        p_atm=101325,
        position=Postures.standing.value,
        units=Units.SI.value,
        limit_inputs=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            v=v,
            rh=rh,
            met=met,
            clo=clo,
            wme=wme,
            body_surface_area=body_surface_area,
            p_atm=p_atm,
            position=position,
            units=units,
            limit_inputs=limit_inputs,
        )


@dataclass
class SolarGainInputs(BaseInputs):
    def __init__(
        self,
        sol_altitude,
        sharp,
        sol_radiation_dir,
        sol_transmittance,
        f_svv,
        f_bes,
        asw=0.7,
        posture=Postures.sitting.value,
        floor_reflectance=0.6,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            sol_altitude=sol_altitude,
            sharp=sharp,
            sol_radiation_dir=sol_radiation_dir,
            sol_transmittance=sol_transmittance,
            f_svv=f_svv,
            f_bes=f_bes,
            asw=asw,
            posture=posture,
            floor_reflectance=floor_reflectance,
        )


@dataclass
class GaggeTwoNodesInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        v,
        rh,
        met,
        clo,
        wme=0,
        body_surface_area=1.8258,
        p_atm=101325,
        position=Postures.standing.value,
        max_skin_blood_flow=90,
        round_output=True,
        max_sweating=500,
        w_max=None,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            v=v,
            rh=rh,
            met=met,
            clo=clo,
            wme=wme,
            body_surface_area=body_surface_area,
            p_atm=p_atm,
            position=position,
            max_skin_blood_flow=max_skin_blood_flow,
            round_output=round_output,
            max_sweating=max_sweating,
            w_max=w_max,
        )


@dataclass
class GaggeTwoNodesJiInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        v,
        met,
        clo,
        vapor_pressure,
        wme,
        body_surface_area,
        p_atm,
        position,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            v=v,
            met=met,
            clo=clo,
            vapor_pressure=vapor_pressure,
            wme=wme,
            body_surface_area=body_surface_area,
            p_atm=p_atm,
            position=position,
        )


@dataclass
class GaggeTwoNodesSleepInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        v,
        rh,
        clo,
        thickness_quilt,
        wme=0,
        p_atm=101325,
    ):
        # Initialise BaseInputs-supported fields
        super().__init__(
            tdb=tdb,
            tr=tr,
            v=v,
            rh=rh,
            clo=clo,
            wme=wme,
            p_atm=p_atm,
            thickness_quilt=thickness_quilt,
        )

    def __post_init__(self):
        super().__post_init__()

        if np.any(np.asarray(self.thickness_quilt, dtype=float) < 0):
            raise ValueError("thickness_quilt must be greater than or equal to 0 cm.")


@dataclass
class RidgeRegressionInputs(BaseInputs):
    """Input validation for the ridge_regression_predict_t_re_t_sk model."""

    def __init__(
        self,
        sex,
        age,
        height,
        weight,
        tdb,
        rh,
        duration,
        t_re=None,
        t_sk=None,
        limit_inputs=True,
        round_output=True,
    ):
        super().__init__(
            sex=sex,
            age=age,
            height=height,
            weight=weight,
            tdb=tdb,
            rh=rh,
            duration=duration,
            t_re=t_re,
            t_sk=t_sk,
            limit_inputs=limit_inputs,
            round_output=round_output,
        )

    def _validate_finite(self, param_name):
        """Validate that a numeric input is finite."""
        value = getattr(self, param_name)
        if value is not None:
            arr = np.asarray(value)
            if not np.all(np.isfinite(arr)):
                message = f"{param_name} must contain only finite values."
                raise ValueError(message)

    def __post_init__(self):
        super().__post_init__()

        # Validate duration is positive integer
        if (
            isinstance(
                self.duration, bool
            )  # bool is a subclass of int; exclude explicitly
            or not isinstance(self.duration, int)
            or self.duration <= 0
        ):
            raise ValueError("Duration must be a positive integer.")

        # Validate that all numeric inputs are finite
        for param_name in [
            "age",
            "height",
            "weight",
            "tdb",
            "rh",
            "t_re",
            "t_sk",
        ]:
            self._validate_finite(param_name)

        # Validate that all array-like inputs are broadcastable
        try:
            np.broadcast_arrays(
                self.sex, self.age, self.height, self.weight, self.tdb, self.rh
            )
        except ValueError as err:
            raise ValueError(
                "Input arrays are not broadcastable to a common shape."
            ) from err

        # Validate either both initial body temp values are provided or neither
        if (self.t_re is None) != (self.t_sk is None):
            raise ValueError("Both t_re and t_sk must be provided, or neither.")

        # If provided, ensure initial_t_* can broadcast to the same shape
        if self.t_re is not None and self.t_sk is not None:
            try:
                # Derive the target shape from a successful broadcast of core inputs
                target = np.broadcast(
                    self.sex, self.age, self.height, self.weight, self.tdb, self.rh
                ).shape
                np.broadcast_to(np.asarray(self.t_re, dtype=float), target)
                np.broadcast_to(np.asarray(self.t_sk, dtype=float), target)
            except ValueError as err:
                raise ValueError(
                    "t_re and t_sk must be broadcastable to the core input shape."
                ) from err

        # Basic plausibility checks
        rh_arr = np.asarray(self.rh)
        if np.any(rh_arr < 0) or np.any(rh_arr > 100):
            raise ValueError("Relative humidity (rh) must be between 0 and 100%.")

        for param_name in ["age", "height", "weight"]:
            value = getattr(self, param_name)
            if value is not None:
                arr = np.asarray(value)
                if np.any(arr <= 0):
                    message = f"{param_name} must be a positive value."
                    raise ValueError(message)


@dataclass
class THIInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        rh,
        round_output=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            rh=rh,
            round_output=round_output,
        )

    def __post_init__(self):
        super().__post_init__()

        rh = np.asarray(self.rh, dtype=float)
        if np.any(rh < 0) or np.any(rh > 100):
            raise ValueError("Relative humidity must be between 0 and 100 %")


@dataclass
class UseFansHeatwavesInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        v,
        rh,
        met,
        clo,
        wme=0,
        body_surface_area=1.8258,
        p_atm=101325,
        position=Postures.standing.value,
        max_skin_blood_flow=80,
        limit_inputs=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            v=v,
            rh=rh,
            met=met,
            clo=clo,
            wme=wme,
            body_surface_area=body_surface_area,
            p_atm=p_atm,
            position=position,
            max_skin_blood_flow=max_skin_blood_flow,
            limit_inputs=limit_inputs,
        )


@dataclass
class UTCIInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        v,
        rh,
        units=Units.SI.value,
        limit_inputs=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            v=v,
            rh=rh,
            units=units,
            limit_inputs=limit_inputs,
        )


@dataclass
class VerticalTGradPPDInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        vertical_tmp_grad,
        units=Units.SI.value,
        limit_inputs=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            clo=clo,
            vertical_tmp_grad=vertical_tmp_grad,
            units=units,
            limit_inputs=limit_inputs,
        )


@dataclass
class WBGTInputs(BaseInputs):
    def __init__(
        self,
        twb,
        tg,
        tdb=None,
        with_solar_load=False,
        round_output=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            twb=twb,
            tg=tg,
            tdb=tdb,
            with_solar_load=with_solar_load,
            round_output=round_output,
        )


@dataclass
class WCIInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        v,
        round_output=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            v=v,
            round_output=round_output,
        )


@dataclass
class WCTInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        v,
        round_output=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            v=v,
            round_output=round_output,
        )


@dataclass
class WorkCapacityHothapsInputs(BaseInputs):
    def __init__(
        self,
        wbgt,
        work_intensity,
    ):
        super().__init__(wbgt=wbgt, work_intensity=work_intensity)


@dataclass
class WorkCapacityStandardsInputs(BaseInputs):
    def __init__(
        self,
        wbgt,
        met,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            wbgt=wbgt,
            met=met,
        )

    def __post_init__(self):
        super().__post_init__()
        met = np.asarray(self.met, dtype=float)
        if np.any(met < 0) or np.any(met > 2500):
            raise ValueError("Metabolic rate out of plausible range")


# New input class for the scale_wind_speed_log function
@dataclass
class ScaleWindSpeedLogInputs(BaseInputs):
    """Inputs for scale_wind_speed_log (logarithmic wind profile).

    Validates types and applicability limits:
    - v_z1, z2, z1, z0, d must be numeric or array-like
    - wind speed must be non-negative
    - z0 > 0
    - z2 > z0
    - z1 > z0 (reference height must be above roughness length)
    - (z1 - d) > 0 and (z2 - d) > 0 (log arguments must be positive)
    - z0 must be less than reference height to avoid singular behavior
    """

    def __init__(
        self,
        v_z1,
        z2,
        z1: NumericInput = 10.0,
        z0: NumericInput = 0.01,
        d: NumericInput = 0.0,
    ):
        super().__init__(
            v_z1=v_z1,
            z2=z2,
            z1=z1,
            z0=z0,
            d=d,
        )

    def __post_init__(self):
        super().__post_init__()

        # Convert to numpy arrays for numeric checks and broadcasting
        v_z1 = np.asarray(self.v_z1, dtype=float)
        z2 = np.asarray(self.z2, dtype=float)
        z1 = np.asarray(self.z1, dtype=float)
        z0 = np.asarray(self.z0, dtype=float)
        d = np.asarray(self.d, dtype=float)

        # Check broadcasting compatibility
        try:
            np.broadcast_arrays(v_z1, z2, z1, z0, d)
        except ValueError as e:
            msg = (
                "Input shapes are incompatible for broadcasting: "
                f"v_z1.shape={v_z1.shape}, z2.shape={z2.shape}, "
                f"z1.shape={z1.shape}, z0.shape={z0.shape}, d.shape={d.shape}"
            )
            raise ValueError(msg) from e

        # Physical/value constraints
        if np.any(v_z1 < 0):
            raise ValueError("Wind speed (v_z1) must be non-negative")

        if np.any(z0 <= 0):
            raise ValueError("Surface roughness length (z0) must be positive ( > 0 )")

        # Displacement height cannot be negative
        if np.any(d < 0):
            raise ValueError("Zero-plane displacement height (d) must be >= 0")

        if np.any(z2 <= 0):
            raise ValueError("Target height (z2) must be positive ( > 0 )")

        if np.any(z1 <= 0):
            raise ValueError("Reference height (z1) must be positive ( > 0 )")

        if np.any(z2 <= z0):
            raise ValueError(
                "Target height (z2) must be greater than surface roughness (z0)"
            )

        if np.any(z1 <= z0):
            raise ValueError(
                "Reference height (z1) must be greater than surface roughness (z0)"
            )

        # Ensure log arguments are strictly > 1 i.e., (z - d) > z0
        if np.any((z1 - d) <= z0):
            raise ValueError(
                "Reference height minus displacement (z1 - d) must be > z0"
            )
        if np.any((z2 - d) <= z0):
            raise ValueError("Target height minus displacement (z2 - d) must be > z0")

        # Prevent log denominator being zero or extremely close to zero
        denom = np.log((z1 - d) / z0)
        if np.any(denom <= 1e-12):
            raise ValueError(
                "Logarithmic denominator log((z1 - d)/z0) is zero or numerically unstable"
            )

        # All checks passed: store the processed numpy arrays back on the instance
        self.v_z1 = v_z1
        self.z2 = z2
        self.z1 = z1
        self.z0 = z0
        self.d = d


@dataclass
class SportsHeatStressInputs(BaseInputs):
    """Inputs for
    :func:`pythermalcomfort.models.sports_heat_stress_risk.sports_heat_stress_risk`.

    Parameters
    ----------
    tdb : float or array-like
        Dry bulb air temperature in degrees Celsius (°C).
    tr : float or array-like
        Mean radiant temperature in degrees Celsius (°C).
    rh : float or array-like
        Relative humidity in percent (%). Values must be in the interval [0, 100].
    vr : float or array-like
        Relative air speed in metres per second (m/s). Values must be non-negative.
        Values lower than ``sport.vr`` are accepted and set to ``sport.vr`` by
        :func:`pythermalcomfort.models.sports_heat_stress_risk.sports_heat_stress_risk`.
    sport : _SportsValues
        Sport-specific parameters. Use one of the entries from
        :class:`pythermalcomfort.models.sports_heat_stress_risk.Sports` (e.g. ``Sports.RUNNING``).

    Raises
    ------
    TypeError
        If ``sport`` is not a ``_SportsValues`` instance.
    ValueError
        If ``rh`` contains values outside [0, 100], ``vr`` contains negative values,
        or the input arrays are not broadcastable to a common shape.

    Examples
    --------
    >>> from pythermalcomfort.classes_input import SportsHeatStressInputs
    >>> from pythermalcomfort.models.sports_heat_stress_risk import Sports
    >>> SportsHeatStressInputs(tdb=35, tr=35, rh=40, vr=0.5, sport=Sports.RUNNING)
    """

    def __init__(
        self,
        tdb: NumericInput,
        tr: NumericInput,
        rh: NumericInput,
        vr: NumericInput,
        sport: _SportsValues,
    ):
        # Store sport before calling super().__init__() as it's not a BaseInputs field
        self.sport = sport

        super().__init__(
            tdb=tdb,
            tr=tr,
            rh=rh,
            vr=vr,
        )

    def __post_init__(self):
        # Validate sport is a _SportsValues instance before calling super().__post_init__()
        from pythermalcomfort.models.sports_heat_stress_risk import _SportsValues

        if not isinstance(self.sport, _SportsValues):
            raise TypeError(
                "sport must be a _SportsValues instance from the Sports dataclass."
            )

        super().__post_init__()

        # Convert to numpy arrays for validation
        tdb = np.asarray(self.tdb, dtype=float)
        tr = np.asarray(self.tr, dtype=float)
        rh = np.asarray(self.rh, dtype=float)
        vr = np.asarray(self.vr, dtype=float)

        # Check broadcasting compatibility
        try:
            np.broadcast_arrays(tdb, tr, rh, vr)
        except ValueError as err:
            raise ValueError(
                "Input arrays (tdb, tr, rh, vr) are not broadcastable to a common shape."
            ) from err

        # Validate relative humidity range
        if np.any(rh < 0) or np.any(rh > 100):
            raise ValueError("Relative humidity (rh) must be between 0 and 100%.")

        # Validate air speed is non-negative
        if np.any(vr < 0):
            raise ValueError("Relative air speed (vr) must be non-negative.")


@dataclass
class IREQInputs(BaseInputs):
    p: float | int | np.ndarray | list = field(
        default=None, metadata={"types": (float, int, np.ndarray, list)}
    )
    walk_sp: float | int | np.ndarray | list = field(
        default=None, metadata={"types": (float, int, np.ndarray, list)}
    )

    def __init__(
        self,
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        p,
        walk_sp,
        wme=0,
        limit_inputs=True,
        round_output=True,
    ):
        self.p = p
        self.walk_sp = walk_sp
        super().__init__(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            clo=clo,
            wme=wme,
            limit_inputs=limit_inputs,
            round_output=round_output,
        )

    def __post_init__(self):
        super().__post_init__()

        values = {
            "tdb": self.tdb,
            "tr": self.tr,
            "vr": self.vr,
            "rh": self.rh,
            "met": self.met,
            "clo": self.clo,
            "p": self.p,
            "walk_sp": self.walk_sp,
            "wme": self.wme,
        }
        arrays = {
            name: np.atleast_1d(np.asarray(value, dtype=float))
            for name, value in values.items()
        }

        try:
            broadcasted = np.broadcast_arrays(*arrays.values())
        except ValueError as err:
            raise ValueError(
                "IREQ inputs are not broadcastable to a common shape."
            ) from err

        for name, value in zip(arrays.keys(), broadcasted, strict=True):
            if np.any(~np.isfinite(value)):
                msg = f"{name} must contain finite values."
                raise ValueError(msg)
            setattr(self, name, value.astype(float))

        if np.any(self.met <= 0):
            raise ValueError("Metabolic rate (met) must be greater than 0.")
        if np.any(self.wme < 0):
            raise ValueError("External work (wme) must be non-negative.")
        if np.any(self.p <= 0):
            raise ValueError("Air permeability (p) must be greater than 0.")
        if np.any(self.vr < 0):
            raise ValueError("Relative air speed (vr) must be non-negative.")
        if np.any(self.walk_sp < 0):
            raise ValueError("Walking speed (walk_sp) must be non-negative.")
        if np.any(self.rh < 0) or np.any(self.rh > 100):
            raise ValueError("Relative humidity (rh) must be between 0 and 100.")
        if np.any(self.clo < 0):
            raise ValueError("Clothing insulation (clo) must be non-negative.")
