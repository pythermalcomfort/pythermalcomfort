from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from typing import Union, Callable, Optional


from pythermalcomfort.classes_input import (
    SkyEmissivityBruntInputs,
    SkyEmissivitySwinbankInputs,
    SkyEmissivityClarkAllenInputs,
)
from pythermalcomfort.utilities import Units, units_converter

# Variant 02: 
# Example:
# eps_sky = SkyEmissivity.brunt(tdp=10)
# eps_sky = SkyEmissivity.brunt(tdp=10, correction=EpsSky.apply_dilley) 
# ...

@dataclass(frozen=True)
class EpsSky:
    """Immutable container for sky emissivity value."""
    eps_sky: Union[float, np.ndarray]

    @staticmethod
    def dilly(eps: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Dilley correction (max 1.0)."""
        return np.minimum(1.0, np.asarray(eps) * 1.05)

    @staticmethod
    def prata(eps: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Placeholder for Prata correction example."""
        return np.minimum(1.0, np.asarray(eps) * 1.03)

    @staticmethod
    def example_correction_with_inputs(
        eps: Union[float, np.ndarray],
        **kwargs
    ) -> Union[float, np.ndarray]:
        """
        Example correction that can take extra inputs from kwargs.

        Accepts keys like 'cloud_fraction', 'tdb', etc.
        """
        cloud_fraction = kwargs.get("cloud_fraction", 0.0)
        factor = 1.03 + 0.05 * cloud_fraction
        return np.minimum(1.0, np.asarray(eps) * factor)



class SkyEmissivity:
    """Collection of empirical sky emissivity models."""

    @staticmethod
    def brunt(
        tdp: float | list[float],
        units: str = Units.SI.value,
        correction: Optional[Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]] = None,
    ) -> EpsSky:
        """Brunt (1975) model."""
        SkyEmissivityBruntInputs(tdp=tdp)

        tdp_arr = np.array(tdp, dtype=float)
        if units.upper() == Units.IP.value:
            tdp_arr = units_converter(from_units=Units.IP.value, tdp=tdp_arr)[0]

        eps = np.clip(0.741 + 0.0062 * tdp_arr, 0.0, 1.0)

        if correction:
            eps = correction(eps)

        return EpsSky(eps_sky=eps)

    @staticmethod
    def swinbank(
        tdb: float | list[float],
        units: str = Units.SI.value,
        correction: Optional[Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]] = None,
    ) -> EpsSky:
        """Swinbank (1963) model — simple ambient air temperature approach."""
        SkyEmissivitySwinbankInputs(tdb=tdb)

        tdb_arr = np.array(tdb, dtype=float)
        if units.upper() == Units.IP.value:
            tdb_arr = units_converter(from_units=Units.IP.value, tdb=tdb_arr)[0]

        T_k = tdb_arr + 273.15
        eps = np.clip(9.37e-6 * T_k**2, 0.0, 1.0)

        if correction:
            eps = correction(eps)

        return EpsSky(eps_sky=eps)

    @staticmethod
    def clark_allen(
        tdp: float | list[float],
        cloud_fraction: float | list[float] = 0.0,
        units: str = Units.SI.value,
        correction: Optional[Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]]] = None,
    ) -> EpsSky:
        """Clark & Allen (1978) model with optional cloud fraction."""
        SkyEmissivityClarkAllenInputs(tdp=tdp, fcn=cloud_fraction)

        tdp_arr = np.array(tdp, dtype=float)
        cloud_arr = np.array(cloud_fraction, dtype=float)

        if units.upper() == Units.IP.value:
            tdp_arr = units_converter(from_units=Units.IP.value, tdp=tdp_arr)[0]

        T_k = tdp_arr + 273.15
        eps_clear = 0.787 + 0.764 * np.log(T_k)
        eps = np.clip(eps_clear * (1 + 0.23 * cloud_arr), 0.0, 1.0)

        if correction:
            eps = correction(eps)

        return EpsSky(eps_sky=eps)
# 
# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Literal

# import numpy as np

# from pythermalcomfort.classes_input import (
#     SkyEmissivityBruntInputs,
#     SkyEmissivityClarkAllenInputs,
#     SkyEmissivitySwinbankInputs,
# )
# from pythermalcomfort.classes_return import AutoStrMixin, Eps_Sky
# from pythermalcomfort.utilities import Units, units_converter

# # Is chaining really a useful approach here?
# # For example it might be better to just optionall apply correction? sky = SkyEmissivity.brunt(tdp=10, correction=EpsSky.apply_dilley) instead of
# # eps_sky = SkyEmissivity.brunt(tdp=10).apply_billy()

# # Variant 01:
# # Example:
# # eps_sky = SkyEmissivity.brunt(tdp=10)
# # eps_sky = SkyEmissivity.brunt(tdp=10).apply_billy()
# # ...


# @dataclass(frozen=True, repr=False)
# class SkyEmissivityResult(AutoStrMixin):
#     """Represents sky emissivity (ε_sky)."""

#     eps_sky: float | np.ndarray

#     def __post_init__(self):
#         if np.any(np.asarray(self.eps_sky) < 0) or np.any(np.asarray(self.eps_sky) > 1):
#             raise ValueError("eps_sky must be in the range [0, 1]")

#     def apply_dilley(self) -> EpsSky:
#         """Apply Dilley correction (max 1.0)."""
#         corrected = np.minimum(1.0, np.asarray(self.eps_sky) * 1.05)
#         return Eps_Sky(eps_sky=corrected)

#     # Add Kimball, Unsworth and Crawford


# @dataclass
# class SkyEmissivity:
#     """Collection of empirical sky emissivity models."""

#     @staticmethod
#     def brunt(
#         tdp: float | list[float],
#         units: Literal["SI", "IP"] = Units.SI.value,
#     ) -> SkyEmissivityResult:
#         """
#         Calculate sky emissivity using the Brunt (1975) empirical model.

#         Parameters
#         ----------
#         tdp : float
#             Dew point temperature of the air.
#         units : {"SI", "IP"}, default "SI"
#             Units of `tdp`. "SI" = °C, "IP" = °F.

#         Returns
#         -------
#         EpsSky
#             Instance containing the computed sky emissivity (eps_sky),
#             clipped to the physical range [0.0, 1.0].
#         """
#         SkyEmissivityBruntInputs(tdp=tdp)

#         if units.upper() == Units.IP.value:
#             tdp = units_converter(from_units=Units.IP.value, tdp=tdp)[0]

#         tdp_arr = np.array(tdp, dtype=float)

#         epsilon = 0.741 + 0.0062 * tdp_arr
#         eps = np.clip(epsilon, 0.0, 1.0)

#         return SkyEmissivityResult(eps_sky=eps)

#     # Revisit: Should we than even include this, if it inaccurate?
#     @staticmethod
#     def swinbank(
#         tdb: float | list[float],
#         units: Literal["SI", "IP"] = Units.SI.value,
#     ) -> SkyEmissivityResult:
#         """
#         Calculate sky emissivity using the Swinbank (1963) empirical formula.

#         .. note::
#             This is a simple calculation based on ambient air temperature.
#             It has shown large deviations compared to modern methods
#             and is **not recommended for accurate modeling**.

#         Parameters
#         ----------
#         tdb : float or list of floats
#             Dry bulb air temperature in degrees Celsius (SI) or Fahrenheit (IP).
#         units : str, optional
#             Units system, 'SI' or 'IP'. Defaults to 'SI'.

#         Returns
#         -------
#         EpsSky
#             A dataclass containing `eps_sky` (clipped to [0, 1]).
#         """

#         # Notes BG: Read here: https://doi.org/10.26868/25222708.2017.569 who cited:
#         # https://doi.org/10.1029/2009JD011800 and
#         # https://doi.org/10.1029/2008WR007394

#         SkyEmissivitySwinbankInputs(tdb=tdb)

#         tdb_arr = np.array(tdb, dtype=float)

#         if units.upper() == Units.IP.value:
#             tdb_arr = units_converter(from_units=Units.IP.value, tdb=tdb_arr)[0]

#         T_k = tdb_arr + 273.15
#         eps_sky = 9.37e-6 * T_k**2
#         eps_sky = np.clip(eps_sky, 0.0, 1.0)

#         # Return a typed dataclass
#         return SkyEmissivityResult(eps_sky=eps_sky)

#     def clark_allen(
#         tdb: float | list[float],
#         fcn: float | list[float],
#         units: str = Units.SI.value,
#     ) -> SkyEmissivityResult:
#         """
#         Calculate sky emissivity using the Clark & Allen (1978) model.

#         Parameters
#         ----------
#         tdp : float or list of floats
#             Dew point temperature in °C (SI) or °F (IP)
#         cloud_fraction : float or list of floats, default=0.0
#             Fraction of cloud cover [0, 1]
#         units : str, default "SI"
#             Units system, "SI" = °C, "IP" = °F

#         Returns
#         -------
#         SkyEmissivityResult
#             Object containing `eps_sky` with emissivity values (clipped to [0,1])

#         """

#         # Notes BG: Read here: https://doi.org/10.26868/25222708.2017.569
#         # https://www.proquest.com/openview/4763607fbd21956404a6329a060ae2b4/1?pq-origsite=gscholar&cbl=18750&diss=y

#         SkyEmissivityClarkAllenInputs(tdp=tdp, fcn=cloud_fraction)

#         tdp_arr = np.array(tdp, dtype=float)
#         cloud_arr = np.array(cloud_fraction, dtype=float)

#         if units.upper() == Units.IP.value:
#             tdp_arr = units_converter(from_units=Units.IP.value, tdp=tdp_arr)[0]

#         T_k = tdp_arr + 273.15
#         epsilon_clear = 0.787 + 0.764 * np.log(T_k)
#         Ca = 0.23
#         epsilon_sky = epsilon_clear * (1 + Ca * cloud_arr)
#         epsilon_sky = np.clip(epsilon_sky, 0.0, 1.0)

#         return SkyEmissivityResult(eps_sky=epsilon_sky)

#         # add Dilly, Prata and Angstrom
