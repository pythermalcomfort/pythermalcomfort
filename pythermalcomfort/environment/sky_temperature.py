from __future__ import annotations
import numpy as np
from pythermalcomfort.classes_input import SkyTemperatureInputs
from pythermalcomfort.classes_return import SkyTemperature 
from pythermalcomfort.utilities import Units, units_converter

def _blackbody_equivalence(p: SkyTemperatureInputs) -> SkyTemperature:
    """
    Internal calculation to convert air temperature and emissivity 
    to an effective sky temperature (T_sky).
    """
    tdb = np.array(p.tdb, dtype=float)
    eps_sky = np.array(p.eps_sky, dtype=float)

    if p.units.upper() == Units.IP.value:
        tdb = units_converter(from_units=Units.IP.value, tdb=tdb)[0]

    # The Core Physics: T_eff = T_air * (eps^0.25)
    tk = tdb + 273.15
    t_sky_k = tk * (eps_sky**0.25)
    t_sky_c = t_sky_k - 273.15

    # If the user started in IP, convert the result back to IP
    if p.units.upper() == Units.IP.value:
        t_sky_final = units_converter(from_units=Units.SI.value, tmp=t_sky_c)[0]
    else:
        t_sky_final = t_sky_c

    return SkyTemperature(t_sky=t_sky_final)

def sky_temperature(params: SkyTemperatureInputs) -> SkyTemperature:
    """
    Calculates the effective sky temperature (brightness temperature) 
    required for radiation balance models like RayMan.
    """
    return _blackbody_equivalence(p=params)