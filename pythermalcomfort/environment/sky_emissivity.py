from __future__ import annotations
import numpy as np
from typing import Union, Literal
from pythermalcomfort.classes_input import (
    SkyEmissivityBruntInputs,
    SkyEmissivitySwinbankInputs,
    SkyEmissivityClarkAllenInputs,
)
from pythermalcomfort.classes_return import SkyEmissivity
from pythermalcomfort.utilities import Units, units_converter

def _brunt(p: SkyEmissivityBruntInputs) -> SkyEmissivity:
    """Internal Brunt (1975) calculation."""
    tdp = np.array(p.tdp, dtype=float)
    if p.units.upper() == Units.IP.value:
        tdp = units_converter(from_units=Units.IP.value, tdp=tdp)[0]

    # Formula: epsilon = 0.741 + 0.0062 * Tdp
    eps = np.clip(0.741 + 0.0062 * tdp, 0.0, 1.0)
    return SkyEmissivity(eps_sky=eps)

def _swinbank(p: SkyEmissivitySwinbankInputs) -> SkyEmissivity:
    """Internal Swinbank (1963) calculation."""
    tdb = np.array(p.tdb, dtype=float)
    if p.units.upper() == Units.IP.value:
        tdb = units_converter(from_units=Units.IP.value, tdb=tdb)[0]

    tk = tdb + 273.15
    eps = np.clip(9.37e-6 * tk**2, 0.0, 1.0)
    return SkyEmissivity(eps_sky=eps)

def _clark_allen(p: SkyEmissivityClarkAllenInputs) -> SkyEmissivity:
    """Internal Clark & Allen (1978) calculation."""
    tdp = np.array(p.tdp, dtype=float)
    fcn = np.array(p.fcn, dtype=float)

    if p.units.upper() == Units.IP.value:
        tdp = units_converter(from_units=Units.IP.value, tdp=tdp)[0]

    tk = tdp + 273.15
    # Clear sky component
    eps_clear = 0.787 + 0.764 * np.log(tk)
    # Cloud correction
    eps = np.clip(eps_clear * (1 + 0.23 * fcn), 0.0, 1.0)
    return SkyEmissivity(eps_sky=eps)

def sky_emissivity(
    params: SkyEmissivityBruntInputs | SkyEmissivitySwinbankInputs | SkyEmissivityClarkAllenInputs
) -> SkyEmissivity:
    """
    Dispatcher for sky emissivity models.
    """
    match params:
        case SkyEmissivityBruntInputs():
            return _brunt(p=params)
        case SkyEmissivitySwinbankInputs():
            return _swinbank(p=params)
        case SkyEmissivityClarkAllenInputs():
            return _clark_allen(p=params)
        case _:
            raise ValueError(f"Unknown sky emissivity model: {type(params)}")

# --- CORRECTIONS --- as standalone functions that can be applied.

def apply_dilley_correction(res: SkyEmissivity) -> SkyEmissivity:
    """Dilley (1998) adjustment for sky emissivity."""
    corrected = np.minimum(1.0, np.asarray(res.eps_sky) * 1.05)
    return SkyEmissivity(eps_sky=corrected)

def apply_cloud_correction_kimball(res: SkyEmissivity, fcn: float) -> SkyEmissivity:
    """Placeholder for Kimball cloud correction."""
    # Logic here...
    return res