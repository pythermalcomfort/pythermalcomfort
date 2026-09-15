from __future__ import annotations

import numpy as np

from pythermalcomfort.utilities import NumericInput


def clo_intrinsic_insulation_ensemble(
    clo_garments: NumericInput,
) -> float:
    """Calculate the intrinsic insulation of a clothing ensemble based on individual
    garments.

    This equation is in accordance with the ISO 9920:2009 standard [ISO9920]_
    Section 4.3. It should be noted that this equation is only valid for clothing
    ensembles with rather uniform insulation values across the body.

    Parameters
    ----------
    clo_garments:  floats or list of floats
        list of floats containing the clothing insulation for each individual garment

    Returns
    -------
    i_cl: float
        intrinsic insulation of the clothing ensemble, [clo]
    """
    clo_garments = np.asarray(clo_garments)
    return np.sum(clo_garments) * 0.835 + 0.161
