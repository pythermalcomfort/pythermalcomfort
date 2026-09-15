from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pythermalcomfort.utilities import NumericInput


def clo_area_factor(
    i_cl: NumericInput,
) -> NDArray[np.float64]:
    """Calculate the clothing area factor (f_cl) of the clothing ensemble as a function
    of the intrinsic insulation of the clothing ensemble. This equation is in accordance
    with the ISO 9920:2009 standard [ISO9920]_ Section 5. The standard warns that the
    correlation between f_cl and i_cl is low especially for non-western clothing
    ensembles. The application of this equation is limited to clothing ensembles with
    clo values between 0.2 and 1.7 clo.

    Parameters
    ----------
    i_cl: float or list of floats
        intrinsic insulation of the clothing ensemble, [clo]

    Returns
    -------
    f_cl: float or list of floats
        area factor of the clothing ensemble, [m2]
    """
    i_cl = np.asarray(i_cl, dtype=np.float64)
    return 1 + 0.28 * i_cl
