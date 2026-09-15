from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _correction_nude(
    _vr: NDArray[np.number[Any]],
    _vw: NDArray[np.number[Any]],
) -> NDArray[np.floating[Any]]:
    """Calculate the correction factor for the total insulation of the clothing
    ensemble."""
    return np.exp(
        -0.533 * (_vr - 0.15)
        + 0.069 * (_vr - 0.15) ** 2
        - 0.462 * _vw
        + 0.201 * _vw**2,
    )


def _correction_normal_clothing(
    _vr: NDArray[np.number[Any]],
    _vw: NDArray[np.number[Any]],
) -> NDArray[np.floating[Any]]:
    """Calculate the correction factor for normal clothing."""
    return np.exp(
        -0.281 * (_vr - 0.15)
        + 0.044 * (_vr - 0.15) ** 2
        - 0.492 * _vw
        + 0.176 * _vw**2,
    )
