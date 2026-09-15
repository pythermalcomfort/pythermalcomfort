from __future__ import annotations

import numpy as np

from pythermalcomfort.utilities import Models, NumericInput


def clo_dynamic_ashrae(
    clo: NumericInput,
    met: NumericInput,
    model: str = Models.ashrae_55_2023.value,
) -> np.ndarray:
    """Estimates the dynamic intrinsic clothing insulation (I :sub:`cl,r`).

    The ASHRAE 55:2023 refers to it as (I :sub:`cl,active`). The activity as well as the air speed
    modify the insulation characteristics of the clothing. Consequently, the ASHRAE 55
    standard provides a correction factor for the clothing insulation (I :sub:`cl`)
    based on the metabolic rate.

    Parameters
    ----------
    clo : float or list of floats
        clothing insulation, [clo]

        .. note::
            this is the basic insulation (I :sub:`cl`) also known as the intrinsic
            clothing insulation value under reference conditions

    met : float or list of floats
        metabolic rate, [met]
    model : str, optional
        Select the version of the ASHRAE 55 Standard to use. Currently, the only
        option available is "55-2023".

    Returns
    -------
    clo : float or list of floats
        dynamic clothing insulation (I :sub:`cl,r`), [clo]
    """
    clo = np.asarray(clo, dtype=np.float64)
    met = np.asarray(met, dtype=np.float64)

    model = model.lower()
    if model not in [Models.ashrae_55_2023.value]:
        invalid_model_msg = (
            f"PMV calculations can only be performed in compliance "
            f"with ASHRAE {Models.ashrae_55_2023.value}"
        )
        raise ValueError(invalid_model_msg)

    return np.where(met > 1.2, np.around(clo * (0.6 + 0.4 / met), 3), clo)
