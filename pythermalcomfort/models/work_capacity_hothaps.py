from __future__ import annotations

import numpy as np

from pythermalcomfort.classes_input import (
    NumericInput,
    WorkCapacityHothapsInputs,
    WorkIntensity,
)
from pythermalcomfort.classes_return import WorkCapacity


def work_capacity_hothaps(
    wbgt: NumericInput,
    work_intensity: str = WorkIntensity.HEAVY.value,
) -> WorkCapacity:
    """Estimate work capacity due to heat, based on [Kjellstrom2018]_.

    Estimates the amount of work that will be done at a given WBGT and
    intensity of work as a percent. 100% means work is unaffected by heat. 0%
    means no work is done. Note that in this version the functions do not reach
    0% as it is assumed that it is always possible to work in short bursts for
    10% of the time.

    Note that for this function "the empirical evidence is from studies in
    heavyly distinct locations, including a gold mine (Wyndham, 1969), 124 rice
    harvesters in West Bengal in India (Sahu et al., 2013), and six women
    observed in a climatic chamber (Nag and Nag, 1992)."
    (https://adaptecca.es/sites/default/files/documentos/2018_jrc_pesetaiii_impact_labour_productivity.pdf).
    The shape of the function is just an assumption, and the fit of the
    sigmoid to the data it is analysing is not especially good.

    Heavy intensity work is sometimes labelled as 400 W, moderate 300 W, light 200
    W, but this is only nominal.

    The correction citation is: Bröde P, Fiala D, Lemke B, Kjellstrom T.
    Estimated work ability in warm outdoor environments depends on the chosen
    heat stress assessment metric. International Journal of Biometeorology.
    2018 Mar;62(3):331 45.


    The relevant definitions of the functions can be found most clearly in:
    Orlov A, Sillmann J, Aunan K, Kjellstrom T, Aaheim A. Economic costs of
    heat-induced reductions in worker productivity due to global warming. Global
    Environmental Change [Internet]. 2020 Jul;63. Available from:
    https://doi.org/10.1016/j.gloenvcha.2020.102087

    For a comparison of different functions see Fig 1 of Day E, Fankhauser S, Kingsmill
    N, Costa H, Mavrogianni A. Upholding labour productivity under climate
    change: an assessment of adaptation options. Climate Policy. 2019
    Mar;19(3):367 85.

    Parameters
    ----------
    wbgt : float or list of floats
        Wet bulb globe temperature, [°C].
    work_intensity : str
        Which work intensity to use for the calculation, choice of "heavy",
        "moderate" or "light".

    Returns
    -------
    WorkCapacity
        A dataclass containing the work capacity. See
        :py:class:`~pythermalcomfort.classes_return.WorkCapacity` for more details. To access the
        `capacity` value, use the `capacity` attribute of the returned `WorkCapacity` instance, e.g.,
        `result.capacity`.
    """
    # validate inputs
    WorkCapacityHothapsInputs(wbgt=wbgt, work_intensity=work_intensity)

    # convert str to enum
    work_intensity = WorkIntensity(work_intensity.lower())
    wbgt = np.asarray(wbgt)

    params = {
        WorkIntensity.HEAVY: {"divisor": 30.94, "exponent": 16.64},
        WorkIntensity.MODERATE: {"divisor": 32.93, "exponent": 17.81},
        WorkIntensity.LIGHT: {"divisor": 34.64, "exponent": 22.72},
    }
    divisor = params[work_intensity]["divisor"]
    exponent = params[work_intensity]["exponent"]
    capacity = np.clip(100 * (0.1 + (0.9 / (1 + (wbgt / divisor) ** exponent))), 0, 100)

    return WorkCapacity(capacity=capacity)
