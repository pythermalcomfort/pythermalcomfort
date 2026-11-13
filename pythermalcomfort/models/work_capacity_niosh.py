from __future__ import annotations

import numpy as np

from pythermalcomfort.classes_input import NumericInput, WorkCapacityStandardsInputs
from pythermalcomfort.classes_return import WorkCapacity


def work_capacity_niosh(
    wbgt: NumericInput,
    met: NumericInput,
) -> WorkCapacity:
    """Estimate work capacity due to heat based on NIOSH standards as described by Brode
    et al.

    Estimates the amount of work that will be done at a given WBGT and
    intensity of work as a percent. 100% means work is unaffected by heat. 0%
    means no work is done.

    The function definitions / parameters can be found in: 1. Bröde P, Fiala D,
    Lemke B, Kjellstrom T. Estimated work ability in warm outdoor environments
    depends on the chosen heat stress assessment metric. International Journal
    of Biometeorology. 2018 Mar;62(3):331-45.

    For a comparison of different functions see Fig 1 of Day E, Fankhauser S, Kingsmill
    N, Costa H, Mavrogianni A. Upholding labour productivity under climate
    change: an assessment of adaptation options. Climate Policy. 2019
    Mar;19(3):367-85.

    Parameters
    ----------
    wbgt : float or list of floats
        Wet bulb globe temperature, [°C].
    met : float or list of floats
        Metabolic heat production in Watts

    Returns
    -------
    WorkCapacity
        A dataclass containing the work capacity. See
        :py:class:`~pythermalcomfort.classes_return.WorkCapacity` for more details. To access the
        `capacity` value, use the `capacity` attribute of the returned `WorkCapacity` instance, e.g.,
        `result.capacity`.
    """
    # Validate inputs
    WorkCapacityStandardsInputs(wbgt=wbgt, met=met)

    wbgt = np.asarray(wbgt)
    met = np.asarray(met)

    met_rest = 117  # assumed resting metabolic rate

    wbgt_lim = 56.7 - 11.5 * np.log10(met)
    wbgt_lim_rest = 56.7 - 11.5 * np.log10(met_rest)
    capacity = ((wbgt_lim_rest - wbgt) / (wbgt_lim_rest - wbgt_lim)) * 100
    capacity = np.clip(capacity, 0, 100)

    return WorkCapacity(capacity=capacity)
