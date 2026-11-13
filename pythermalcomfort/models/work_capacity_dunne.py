from __future__ import annotations

import numpy as np

from pythermalcomfort.classes_input import (
    NumericInput,
    WorkCapacityHothapsInputs,
    WorkIntensity,
)
from pythermalcomfort.classes_return import WorkCapacity


def work_capacity_dunne(
    wbgt: NumericInput,
    work_intensity: str = WorkIntensity.HEAVY.value,
) -> WorkCapacity:
    """Estimate work capacity due to heat based on Dunne et al [Dunne2013]_.

    Estimates the amount of work that will be done at a given WBGT and
    intensity of work as a percent. 100% means work is unaffected by heat. 0%
    means no work is done.

    This is based upon NIOSH safety standards. See:
    Dunne JP, Stouffer RJ, John JG. Reductions in labour capacity from heat
    stress under climate warming. Nature Climate Change. 2013 Jun;3(6):563-6.

    Heavy intensity work is sometimes labelled as 400 W, but this is only
    nominal. Moderate work is assumed to be half as much as heavy intensity, and
    light half as much as moderate.

    Parameters
    ----------
    wbgt : float or list of floats
        Wet bulb globe temperature, [°C].
    work_intensity : str
        Which work intensity to use for the calculation, choice of "heavy",
        "moderate" or "light". Default is "heavy".

        .. note::
            Dunne et al [Dunne2013]_ suggests that heavy intensity work is 350-500 kcal/h, moderate
            is 200-350 kcal/h, and light is less than 100-200 kcal/h.

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

    capacity = np.clip((100 - (25 * (np.maximum(0, wbgt - 25)) ** (2 / 3))), 0, 100)

    factor_map = {
        WorkIntensity.HEAVY: 1,
        WorkIntensity.MODERATE: 2,
        WorkIntensity.LIGHT: 4,
    }

    capacity = np.clip(capacity * factor_map[work_intensity], 0, 100)

    return WorkCapacity(capacity=capacity)
