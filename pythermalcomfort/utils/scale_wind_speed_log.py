from __future__ import annotations

import warnings
from functools import wraps

from pythermalcomfort.environment import scale_wind_speed_log as _scale_wind_speed_log


@wraps(_scale_wind_speed_log)
def scale_wind_speed_log(*args, **kwargs):
    """Call the relocated wind-profile helper through its deprecated path."""
    warnings.warn(
        "pythermalcomfort.utils.scale_wind_speed_log is deprecated; "
        "import it from pythermalcomfort.environment instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _scale_wind_speed_log(*args, **kwargs)


scale_wind_speed_log.__doc__ = (
    "Deprecated alias for pythermalcomfort.environment.scale_wind_speed_log(). "
    "Import from there instead; this path will be removed after two minor releases."
)
