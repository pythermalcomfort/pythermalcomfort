from __future__ import annotations

import warnings

import numpy as np

from pythermalcomfort._internal.validation import _format_violation_detail, _valid_range
from pythermalcomfort.environment import operative_tmp


def _check_ashrae55_compliance(**kwargs):
    """ASHRAE 55-2023 input applicability check.

    Implements the Table 7.3.4 single-variable limits and the §7.2.1.2
    ``airspeed_control`` cross-variable rules. Single-use compliance checks for other
    standards (ISO 7730, ISO 7933, fan_heatwaves) live inline in the respective model
    files.

    ``v_param_name`` (default ``"v"``) controls how the airspeed variable is rendered in
    UserWarnings emitted by this function. Callers whose signature names the airspeed
    argument ``vr`` should pass ``v_param_name="vr"`` so the warning text matches the
    caller's variable name. It does not affect which value is checked.
    """
    default_kwargs = {"airspeed_control": True, "v_param_name": "v"}
    params = {**default_kwargs, **kwargs}
    v_param_name = params["v_param_name"]
    values_to_return = {}

    tdb_valid = _valid_range(params["tdb"], (10.0, 40.0), param_name="tdb")
    tr_valid = _valid_range(params["tr"], (10.0, 40.0), param_name="tr")

    values_to_return["tdb"] = tdb_valid
    values_to_return["tr"] = tr_valid

    if "v" in params:
        v_valid = _valid_range(params["v"], (0.0, 2.0), param_name=v_param_name)
        values_to_return["v"] = v_valid

    if not params["airspeed_control"]:
        v_arr = np.asarray(params["v"])
        cond1 = (params["v"] > 0.8) & (params["clo"] < 0.7) & (params["met"] < 1.3)
        if np.any(cond1):
            detail = _format_violation_detail(v_arr, cond1)
            warnings.warn(
                f"airspeed_control=False: '{v_param_name}' has {detail} exceeding "
                f"0.8 m/s under low clothing (clo < 0.7) and low activity "
                f"(met < 1.3) and will be set to NaN.",
                UserWarning,
                stacklevel=2,
            )
        v_valid = np.where(cond1, np.nan, v_valid)

        to = operative_tmp(params["tdb"], params["tr"], params["v"])
        v_limit = 50.49 - 4.4047 * to + 0.096425 * to * to

        cond2 = (
            (to > 23)
            & (to < 25.5)
            & (params["v"] > v_limit)
            & (params["clo"] < 0.7)
            & (params["met"] < 1.3)
        )
        if np.any(cond2):
            detail = _format_violation_detail(v_arr, cond2)
            warnings.warn(
                f"airspeed_control=False: '{v_param_name}' has {detail} exceeding "
                f"the ASHRAE 55 airspeed limit in the comfort zone "
                f"(23°C < to < 25.5°C) and will be set to NaN.",
                UserWarning,
                stacklevel=2,
            )
        v_valid = np.where(cond2, np.nan, v_valid)

        cond3 = (
            (to <= 23)
            & (params["v"] > 0.2)
            & (params["clo"] < 0.7)
            & (params["met"] < 1.3)
        )
        if np.any(cond3):
            detail = _format_violation_detail(v_arr, cond3)
            warnings.warn(
                f"airspeed_control=False: '{v_param_name}' has {detail} exceeding "
                f"0.2 m/s when operative temperature is ≤ 23°C and will be set "
                f"to NaN.",
                UserWarning,
                stacklevel=2,
            )
        v_valid = np.where(cond3, np.nan, v_valid)

        values_to_return["v"] = v_valid

    if "met" in params:
        met_valid = _valid_range(params["met"], (1.0, 4.0), param_name="met")
        clo_valid = _valid_range(params["clo"], (0.0, 1.5), param_name="clo")

        values_to_return["met"] = met_valid
        values_to_return["clo"] = clo_valid

    if "v_limited" in params:
        valid = _valid_range(params["v_limited"], (0.0, 0.2), param_name="v_limited")
        values_to_return["v_limited"] = valid

    return values_to_return.values()
