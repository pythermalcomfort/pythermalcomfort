import ast
import inspect
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np


def _valid_range(
    x,
    valid: tuple[float, float],
    param_name: str | None = None,
) -> np.ndarray:
    """Filter values based on a valid range, warning on out-of-range entries.

    Parameters
    ----------
    x : float or array-like
        Value(s) to validate.
    valid : tuple[float, float]
        Inclusive lower and upper bounds of the applicability range.
    param_name : str, optional
        Label for the parameter, used in the UserWarning text. When omitted,
        the name is auto-extracted from the caller's source via frame
        inspection. Pass ``param_name`` explicitly when the first argument is
        an expression, a subscript, or a literal — frame inspection only
        recovers bare variable names.

    Returns
    -------
    np.ndarray
        Array with out-of-range entries replaced by ``np.nan``.

    Warns
    -----
    UserWarning
        Emitted when any element falls outside ``valid``. Example messages::

            'tdb' has value 50.0 outside the applicability limits [10.0, 40.0] and will be set to NaN.
            'tdb' has 2 values [50.0, 45.0] at indices [1, 3] outside the applicability limits [10.0, 40.0] and will be set to NaN.
    """
    if param_name is None:
        param_name = _extract_caller_argname() or "<unknown>"

    x_arr = np.asarray(x)
    out_of_range_mask = (x_arr < valid[0]) | (x_arr > valid[1])

    if np.any(out_of_range_mask):
        detail = _format_violation_detail(x_arr, out_of_range_mask)
        msg = (
            f"'{param_name}' has {detail} outside the applicability "
            f"limits [{valid[0]}, {valid[1]}] and will be set to NaN."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

    return np.where(~out_of_range_mask, x_arr, np.nan)


def _format_violation_detail(x_arr: np.ndarray, mask: np.ndarray) -> str:
    """Format the detail portion of a UserWarning describing out-of-range values.

    Shared by :func:`_valid_range` and by the ASHRAE 55 ``airspeed_control``
    checks so warning bodies stay consistent.

    Output formats
    --------------
    Scalar (0-d) input::

        value 50.0

    Array input::

        1 value [50.0] at indices [0]
        2 values [50.0, 45.0] at indices [1, 3]
    """
    if x_arr.ndim == 0:
        return f"value {float(x_arr.item())}"
    n_bad = int(mask.sum())
    bad_values = x_arr[mask].tolist()
    bad_indices = np.flatnonzero(mask).tolist()
    noun = "value" if n_bad == 1 else "values"
    return f"{n_bad} {noun} {bad_values} at indices {bad_indices}"


def _extract_caller_argname() -> str | None:
    """Return the name of the first positional argument of the caller's ``_valid_range``
    call, or ``None`` if it cannot be recovered.

    The caller's source is inspected via :mod:`inspect` and parsed with :mod:`ast`.
    ``None`` is returned when the source is unavailable (e.g. code compiled without
    source, some REPL contexts) or when the first argument is not a bare ``Name`` node
    (e.g. expressions, subscripts, literals). Callers should pass ``param_name``
    explicitly in those cases.
    """
    frame = inspect.currentframe()
    try:
        if frame is None or frame.f_back is None or frame.f_back.f_back is None:
            return None
        caller = frame.f_back.f_back
        info = inspect.getframeinfo(caller, context=10)
        if not info.code_context or info.index is None:
            return None
        lines = info.code_context
        idx = info.index
        n = len(lines)

        # Try parseable windows containing idx, growing outward from the
        # call line. Smaller windows are tried first so an unrelated nearby
        # call cannot be mistaken for the active one.
        windows = [(idx, idx + 1)]
        for radius in range(1, max(idx, n - idx - 1) + 1):
            start = max(0, idx - radius)
            end = min(n, idx + radius + 1)
            windows.append((start, end))

        for start, end in windows:
            src = "".join(lines[start:end]).strip()
            if not src:
                continue
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_valid_range"
                    and node.args
                    and isinstance(node.args[0], ast.Name)
                ):
                    return node.args[0].id
        return None
    finally:
        del frame


def _finalize_scalar_or_array(arr: Any) -> Any:
    """Convert 0d arrays to Python scalars, preserve np.nan, return arrays as-is.

    Args:
        arr: np.ndarray, scalar, or array-like.

    Returns:
        Python scalar (with np.nan preserved) if input is scalar, else array.

    Examples
    --------
    >>> _finalize_scalar_or_array(np.array(True, dtype=object))
    True
    >>> _finalize_scalar_or_array(np.array(np.nan, dtype=object))
    nan
    >>> _finalize_scalar_or_array(np.array([True, False, np.nan], dtype=object))
    array([True, False, nan], dtype=object)
    """
    arr = np.asarray(arr, dtype=object)
    if arr.shape == ():
        val = arr.item()
        if isinstance(val, float) and np.isnan(val):
            return np.nan
        # Convert np.bool_ to Python bool
        if isinstance(val, (np.bool_ | bool)):
            return bool(val)
        return val
    return arr


# Stress category thresholds shared by the Heat Index models (e.g. Rothfusz, Schoen).
# Each model prepends its own "no risk" floor, since that lower bound depends on the
# model's applicability range, e.g. {27.0: "no risk", **HEAT_INDEX_STRESS_CATEGORIES}.
HEAT_INDEX_STRESS_CATEGORIES = {
    32.0: "caution",
    41.0: "extreme caution",
    54.0: "danger",
    1000.0: "extreme danger",
}


def _mapping(
    value: float | np.ndarray, map_dictionary: Mapping[float, Any], right: bool = True
) -> np.ndarray:
    """Map a temperature array to stress categories.

    Parameters
    ----------
    value : float or array-like
        Temperature(s) to map.
    map_dictionary : dict
        Dictionary mapping bin edges to categories.
    right : bool, optional
        If True, intervals include the right bin edge.

    Returns
    -------
    np.ndarray
        Stress category for each input temperature. np.nan for unmapped.

    Raises
    ------
    TypeError
        If input types are invalid.

    Examples
    --------
    >>> _mapping([20, 25, 30], {15: "low", 25: "medium", 35: "high"})
    array(['low', 'medium', 'high'], dtype=object)
    """
    if not isinstance(map_dictionary, dict):
        raise TypeError("map_dictionary must be a dict")
    value_arr = np.asarray(value)
    bins = np.asarray(list(map_dictionary.keys()))
    categories = np.array(list(map_dictionary.values()), dtype=object)
    # Append np.nan for out-of-range values
    categories = np.append(categories, np.nan)
    idx = np.digitize(value_arr, bins, right=right)
    return categories[idx]


def validate_type(
    value: Any,
    name: str,
    allowed_types: tuple[type, ...],
) -> Any:
    """Validate a value and return it with NumPy scalars normalized."""
    if isinstance(value, np.generic):
        value = value.item()
    if not isinstance(value, allowed_types):
        invalid_type_msg = (
            f"{name} must be one of the following types: {allowed_types}."
        )
        raise TypeError(invalid_type_msg)
    return value
