import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np


def valid_range(x, valid, param_name=None) -> np.ndarray:
    """Filter values based on a valid range."""
    x_arr = np.asarray(x)
    out_of_range_mask = (x_arr < valid[0]) | (x_arr > valid[1])

    if param_name is not None and np.any(out_of_range_mask):
        if x_arr.ndim == 0:  # scalar input
            warnings.warn(
                f"Value of '{param_name}' ({float(x_arr.item())}) is outside the "
                f"applicability limits [{valid[0]}, {valid[1]}] and will be "
                f"set to NaN.",
                UserWarning,
                stacklevel=2,
            )
        else:  # array input
            bad_indices = np.flatnonzero(out_of_range_mask).tolist()
            bad_values = x_arr[out_of_range_mask].tolist()
            warnings.warn(
                f"Some values of '{param_name}' are outside the applicability "
                f"limits [{valid[0]}, {valid[1]}] and will be set to NaN. "
                f"Out-of-range values: {bad_values} at indices {bad_indices}.",
                UserWarning,
                stacklevel=2,
            )

    return np.where(~out_of_range_mask, x_arr, np.nan)


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


def mapping(
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
    >>> mapping([20, 25, 30], {15: "low", 25: "medium", 35: "high"})
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
