from __future__ import annotations

import json
from collections.abc import Callable
from enum import Enum
from typing import Any

import numpy as np
import pytest
import requests

# The conftest.py file serves as a means of providing fixtures for an entire directory.
# Fixtures defined in a conftest.py can be used by any test in that package
# without needing to import them (pytest will automatically discover them).


# Pinned to a tagged release rather than `main` so that changes to the
# validation-data-comfort-models repo can't silently break this repo's CI
# before they've been reviewed and the pin deliberately bumped. See that
# repo's CHANGELOG.md when bumping this tag.
unit_test_data_prefix = "https://raw.githubusercontent.com/FedericoTartarini/validation-data-comfort-models/v1.0.1/"


class Urls(Enum):
    """Enum for URLs of test data files."""

    ADAPTIVE_EN = "ts_adaptive_en.json"
    ADAPTIVE_ASHRAE = "ts_adaptive_ashrae.json"
    A_PMV = "ts_a_pmv.json"
    TWO_NODES = "ts_two_nodes_gagge.json"
    SOLAR_GAIN = "ts_solar_gain.json"
    ANKLE_DRAFT = "ts_ankle_draft.json"
    PHS = "ts_phs.json"
    E_PMV = "ts_e_pmv.json"
    AT = "ts_at.json"
    ATHB = "ts_athb.json"
    CLO_TOUT = "ts_clo_tout.json"
    COOLING_EFFECT = "ts_cooling_effect.json"
    VERTICAL_TMP_GRAD_PPD = "ts_vertical_tmp_grad_ppd.json"
    WBGT = "ts_wbgt.json"
    HEAT_INDEX = "ts_heat_index.json"
    NET = "ts_net.json"
    PMV_PPD = "ts_pmv_ppd.json"
    PMV = "ts_pmv.json"
    SET = "ts_set.json"
    HUMIDEX = "ts_humidex.json"
    USE_FANS_HEATWAVES = "ts_use_fans_heatwaves.json"
    UTCI = "ts_utci.json"
    WIND_CHILL = "ts_wind_chill.json"
    WIND_CHILL_TEMPERATURE = "ts_wind_chill_temperature.json"
    PET_STEADY = "ts_pet_steady.json"
    DISCOMFORT_INDEX = "ts_discomfort_index.json"


@pytest.fixture
def get_test_url() -> Callable[[str], str]:
    """Return a function that builds the full test data URL for a given model name.

    Returns
    -------
    Callable[[str], str]
        Callable accepting a model_name string and returning the complete URL.

    """

    def _get_test_url(model_name: str) -> str:
        """Construct the test URL for the given model name or return empty on failure."""
        try:
            return unit_test_data_prefix + Urls[model_name.upper()].value
        except KeyError:
            return ""

    return _get_test_url


@pytest.fixture
def retrieve_data() -> Callable[[str], dict[str, Any] | None]:
    """Return a function that fetches JSON data from a URL.

    Returns
    -------
    Callable[[str], Optional[Dict[str, Any]]]
        A callable accepting a URL string and returning a dict or None.

    """

    def _retrieve_data(url: str) -> dict[str, Any] | None:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return json.loads(response.text)
            response.raise_for_status()
        except requests.RequestException as e:
            message = f"Error fetching data from {url}: {e}"
            raise RuntimeError(message) from e
        return None

    return _retrieve_data


def is_equal(a, b, tolerance=1e-6) -> bool:
    """Compare two values for equality with a specified tolerance."""
    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray):
            b = np.asarray(b, dtype=a.dtype)
        if a.dtype.kind in "UOS":  # U = unicode, O = objects, S = string
            return np.array_equal(a, b)
        b = np.where(b is None, np.nan, b)  # Replace None with np.nan
        # Return True if arrays are close enough, including handling of NaN values
        return np.allclose(a, b, atol=tolerance, equal_nan=True)
    if (a is None and np.isnan(b)) or (b is None and np.isnan(a)):
        return True
    if isinstance(a, int | float) and isinstance(b, int | float):
        # Compare scalar values with tolerance
        return np.isclose(a, b, atol=tolerance)
    return a == b


def retrieve_reference_table(get_test_url: str, retrieve_data, url_name):
    """Retrieve the reference table for a given model from the test data URL."""
    reference_table = retrieve_data(get_test_url(url_name))
    if reference_table is None:
        pytest.fail(f"Failed to retrieve reference table for {url_name.lower()}")
    return reference_table


def validate_result(result: dict, expected_output: dict, tolerance: dict) -> None:
    """Validate the result of a function against expected output with a tolerance.

    Parameters
    ----------
    result : dict
        this is the result of the function that is being tested
    expected_output :dict
        this is the expected output of the function that is being tested
    tolerance : dict
        this is the tolerance that is used to compare the result with the
        expected output

    Returns
    -------
    None

    """
    for key in expected_output:
        _expected_output = expected_output[key]

        # some functions return a dictionary or class with multiple values while
        # others return a single value
        # TODO remove this once all the functions return a dictionary or class
        try:
            _result = result[key]
        except (IndexError, KeyError, TypeError):
            _result = result

        # if the key is not in the tolerance dictionary we set the tolerance to 1e-6
        _tolerance = tolerance.get(key, 1e-06)

        try:
            assert is_equal(_result, _expected_output, _tolerance)
        except Exception as e:
            message = f"Expected {_expected_output}, got {_result}\nError: {e!s}"
            raise RuntimeError(message) from e
