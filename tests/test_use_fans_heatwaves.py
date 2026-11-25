import warnings
from dataclasses import asdict

import numpy as np
import pytest

from pythermalcomfort.models import use_fans_heatwaves
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_use_fans_heatwaves(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the use of fans during heatwaves correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.USE_FANS_HEATWAVES.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = use_fans_heatwaves(**inputs)

        validate_result(result, outputs, tolerance)


FLAGS = (
    "heat_strain",
    "heat_strain_blood_flow",
    "heat_strain_w",
    "heat_strain_sweating",
)


def test_use_fans_heatwaves_limit_inputs_false_scalar() -> None:
    """With limit_inputs=False and the default round_output, the call returns floats.

    On 4.7.0 this call raised a casting error because the yes/no outputs were booleans.
    """
    result = use_fans_heatwaves(
        tdb=35, tr=35, v=1.0, rh=50, met=1.2, clo=0.5, limit_inputs=False
    )
    assert result.e_skin == 63.0
    for name in FLAGS:
        assert getattr(result, name) == 0.0


def test_use_fans_heatwaves_limit_inputs_false_array() -> None:
    """The yes/no outputs are 1.0 or 0.0 for array inputs too."""
    result = use_fans_heatwaves(
        tdb=[35, 45], tr=[35, 45], v=1.0, rh=50, met=1.2, clo=0.5, limit_inputs=False
    )
    np.testing.assert_array_equal(result.heat_strain, [0.0, 1.0])
    np.testing.assert_array_equal(result.heat_strain_blood_flow, [0.0, 1.0])
    np.testing.assert_array_equal(result.heat_strain_w, [0.0, 1.0])
    np.testing.assert_array_equal(result.heat_strain_sweating, [0.0, 0.0])
    np.testing.assert_allclose(result.e_skin, [63.0, 65.5])


# Each case has one input outside the applicability limits, which are
# 20 < tdb, tr < 50, 0.1 < v < 4.5, 0.7 < met < 2 and 0 < clo < 1.
OUT_OF_LIMITS = [
    {"tdb": 55, "tr": 55, "v": 1.0, "met": 1.2, "clo": 0.5},
    {"tdb": 15, "tr": 15, "v": 0.2, "met": 1.2, "clo": 0.5},
    {"tdb": 35, "tr": 35, "v": 0.05, "met": 1.2, "clo": 0.5},
    {"tdb": 35, "tr": 35, "v": 6.0, "met": 1.2, "clo": 0.5},
    {"tdb": 35, "tr": 35, "v": 1.0, "met": 2.5, "clo": 0.5},
    {"tdb": 35, "tr": 35, "v": 1.0, "met": 1.2, "clo": 1.5},
]


@pytest.mark.parametrize("inputs", OUT_OF_LIMITS)
def test_use_fans_heatwaves_limit_inputs_false_outside_limits(inputs) -> None:
    """Outside the limits, limit_inputs=True gives NaN and False gives finite values."""
    with pytest.warns(UserWarning):
        limited = use_fans_heatwaves(rh=50, **inputs)
    assert np.isnan(limited.e_skin)
    assert np.isnan(limited.heat_strain)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = use_fans_heatwaves(rh=50, **inputs, limit_inputs=False)
    for name, value in asdict(result).items():
        assert np.isfinite(value), name
    for name in FLAGS:
        assert getattr(result, name) in (0.0, 1.0)
    assert 36.0 < result.t_core < 42.0
    assert 0.0 <= result.w <= result.w_max <= 1.0
    assert result.e_skin >= 0.0
    assert 0.0 <= result.m_bl <= 80.0


def test_use_fans_heatwaves_limit_inputs_false_mixed_array() -> None:
    """In an array, only the points outside the limits differ between the two modes."""
    inputs = {
        "tdb": [35, 55, 15, 40],
        "tr": [35, 55, 15, 40],
        "v": [1.0, 1.0, 0.2, 0.5],
        "rh": 50,
        "met": 1.2,
        "clo": 0.5,
    }
    with pytest.warns(UserWarning):
        limited = use_fans_heatwaves(**inputs)
    unlimited = use_fans_heatwaves(**inputs, limit_inputs=False)
    inside = np.array([True, False, False, True])
    for name, limited_value in asdict(limited).items():
        unlimited_value = getattr(unlimited, name)
        assert np.all(np.isfinite(unlimited_value)), name
        np.testing.assert_array_equal(limited_value[inside], unlimited_value[inside])
        assert np.all(np.isnan(limited_value[~inside])), name
    # At 55 °C the body gains heat and strain is flagged. At 15 °C there is none.
    np.testing.assert_array_equal(unlimited.heat_strain[1:3], [1.0, 0.0])
    assert unlimited.t_core[1] > unlimited.t_core[0]
