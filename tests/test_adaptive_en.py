import numpy as np
import pytest

from pythermalcomfort.models import adaptive_en
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_adaptive_en(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the Adaptive Thermal Comfort Model (ASHRAE 55) correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.ADAPTIVE_EN.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = adaptive_en(
            inputs["tdb"],
            inputs["tr"],
            inputs["t_running_mean"],
            inputs["v"],
        )

        validate_result(result, outputs, tolerance)


def test_adaptive_en_running_mean_limits() -> None:
    """EN 16798-1 applies for running mean outdoor temperatures of 10 to 30 °C."""
    result = adaptive_en(tdb=25, tr=25, t_running_mean=[10, 30], v=0.1)
    np.testing.assert_allclose(result.tmp_cmf, [22.1, 28.7])

    with pytest.warns(UserWarning, match="t_running_mean"):
        result = adaptive_en(tdb=25, tr=25, t_running_mean=[9.9, 30.1, 33], v=0.1)
    assert np.all(np.isnan(result.tmp_cmf))

    result = adaptive_en(tdb=25, tr=25, t_running_mean=33, v=0.1, limit_inputs=False)
    assert np.isclose(result.tmp_cmf, 29.7)


def test_ashrae_inputs_invalid_units() -> None:
    """Test that the function raises a ValueError for invalid units."""
    with pytest.raises(ValueError):
        adaptive_en(tdb=25, tr=25, t_running_mean=20, v=0.1, units="INVALID")


def test_ashrae_inputs_invalid_tdb_type() -> None:
    """Test that the function raises a TypeError for invalid tdb type."""
    with pytest.raises(TypeError):
        adaptive_en(tdb="invalid", tr=25, t_running_mean=20, v=0.1)


def test_ashrae_inputs_invalid_tr_type() -> None:
    """Test that the function raises a TypeError for invalid tr type."""
    with pytest.raises(TypeError):
        adaptive_en(tdb=25, tr="invalid", t_running_mean=20, v=0.1)


def test_ashrae_inputs_invalid_t_running_mean_type() -> None:
    """Test that the function raises a TypeError for invalid t_running_mean type."""
    with pytest.raises(TypeError):
        adaptive_en(tdb=25, tr=25, t_running_mean="invalid", v=0.1)


def test_ashrae_inputs_invalid_v_type() -> None:
    """Test that the function raises a TypeError for invalid v type."""
    with pytest.raises(TypeError):
        adaptive_en(tdb=25, tr=25, t_running_mean=20, v="invalid")


def test_round_output_default_preserves_behaviour() -> None:
    """Default call must match an explicit round_output=True call."""
    default_result = adaptive_en(tdb=25, tr=25, t_running_mean=20.5, v=0.1)
    explicit_result = adaptive_en(
        tdb=25,
        tr=25,
        t_running_mean=20.5,
        v=0.1,
        round_output=True,
    )

    assert default_result.tmp_cmf == explicit_result.tmp_cmf
    assert default_result.tmp_cmf_cat_i_up == explicit_result.tmp_cmf_cat_i_up
    assert default_result.tmp_cmf_cat_i_low == explicit_result.tmp_cmf_cat_i_low
    assert default_result.tmp_cmf_cat_ii_up == explicit_result.tmp_cmf_cat_ii_up
    assert default_result.tmp_cmf_cat_iii_low == explicit_result.tmp_cmf_cat_iii_low


def test_round_output_false_returns_unrounded() -> None:
    """round_output=False must return unrounded t_cmf and bounds at full precision."""
    # 0.33 * 20.5 + 18.8 = 25.565; rounded to 1 dp this is 25.6.
    unrounded = adaptive_en(
        tdb=25,
        tr=25,
        t_running_mean=20.5,
        v=0.1,
        round_output=False,
    )
    rounded = adaptive_en(
        tdb=25,
        tr=25,
        t_running_mean=20.5,
        v=0.1,
        round_output=True,
    )

    assert np.isclose(unrounded.tmp_cmf, 25.565)
    assert np.isclose(rounded.tmp_cmf, 25.6)
    assert unrounded.tmp_cmf != rounded.tmp_cmf

    # Bounds derive from the unrounded value when round_output=False.
    assert np.isclose(unrounded.tmp_cmf_cat_i_low, 25.565 - 3.0)
    assert np.isclose(unrounded.tmp_cmf_cat_ii_up, 25.565 + 3.0)


def test_round_output_invalid_type_raises() -> None:
    """A non-boolean round_output must raise TypeError."""
    with pytest.raises(TypeError):
        adaptive_en(tdb=25, tr=25, t_running_mean=20, v=0.1, round_output="yes")
