import numpy as np
import pytest

from pythermalcomfort.psychrometrics import (
    dew_point_tmp,
    enthalpy_air,
    p_sat,
    psy_ta_rh,
    wet_bulb_tmp,
)
from tests.conftest import is_equal


def test_t_dp() -> None:
    """Test the dew point temperature function with various inputs."""
    # Scalar inputs
    assert dew_point_tmp(31.6, 59.6) == pytest.approx(22.778, abs=1e-1)
    assert dew_point_tmp(29.3, 75.4) == pytest.approx(24.497, abs=1e-1)
    assert dew_point_tmp(27.1, 66.4) == pytest.approx(20.302, abs=1e-1)

    # Edge cases
    # rh = 100%: dew point should equal tdb
    assert dew_point_tmp(25.0, 100.0) == pytest.approx(25.0, abs=1e-1)
    # rh = 0%: dew point should be very low (approaching -inf, but np.nan due to log(0))
    result_rh0 = dew_point_tmp(25.0, 0.0)
    assert np.isnan(result_rh0)  # Should be NaN due to log(0)

    # Array inputs
    tdb_array = [31.6, 29.3, 27.1]
    rh_array = [59.6, 75.4, 66.4]
    expected = [22.778, 24.497, 20.302]
    result_array = dew_point_tmp(tdb_array, rh_array)
    np.testing.assert_allclose(result_array, expected, atol=1e-1)

    # Broadcasting: tdb as array, rh as scalar
    tdb_array2 = [31.6, 29.3]
    rh_scalar = 59.6
    expected_broadcast = [22.778, 20.625]  # Same rh for both
    result_broadcast = dew_point_tmp(tdb_array2, rh_scalar)
    np.testing.assert_allclose(result_broadcast, expected_broadcast, atol=1e-1)

    # Numpy array inputs
    tdb_np = np.array([31.6, 29.3])
    rh_np = np.array([59.6, 75.4])
    expected_np = np.array([22.778, 24.497])
    result_np = dew_point_tmp(tdb_np, rh_np)
    np.testing.assert_allclose(result_np, expected_np, atol=1e-1)

    # More temperature ranges
    assert dew_point_tmp(10.0, 50.0) == pytest.approx(0.064, abs=1e-1)  # Low temp
    assert dew_point_tmp(40.0, 50.0) == pytest.approx(27.587, abs=1e-1)


def test_t_dp_invalid_rh() -> None:
    """Test that dew_point_tmp raises ValueError for invalid relative humidity."""
    with pytest.raises(ValueError):
        dew_point_tmp(25, 110)
    with pytest.raises(ValueError):
        dew_point_tmp(25, -10)


def test_t_wb() -> None:
    """Test the wet bulb temperature function with various inputs."""
    assert wet_bulb_tmp(27.1, 66.4) == pytest.approx(22.4, abs=1e-1)
    assert wet_bulb_tmp(25, 50) == pytest.approx(18.0, abs=1e-1)


def test_enthalpy() -> None:
    """Test the enthalpy function with various inputs."""
    assert is_equal(enthalpy_air(25, 0.01), 50561.25, 0.1)
    assert is_equal(enthalpy_air(27.1, 0.01), 52707.56, 0.1)


def test_psy_ta_rh() -> None:
    """Test the psychrometric function with various inputs."""
    results = psy_ta_rh(25, 50, p_atm=101325)
    assert results.p_sat == pytest.approx(3169.2, abs=1e-1)
    assert results.p_vap == pytest.approx(1584.6, abs=1e-1)
    assert results.hr == pytest.approx(0.00988, abs=1e-3)
    assert results.wet_bulb_tmp == pytest.approx(18.0, abs=1e-1)
    assert results.dew_point_tmp == pytest.approx(13.8, abs=1e-1)
    assert results.h == pytest.approx(50259.79, abs=1e-1)


def test_p_sat() -> None:
    """Test the saturation pressure function with various inputs."""
    assert pytest.approx(p_sat(tdb=25), abs=1e-1) == 3169.2
    assert pytest.approx(p_sat(tdb=50), abs=1e-1) == 12349.9
