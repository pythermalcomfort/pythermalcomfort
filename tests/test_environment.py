import numpy as np
import pytest

from pythermalcomfort.environment import (
    f_svv,
    mean_radiant_tmp,
    operative_tmp,
    running_mean_outdoor_temperature,
    scale_wind_speed_log,
    transpose_sharp_altitude,
    v_relative,
)
from pythermalcomfort.utilities import Units


def test_transpose_sharp_altitude() -> None:
    """Test the transpose_sharp_altitude function."""
    assert transpose_sharp_altitude(sharp=0, altitude=0) == (0, 90)
    assert transpose_sharp_altitude(sharp=0, altitude=20) == (0, 70)
    assert transpose_sharp_altitude(sharp=0, altitude=45) == (0, 45)
    assert transpose_sharp_altitude(sharp=0, altitude=60) == (0, 30)
    assert transpose_sharp_altitude(sharp=90, altitude=0) == (90, 0)
    assert transpose_sharp_altitude(sharp=90, altitude=45) == (45, 0)
    assert transpose_sharp_altitude(sharp=90, altitude=30) == (60, 0)
    assert transpose_sharp_altitude(sharp=135, altitude=60) == (22.208, 20.705)
    assert transpose_sharp_altitude(sharp=120, altitude=75) == (13.064, 7.435)
    assert transpose_sharp_altitude(sharp=150, altitude=30) == (40.893, 48.590)


def test_f_svv() -> None:
    """Test the f_svv function for calculating the clothing insulation factor."""
    assert np.isclose(round(f_svv(30, 10, 3.3), 2), 0.27, atol=1e-09)
    assert np.isclose(round(f_svv(150, 10, 3.3), 2), 0.31, atol=1e-09)
    assert np.isclose(round(f_svv(30, 6, 3.3), 2), 0.20, atol=1e-09)
    assert np.isclose(round(f_svv(150, 6, 3.3), 2), 0.23, atol=1e-09)
    assert np.isclose(round(f_svv(30, 10, 6), 2), 0.17, atol=1e-09)
    assert np.isclose(round(f_svv(150, 10, 6), 2), 0.21, atol=1e-09)
    assert np.isclose(round(f_svv(30, 6, 6), 2), 0.11, atol=1e-09)
    assert np.isclose(round(f_svv(150, 6, 6), 2), 0.14, atol=1e-09)
    assert np.isclose(round(f_svv(6, 9, 3.3), 2), 0.14, atol=1e-09)
    assert np.isclose(round(f_svv(6, 6, 3.3), 2), 0.11, atol=1e-09)
    assert np.isclose(round(f_svv(6, 6, 6), 2), 0.04, atol=1e-09)
    assert np.isclose(round(f_svv(4, 4, 3.3), 2), 0.06, atol=1e-09)
    assert np.isclose(round(f_svv(4, 4, 6), 2), 0.02, atol=1e-09)


def test_running_mean_outdoor_temperature() -> None:
    """Test the running mean outdoor temperature function."""
    assert (running_mean_outdoor_temperature([20, 20], alpha=0.7)) == 20
    assert (running_mean_outdoor_temperature([20, 20], alpha=0.9)) == 20
    assert (running_mean_outdoor_temperature([20, 20, 20, 20], alpha=0.7)) == 20
    assert (running_mean_outdoor_temperature([20, 20, 20, 20], alpha=0.5)) == 20
    temperatures_ip = [77, 77, 77, 77, 77, 77, 77]
    assert (
        running_mean_outdoor_temperature(
            temperatures_ip,
            alpha=0.8,
            units=Units.IP.value,
        )
    ) == 77
    assert temperatures_ip == [77, 77, 77, 77, 77, 77, 77]
    assert (
        running_mean_outdoor_temperature(
            temperatures_ip,
            alpha=0.8,
            units=Units.IP.value,
        )
    ) == 77


def test_v_relative() -> None:
    """Test the v_relative function for calculating relative air speed."""
    # Test case when met is equal to or lower than 1
    v = 2.0
    met = 1.0
    expected_result = v
    assert np.allclose(v_relative(v, met), expected_result)

    # Test case when met is greater than 1
    v = np.asarray([1.0, 2.0, 3.0])
    met = 2.0
    expected_result = np.asarray([1.3, 2.3, 3.3])
    assert np.allclose(v_relative(v, met), expected_result, atol=1e-6)

    # Test case with negative values for v
    v = -1.5
    met = 1.5
    expected_result = -1.5 + 0.3 * 0.5
    assert np.allclose(v_relative(v, met), expected_result, atol=1e-6)


def test_t_o() -> None:
    """Test the operative temperature function with various inputs."""
    assert operative_tmp(25, 25, 0.1) == 25
    assert np.allclose(
        operative_tmp([25, 20], 30, 0.3),
        [26.83, 23.66],
        atol=1e-2,
    )
    assert operative_tmp(25, 25, 0.1, standard="ASHRAE") == 25
    assert operative_tmp(20, 30, 0.1, standard="ASHRAE") == 25
    assert operative_tmp(20, 30, 0.3, standard="ASHRAE") == 24
    assert operative_tmp(20, 30, 0.7, standard="ASHRAE") == 23


def test_t_mrt() -> None:
    """Test the mean radiant temperature function with various inputs."""
    assert np.allclose(
        mean_radiant_tmp(
            tg=[53.2, 55, 55],
            tdb=30,
            v=[0.3, 0.3, 0.1],
            d=0.1,
            standard="ISO",
        ),
        [74.8, 77.8, 71.9],
        atol=1e-1,
    )
    assert np.allclose(
        mean_radiant_tmp(
            tg=[25.42, 26.42, 26.42, 26.42],
            tdb=26.10,
            v=0.1931,
            d=[0.1, 0.1, 0.5, 0.03],
            standard="Mixed Convection",
        ),
        [24.2, 27.0, np.nan, np.nan],
        atol=1e-1,
        equal_nan=True,
    )


def test_compare_results_wind_profile_calculator() -> None:
    """Compare results with Wind Profile Calculator online tool.

    Reference:
    https://wind-data.ch/tools/profile.php?h=2&v=10&z0=0.01&abfrage=Refresh
    """
    v10 = 6.52  # m/s
    z1 = 10  # m
    z2 = 2.0  # m
    z0 = 0.01  # m
    expected = 5  # m/s from Wind Profile Calculator
    result = scale_wind_speed_log(v_z1=v10, z2=z2, z1=z1, z0=z0, round_output=False)
    assert np.allclose(result.v_z2, expected, rtol=1e-2)

    v10 = 7.69  # m/s
    z2 = 2.0  # m
    z0 = 0.1  # m
    expected = 5  # m/s from Wind Profile Calculator
    result = scale_wind_speed_log(v_z1=v10, z2=z2, z1=10, z0=z0, round_output=False)
    assert np.allclose(result.v_z2, expected, rtol=1e-2)

    v_z1 = 5.0  # m/s
    z2 = 90.0  # m
    z0 = 0.1  # m
    expected = 7.39  # m/s from Wind Profile Calculator
    result = scale_wind_speed_log(v_z1=v_z1, z2=z2, z1=10, z0=z0, round_output=False)
    assert np.allclose(result.v_z2, expected, rtol=1e-2)


def test_scale_winds_speed_scalar() -> None:
    """Test scaling wind speed from 10m to 2m (scalar inputs)."""
    v10 = 5.0
    z2 = 2.0
    expected = v10 * np.log((z2 - 0.0) / 0.01) / np.log((10.0 - 0.0) / 0.01)
    result = scale_wind_speed_log(v10, z2, round_output=False)
    assert np.allclose(result.v_z2, expected, rtol=1e-5)


def test_scale_winds_speed_array() -> None:
    """Test scaling wind speed for array inputs."""
    v10 = np.asarray([3.0, 5.0])
    z2 = np.asarray([1.5, 2.5])
    expected = v10 * np.log((z2 - 0.0) / 0.01) / np.log((10.0 - 0.0) / 0.01)
    result = scale_wind_speed_log(v10, z2, round_output=False)
    assert np.allclose(result.v_z2, expected, rtol=1e-5)


def test_scale_wind_speed_broadcasting() -> None:
    """Test broadcasting with different z0 for each measurement."""
    v10 = [3.0, 5.0]
    z2 = [1.5, 2.5]
    z0 = [0.01, 0.1]
    expected = np.asarray(
        [
            3.0 * np.log((1.5 - 0.0) / 0.01) / np.log((10.0 - 0.0) / 0.01),
            5.0 * np.log((2.5 - 0.0) / 0.1) / np.log((10.0 - 0.0) / 0.1),
        ]
    )
    result = scale_wind_speed_log(v10, z2, z0=z0, round_output=False)
    assert np.allclose(result.v_z2, expected, rtol=1e-5)


def test_scale_winds_speed_with_displacement() -> None:
    """Test with nonzero displacement height d."""
    v10 = 5.0
    z2 = 2.0
    z1 = 10.0
    z0 = 0.1
    d = 0.5
    expected = v10 * np.log((z2 - d) / z0) / np.log((z1 - d) / z0)
    result = scale_wind_speed_log(v10, z2, z1=z1, z0=z0, d=d, round_output=False)
    assert np.allclose(result.v_z2, expected, rtol=1e-5)


def test_invalid_types() -> None:
    """Test that invalid types raise TypeError."""
    with pytest.raises(TypeError):
        scale_wind_speed_log("bad", 2.0)
    with pytest.raises(TypeError):
        scale_wind_speed_log(5.0, "bad")
    with pytest.raises(TypeError):
        scale_wind_speed_log(5.0, 2.0, z0="bad")


def test_negative_and_zero_values() -> None:
    """Test that negative and zero values raise ValueError."""
    # Negative wind speed
    with pytest.raises(ValueError):
        scale_wind_speed_log(-1.0, 2.0, round_output=False)
    # Negative z2
    with pytest.raises(ValueError):
        scale_wind_speed_log(5.0, -2.0, round_output=False)
    # z0 <= 0
    with pytest.raises(ValueError):
        scale_wind_speed_log(5.0, 2.0, z0=0.0, round_output=False)
    with pytest.raises(ValueError):
        scale_wind_speed_log(5.0, 2.0, z0=-0.1, round_output=False)
    # z2 <= d
    with pytest.raises(ValueError):
        scale_wind_speed_log(5.0, 2.0, d=2.0, round_output=False)
    # z1 <= d
    with pytest.raises(ValueError):
        scale_wind_speed_log(5.0, 2.0, z1=1.0, d=1.0, round_output=False)
    # z2 <= z0
    with pytest.raises(ValueError):
        scale_wind_speed_log(5.0, 0.01, z0=0.01, round_output=False)
    # z1 <= z0
    with pytest.raises(ValueError):
        scale_wind_speed_log(5.0, 2.0, z1=0.01, z0=0.01, round_output=False)


def test_edge_case_z2_less_than_z1() -> None:
    """Test scaling when z2 < z1 (should still work if all constraints are met)."""
    v10 = 5.0
    z2 = 2.0
    z1 = 10.0
    result = scale_wind_speed_log(v10, z2, z1=z1, round_output=False)
    assert result.v_z2 < v10


def test_large_and_small_z0() -> None:
    """Test with very large and very small roughness lengths."""
    v10 = 5.0
    z2 = 2.0
    # Very small z0
    result_small = scale_wind_speed_log(v10, z2, z0=1e-6, round_output=False)
    assert result_small.v_z2 > 0
    # Very large z0 (should be close to zero wind speed)
    result_large = scale_wind_speed_log(v10, z2, z0=1.0, round_output=False)
    assert result_large.v_z2 > 0


@pytest.mark.parametrize(
    "module_path",
    [
        pytest.param("pythermalcomfort.utils", id="package"),
        pytest.param("pythermalcomfort.utils.scale_wind_speed_log", id="submodule"),
    ],
)
def test_legacy_scale_wind_speed_log_shims(module_path: str) -> None:
    """Both former utils import paths return the expected value and warn."""
    from importlib import import_module

    legacy_scale = import_module(module_path).scale_wind_speed_log

    assert legacy_scale.__doc__ == (
        "Deprecated alias for pythermalcomfort.environment.scale_wind_speed_log(). "
        "Import from there instead; this path will be removed after two minor releases."
    )

    with pytest.warns(DeprecationWarning, match="pythermalcomfort.environment"):
        legacy = legacy_scale(5.0, 2.0, round_output=False)

    assert legacy.v_z2 == pytest.approx(3.83504999)
