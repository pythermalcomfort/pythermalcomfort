import numpy as np
import pytest

from pythermalcomfort.shared_functions import valid_range
from pythermalcomfort.utilities import (
    Units,
    _check_standard_compliance_array,
    body_surface_area,
    clo_area_factor,
    clo_correction_factor_environment,
    clo_dynamic_ashrae,
    clo_dynamic_iso,
    clo_insulation_air_layer,
    clo_intrinsic_insulation_ensemble,
    clo_total_insulation,
    f_svv,
    running_mean_outdoor_temperature,
    transpose_sharp_altitude,
    units_converter,
    v_relative,
    validate_type,
)


def test_intrinsic_insulation_ensemble() -> None:
    """Test the intrinsic insulation ensemble function."""
    assert clo_intrinsic_insulation_ensemble([0.5, 0.5]) == 0.835 + 0.161
    assert clo_intrinsic_insulation_ensemble([1, 1]) == 2 * 0.835 + 0.161
    assert clo_intrinsic_insulation_ensemble(2) == 2 * 0.835 + 0.161
    assert clo_intrinsic_insulation_ensemble([0]) == 0.161


def test_clo_area_factor() -> None:
    """Test the clothing area factor function."""
    assert clo_area_factor(1) == 1.28
    assert np.allclose(clo_area_factor(i_cl=[1, 2]), np.asarray([1.28, 1.56]))


def test_clo_air_layer_insulation() -> None:
    """Test the clothing insulation air layer function."""
    assert np.isclose(
        clo_insulation_air_layer(vr=1, v_walk=1, i_a_static=0.71),
        0.365,
        atol=0.001,
    )
    assert np.isclose(
        clo_insulation_air_layer(vr=0.2, v_walk=1, i_a_static=0.71),
        0.532,
        atol=0.001,
    )
    assert np.allclose(
        clo_insulation_air_layer(vr=[0.2, 1], v_walk=1, i_a_static=0.71),
        [0.532, 0.365],
        atol=0.001,
    )


def test_clo_total_insulation() -> None:
    """Test the total clothing insulation function."""
    assert np.allclose(
        clo_total_insulation(
            i_t=[1.21, 1.26, 1.56],
            vr=0.15,
            v_walk=0,
            i_a_static=0.5,
            i_cl=[0.61, 0.71, 1.01],
        ),
        [1.21, 1.26, 1.56],
        atol=0.001,
    )

    # compare the normal_clothing results with the figure in the standard
    assert np.allclose(
        clo_total_insulation(
            i_t=[1.21, 1.26, 1.56],
            vr=2,
            v_walk=[1, 0.5, 0.25],
            i_a_static=0.5,
            i_cl=[0.61, 0.71, 1.01],
        ),
        [1.21 * 0.5, 1.26 * 0.565, 1.56 * 0.62],
        atol=0.005,
    )

    # test that the nude function works as expected
    assert np.allclose(
        clo_total_insulation(
            i_t=0,
            vr=0.15,
            v_walk=0,
            i_a_static=[0.71, 0.61, 0.5],
            i_cl=0,
        ),
        [0.71, 0.61, 0.5],
        atol=0.001,
    )

    # compare the nude results with the figure in the standard
    assert np.allclose(
        clo_total_insulation(
            i_t=0,
            vr=[0.5, 2, 3],
            v_walk=0.5,
            i_a_static=[0.71, 0.61, 0.5],
            i_cl=0,
        ),
        [0.71 * 0.7, 0.61 * 0.4, 0.50 * 0.32],
        atol=0.004,
    )

    # test that the low_clothing function works as expected
    assert np.allclose(
        clo_total_insulation(
            i_t=[1.2, 0.6],
            vr=0.15,
            v_walk=0,
            i_a_static=[0.6, 0.6],
            i_cl=[0.6, 0],
        ),
        [1.2, 0.6],
        atol=0.001,
    )

    clo = 0.3
    i_a = 0.7
    np.isclose(
        clo_total_insulation(
            i_t=clo + i_a,
            vr=0.26,
            v_walk=0.06,
            i_a_static=i_a,
            i_cl=clo,
        ),
        0.79,
        atol=0.01,
    )


def test_clo_correction_factor_environment() -> None:
    """Test the clothing correction factor for environment function."""
    assert np.allclose(
        clo_correction_factor_environment(
            vr=0.15,
            v_walk=0,
            i_cl=[0.61, 0.71, 1.01],
        ),
        [1, 1, 1],
        atol=0.001,
    )

    # compare the normal_clothing results with the figure in the standard
    assert np.allclose(
        clo_correction_factor_environment(
            vr=2,
            v_walk=[1, 0.5, 0.25],
            i_cl=[0.61, 0.71, 1.01],
        ),
        [0.503, 0.564, 0.618],
        atol=0.001,
    )

    # test that the nude function works as expected
    assert np.allclose(
        clo_correction_factor_environment(
            vr=0.15,
            v_walk=0,
            i_cl=0,
        ),
        [1],
        atol=0.001,
    )

    # compare the nude results with the figure in the standard
    assert np.allclose(
        clo_correction_factor_environment(
            vr=[0.5, 2, 3],
            v_walk=0.5,
            i_cl=0,
        ),
        [0.698, 0.394, 0.320],
        atol=0.001,
    )

    # test that the low_clothing function works as expected
    assert np.allclose(
        clo_correction_factor_environment(
            vr=0.15,
            v_walk=0,
            i_cl=[0.6, 0],
        ),
        [1, 1],
        atol=0.001,
    )


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
    assert (
        running_mean_outdoor_temperature(
            [77, 77, 77, 77, 77, 77, 77],
            alpha=0.8,
            units=Units.IP.value,
        )
    ) == 77
    assert (
        running_mean_outdoor_temperature(
            [77, 77, 77, 77, 77, 77, 77],
            alpha=0.8,
            units=Units.IP.value,
        )
    ) == 77


def test_ip_units_converter() -> None:
    """Test the units converter for IP and SI units."""
    assert (units_converter(tdb=77, tr=77, v=3.2, from_units=Units.IP.value)) == [
        25.0,
        25.0,
        0.975312404754648,
    ]
    assert (units_converter(pressure=1, area=1 / 0.09, from_units=Units.IP.value)) == [
        101325,
        1.0322474090590033,
    ]

    expected_result = [25.0, 3.047]
    assert np.allclose(
        units_converter(Units.IP.value, tdb=77, v=10),
        expected_result,
        atol=0.01,
    )

    # Test case 2: Conversion from SI to IP for temperature and velocity
    expected_result = [68, 6.562]
    assert np.allclose(
        units_converter(Units.SI.value, tdb=20, v=2),
        expected_result,
        atol=0.01,
    )

    # Test case 3: Conversion from IP to SI for area and pressure
    expected_result = [9.29, 1489477.5]
    assert np.allclose(
        units_converter(Units.IP.value, area=100, pressure=14.7),
        expected_result,
        atol=0.01,
    )

    # Test case 4: Conversion from SI to IP for area and pressure
    expected_result = [538.199, 1]
    assert np.allclose(
        units_converter(Units.SI.value, area=50, pressure=101325),
        expected_result,
        atol=0.01,
    )


def test_clo_dynamic_ashrae() -> None:
    """Test the dynamic clothing insulation function for ASHRAE standards."""
    assert clo_dynamic_ashrae(clo=1, met=1) == 1
    assert clo_dynamic_ashrae(clo=1, met=0.5) == 1
    assert clo_dynamic_ashrae(clo=2, met=0.5) == 2
    assert np.allclose(clo_dynamic_ashrae(1.0, 1.0), np.asarray(1))
    assert np.allclose(clo_dynamic_ashrae(1.0, 1.2), np.asarray(1))
    assert np.allclose(clo_dynamic_ashrae(1.0, 2.0), np.asarray(0.8))

    # Test invalid standard input
    with pytest.raises(ValueError):
        clo_dynamic_ashrae(1.0, 1.0, model="invalid")


def test_clo_dynamic_iso() -> None:
    """Test the dynamic clothing insulation function for ISO standards."""
    assert np.isclose(clo_dynamic_iso(clo=1, met=1, v=0.2), 0.99, atol=0.01)
    assert np.allclose(
        clo_dynamic_iso(clo=[1, 1.5], met=1, v=0.2),
        [0.99, 1.48],
        atol=0.01,
    )
    assert np.allclose(
        clo_dynamic_iso(clo=[1, 1.5], met=1, v=0.2),
        [0.99, 1.48],
        atol=0.01,
    )
    assert np.allclose(
        clo_dynamic_iso(
            clo=[0.95, 1.07, 0.88, 0.59, 0.83, 0.66, 1.02, 0.71, 1.1, 0.68, 0.3],
            met=[1.71, 1.11, 1.21, 1.77, 1.48, 1.5, 1.33, 1.33, 1.26, 1.47, 1.27],
            v=[0.03, 0.08, 0.04, 0.03, 0.15, 0.15, 0.06, 0.03, 0.25, 0.05, 0.12],
        ),
        [0.85, 1.06, 0.86, 0.52, 0.76, 0.61, 0.97, 0.68, 1.03, 0.63, 0.17],
        atol=0.01,
    )

    # Test invalid standard input
    with pytest.raises(ValueError):
        clo_dynamic_iso(1.0, 1.0, v=0.2, model="invalid")


def test_body_surface_area() -> None:
    """Test the body surface area calculations with various formulas."""
    assert body_surface_area(weight=80, height=1.8) == pytest.approx(1.9917, rel=1e-2)
    assert body_surface_area(70, 1.8, "dubois") == pytest.approx(1.88, rel=1e-2)
    assert body_surface_area(75, 1.75, "takahira") == pytest.approx(1.91, rel=1e-2)
    assert body_surface_area(80, 1.7, "fujimoto") == pytest.approx(1.872, rel=1e-2)
    assert body_surface_area(85, 1.65, "kurazumi") == pytest.approx(1.89, rel=1e-2)
    with pytest.raises(ValueError):
        body_surface_area(70, 1.8, "invalid_formula")


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


class TestValidRange:
    """Tests for valid_range warning behaviour."""

    def test_scalar_out_of_range_warns(self) -> None:
        """Scalar value outside range triggers UserWarning with the value."""
        with pytest.warns(
            UserWarning, match=r"Value of 'tdb' \(50\.0\).*\[10\.0, 40\.0\]"
        ):
            result = valid_range(50.0, (10.0, 40.0), "tdb")
        assert np.isnan(result)

    def test_array_out_of_range_warns(self) -> None:
        """Array with out-of-range values triggers UserWarning with indices and values."""
        with pytest.warns(
            UserWarning, match=r"'tdb'.*\[10\.0, 40\.0\].*\[50\.0, 45\.0\].*\[1, 3\]"
        ):
            result = valid_range([20.0, 50.0, 30.0, 45.0], (10.0, 40.0), "tdb")
        assert np.isnan(result[1]) and np.isnan(result[3])
        assert result[0] == 20.0 and result[2] == 30.0

    def test_in_range_no_warning(self, recwarn) -> None:
        """Values within range produce no warning."""
        result = valid_range(25.0, (10.0, 40.0), "tdb")
        assert len(recwarn) == 0
        assert result == 25.0

    def test_no_param_name_no_warning(self, recwarn) -> None:
        """Without param_name, no warning is issued even for out-of-range values."""
        result = valid_range(50.0, (10.0, 40.0))
        assert len(recwarn) == 0
        assert np.isnan(result)


class TestCheckStandardComplianceArray:
    """Tests for _check_standard_compliance_array warning behaviour (airspeed_control=False)."""

    def test_airspeed_control_cond1_warns(self) -> None:
        """cond1: v > 0.8 with clo < 0.7 and met < 1.3 triggers UserWarning.

        tdb=tr=30 → to=30 > 25.5, so cond2 does not trigger alongside cond1.
        """
        with pytest.warns(UserWarning, match=r"exceed 0.8 m/s") as record:
            _check_standard_compliance_array(
                "55-2023",
                tdb=np.float64(30),
                tr=np.float64(30),
                v=np.float64(1.0),
                met=np.float64(1.2),
                clo=np.float64(0.5),
                airspeed_control=False,
            )
        assert len(record) == 1

    def test_airspeed_control_cond2_warns(self) -> None:
        """cond2: v exceeds ASHRAE comfort-zone limit (23°C < to < 25.5°C) triggers UserWarning.

        With tdb=tr=24, to=24; v_limit ≈ 0.32; v=0.5 > v_limit triggers cond2.
        """
        with pytest.warns(UserWarning, match=r"comfort zone"):
            _check_standard_compliance_array(
                "55-2023",
                tdb=np.float64(24),
                tr=np.float64(24),
                v=np.float64(0.5),
                met=np.float64(1.2),
                clo=np.float64(0.5),
                airspeed_control=False,
            )

    def test_airspeed_control_cond3_warns(self) -> None:
        """cond3: v > 0.2 when to <= 23°C triggers UserWarning.

        With tdb=tr=22, to=22 <= 23; v=0.3 > 0.2 triggers cond3.
        """
        with pytest.warns(UserWarning, match=r"operative temperature is ≤ 23°C"):
            _check_standard_compliance_array(
                "55-2023",
                tdb=np.float64(22),
                tr=np.float64(22),
                v=np.float64(0.3),
                met=np.float64(1.2),
                clo=np.float64(0.5),
                airspeed_control=False,
            )

    def test_airspeed_control_true_no_condition_warning(self, recwarn) -> None:
        """airspeed_control=True skips cond1/cond2/cond3 checks even when v=1.0."""
        _check_standard_compliance_array(
            "55-2023",
            tdb=np.float64(25),
            tr=np.float64(25),
            v=np.float64(1.0),
            met=np.float64(1.2),
            clo=np.float64(0.5),
            airspeed_control=True,
        )
        assert len(recwarn) == 0


def test_validate_type() -> None:
    """Test the validate_type function for type validation."""
    allowed = (float, int, list, np.ndarray)

    # valid cases
    # native Python types
    validate_type(1, "int_value", allowed)
    validate_type(3.1415, "float_value", allowed)
    validate_type([1, 2, 3], "list_value", allowed)
    validate_type(np.asarray([1, 2, 3]), "array_value", allowed)

    # np scalars should be converted to native types via .item()
    validate_type(np.float32(40.0), "np_float32", allowed)
    validate_type(np.int32(100), "np_int32", allowed)
    validate_type(np.int64(200), "np_int64", allowed)

    # np array of floats and ints should be allowed
    arr_numeric = np.asarray([np.float32(1.0), np.int32(2), 3, 3.52])
    validate_type(arr_numeric, "arr_numeric", allowed)

    # empty NumPy array should also pass
    validate_type(np.asarray([]), "empty_array", allowed)

    # empty list should pass
    validate_type([], "empty_list", allowed)

    # --- Invalid cases ---

    with pytest.raises(TypeError) as exc_info:
        validate_type({"a": 1}, "dict_type", allowed)
    assert "dict_type must be one of the following types:" in str(exc_info.value)

    with pytest.raises(TypeError) as exc_info:
        validate_type("hello", "str_value", allowed)
    assert "str_value must be one of the following types:" in str(exc_info.value)

    with pytest.raises(TypeError) as exc_info:
        validate_type(np.str_("hello"), "np_str", allowed)
    assert "np_str must be one of the following types:" in str(exc_info.value)
