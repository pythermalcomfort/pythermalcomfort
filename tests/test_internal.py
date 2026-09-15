import numpy as np
import pytest

import pythermalcomfort._internal.validation as validation
from pythermalcomfort._internal.ashrae55 import _check_ashrae55_compliance
from pythermalcomfort._internal.validation import _mapping, _valid_range, validate_type
from pythermalcomfort.classes_input import BaseInputs


class TestValidRange:
    """Tests for _valid_range warning behaviour."""

    def test_scalar_out_of_range_warns(self) -> None:
        """Scalar value outside range triggers UserWarning with the value."""
        with pytest.warns(
            UserWarning, match=r"'tdb' has value 50\.0.*\[10\.0, 40\.0\]"
        ):
            result = _valid_range(50.0, (10.0, 40.0), "tdb")
        assert np.isnan(result)

    def test_array_out_of_range_warns(self) -> None:
        """Array with out-of-range values triggers UserWarning with count, values, and indices."""
        with pytest.warns(
            UserWarning,
            match=r"'tdb' has 2 values \[50\.0, 45\.0\] at indices \[1, 3\].*\[10\.0, 40\.0\]",
        ):
            result = _valid_range([20.0, 50.0, 30.0, 45.0], (10.0, 40.0), "tdb")
        assert np.isnan(result[1]) and np.isnan(result[3])
        assert result[0] == 20.0 and result[2] == 30.0

    def test_in_range_no_warning(self, recwarn) -> None:
        """Values within range produce no warning."""
        result = _valid_range(25.0, (10.0, 40.0), "tdb")
        assert len(recwarn) == 0
        assert result == 25.0

    def test_no_param_name_auto_extracts_from_caller(self) -> None:
        """Without ``param_name``, the caller's variable name is auto-extracted."""
        tdb = 50.0
        with pytest.warns(
            UserWarning, match=r"'tdb' has value 50\.0.*\[10\.0, 40\.0\]"
        ):
            result = _valid_range(tdb, (10.0, 40.0))
        assert np.isnan(result)

    def test_no_param_name_literal_falls_back_to_unknown(self) -> None:
        """Literal first arg cannot be auto-named; warning falls back to ``<unknown>``."""
        with pytest.warns(
            UserWarning, match=r"'<unknown>' has value 50\.0.*\[10\.0, 40\.0\]"
        ):
            result = _valid_range(50.0, (10.0, 40.0))
        assert np.isnan(result)


def test_internal_renames_have_no_compatibility_aliases() -> None:
    """Internal helpers expose only their new private names."""
    assert not hasattr(validation, "valid_range")
    assert not hasattr(validation, "mapping")
    assert _mapping([10, 20, 30], {15: "low", 25: "medium", 35: "high"}).tolist() == [
        "low",
        "medium",
        "high",
    ]


class TestCheckAshrae55Compliance:
    """Tests for _check_ashrae55_compliance warning behaviour (airspeed_control=False)."""

    def test_airspeed_control_cond1_warns(self) -> None:
        """cond1: v > 0.8 with clo < 0.7 and met < 1.3 triggers UserWarning.

        tdb=tr=30 → to=30 > 25.5, so cond2 does not trigger alongside cond1.
        """
        with pytest.warns(UserWarning, match=r"exceeding 0\.8 m/s") as record:
            _check_ashrae55_compliance(
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
            _check_ashrae55_compliance(
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
            _check_ashrae55_compliance(
                tdb=np.float64(22),
                tr=np.float64(22),
                v=np.float64(0.3),
                met=np.float64(1.2),
                clo=np.float64(0.5),
                airspeed_control=False,
            )

    def test_airspeed_control_true_no_condition_warning(self, recwarn) -> None:
        """airspeed_control=True skips cond1/cond2/cond3 checks even when v=1.0."""
        _check_ashrae55_compliance(
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
    normalized_float = validate_type(np.float32(40.0), "np_float32", allowed)
    normalized_int32 = validate_type(np.int32(100), "np_int32", allowed)
    normalized_int64 = validate_type(np.int64(200), "np_int64", allowed)
    assert normalized_float == 40.0
    assert normalized_int32 == 100
    assert normalized_int64 == 200
    assert not isinstance(normalized_float, np.generic)
    assert not isinstance(normalized_int32, np.generic)
    assert not isinstance(normalized_int64, np.generic)

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


def test_base_inputs_store_normalized_numpy_scalar() -> None:
    """NumPy scalar inputs are stored as their native Python equivalents."""
    inputs = BaseInputs(tdb=np.float32(25.0))

    assert inputs.tdb == 25.0
    assert not isinstance(inputs.tdb, np.generic)
