import numpy as np
import pytest

from pythermalcomfort.models.ridge_regression_predict_t_re_t_sk import (
    _inverse_scale_output,
    _scale_features,
    ridge_regression_predict_t_re_t_sk,
)
from pythermalcomfort.utilities import Sex


def test_ridge_regression_scale_features():
    """Test the feature scaling function."""
    # Create a sample feature array: [sex, age, height, mass, temp, humidity, tre, mtsk]
    features = np.array([[0, 30, 180, 75, 25, 50, 37.0, 34.0]])
    scaled = _scale_features(features)
    assert scaled.shape == (1, 8)
    # Check if the scaling is applied correctly by comparing with pre-calculated values
    expected = np.array(
        [
            [
                0.0,
                0.18032787,
                0.72916667,
                0.34182563,
                0.16666667,
                1.0,
                0.49972339,
                0.6549385,
            ]
        ]
    )
    np.testing.assert_allclose(scaled, expected, rtol=1e-6)


def test_ridge_regression_inverse_scale_output():
    """Test the inverse output scaling function."""
    # Create a sample scaled output array: [scaled_tre, scaled_mtsk]
    scaled_output = np.array([[0.54714297, 0.51841453]])
    inversed = _inverse_scale_output(scaled_output)
    assert inversed.shape == (1, 2)
    # Check if inverse scaling is applied correctly
    expected = np.array([[37.15, 32.15]])
    np.testing.assert_allclose(inversed, expected, rtol=1e-6)


@pytest.mark.parametrize("sex_in", [Sex.male, Sex.male.value])
def test_ridge_regression_scalar_parametrised(sex_in):
    """Test the model with scalar inputs."""
    duration = 540
    result = ridge_regression_predict_t_re_t_sk(
        sex=sex_in,
        age=60,
        height=1.80,
        weight=75,
        tdb=35,
        rh=60,
        duration=duration,
    )
    # Check that the output is a numpy array with the correct shape
    assert isinstance(result.t_re, np.ndarray)
    assert result.t_re.shape == (duration,)
    assert isinstance(result.t_sk, np.ndarray)
    assert result.t_sk.shape == (duration,)

    # Check the final value of the history against known values
    assert result.t_re[-1] == pytest.approx(38.15, abs=1e-2)
    assert result.t_sk[-1] == pytest.approx(37.02, abs=1e-2)


@pytest.mark.parametrize(("tdb", "shape"), [(35, (1,)), ([35, 40], (2, 1))])
def test_ridge_regression_duration_one_keeps_time_axis(tdb, shape):
    """Test that duration=1 keeps the time axis instead of giving a 0-d array."""
    inputs = {"sex": "male", "age": 70, "height": 1.8, "weight": 75, "rh": 60}
    result = ridge_regression_predict_t_re_t_sk(tdb=tdb, duration=1, **inputs)
    longer = ridge_regression_predict_t_re_t_sk(tdb=tdb, duration=2, **inputs)

    assert result.t_re.shape == shape
    assert result.t_sk.shape == shape
    np.testing.assert_array_equal(result.t_re, longer.t_re[..., :1])
    np.testing.assert_array_equal(result.t_sk, longer.t_sk[..., :1])


def test_ridge_regression_vectorized():
    """Test the model with array inputs for vectorization."""
    duration = 540
    sex = [Sex.male.value, Sex.female.value]
    age = [60, 65]
    height = [1.80, 1.65]
    weight = [75, 60]
    tdb = [35, 40]
    rh = [60, 50]

    result = ridge_regression_predict_t_re_t_sk(
        sex=sex,
        age=age,
        height=height,
        weight=weight,
        tdb=tdb,
        rh=rh,
        duration=duration,
    )
    # Check that the output is a numpy array with the correct shape
    assert isinstance(result.t_re, np.ndarray)
    assert result.t_re.shape == (2, duration)
    assert isinstance(result.t_sk, np.ndarray)
    assert result.t_sk.shape == (2, duration)

    # Check the final values of the history for each case
    final_t_re = result.t_re[:, -1]
    final_t_sk = result.t_sk[:, -1]

    assert final_t_re[0] == pytest.approx(38.15, abs=1e-2)
    assert final_t_sk[0] == pytest.approx(37.02, abs=1e-2)
    assert final_t_re[1] == pytest.approx(38.53, abs=1e-2)
    assert final_t_sk[1] == pytest.approx(38.04, abs=1e-2)


def test_ridge_regression_broadcasting():
    """Test NumPy broadcasting with mixed scalar and array inputs."""
    duration = 540
    sex = [Sex.male.value, Sex.female.value]
    age = 75  # scalar
    height = 1.70  # scalar
    weight = 70  # scalar
    tdb = [30, 35]
    rh = 50  # scalar

    result = ridge_regression_predict_t_re_t_sk(
        sex=sex,
        age=age,
        height=height,
        weight=weight,
        tdb=tdb,
        rh=rh,
        duration=duration,
    )
    # Check that the output is a numpy array with the correct shape
    assert result.t_re.shape == (2, duration)
    assert result.t_sk.shape == (2, duration)

    # Check the final values of the history for each broadcasted case
    final_t_re = result.t_re[:, -1]
    final_t_sk = result.t_sk[:, -1]

    assert final_t_re[0] == pytest.approx(37.84, abs=1e-2)
    assert final_t_sk[0] == pytest.approx(35.92, abs=1e-2)
    assert final_t_re[1] == pytest.approx(38.26, abs=1e-2)
    assert final_t_sk[1] == pytest.approx(37.02, abs=1e-2)

    # Ensure it raises error for incompatible shapes
    with pytest.raises(ValueError, match="broadcastable to a common shape"):
        ridge_regression_predict_t_re_t_sk(
            sex=[Sex.male.value, Sex.female.value],
            age=[30, 40, 50],  # Incompatible shape
            height=1.70,
            weight=70,
            tdb=30,
            rh=50,
            duration=duration,
        )


def test_ridge_regression_initial_body_temp():
    """Test the model with initial body temperatures provided."""
    duration = 60
    result = ridge_regression_predict_t_re_t_sk(
        sex=Sex.male.value,
        age=70,
        height=1.80,
        weight=75,
        tdb=35,
        rh=60,
        duration=duration,
        t_re=37.0,
        t_sk=32.0,
    )
    # Check that the output is a numpy array with the correct shape
    assert isinstance(result.t_re, np.ndarray)
    assert result.t_re.shape == (duration,)
    assert isinstance(result.t_sk, np.ndarray)
    assert result.t_sk.shape == (duration,)

    # Check the final value of the history against known values from the docstring example
    assert result.t_re[-1] == pytest.approx(37.33, abs=1e-2)
    assert result.t_sk[-1] == pytest.approx(37.00, abs=1e-2)


def test_ridge_regression_initial_t_broadcast_error():
    with pytest.raises(ValueError, match="broadcastable"):
        ridge_regression_predict_t_re_t_sk(
            sex=[Sex.male.value, Sex.female.value],
            age=[70, 75],
            height=[1.80, 1.70],
            weight=[75, 70],
            tdb=[35, 35],
            rh=[60, 60],
            duration=60,
            t_re=[37.0, 37.1, 37.2],  # bad shape
            t_sk=32.0,
        )


def test_invalid_sex():
    with pytest.raises(ValueError):
        ridge_regression_predict_t_re_t_sk(
            sex="unknown", age=70, height=1.8, weight=75, tdb=35, rh=60, duration=60
        )


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_inputs(bad):
    with pytest.raises(ValueError):
        ridge_regression_predict_t_re_t_sk(
            sex=Sex.male, age=70, height=1.8, weight=75, tdb=bad, rh=60, duration=60
        )


@pytest.mark.parametrize("bad_dur", [0, -5])
def test_invalid_duration(bad_dur):
    with pytest.raises(ValueError, match="positive integer"):
        ridge_regression_predict_t_re_t_sk(
            sex=Sex.male, age=70, height=1.8, weight=75, tdb=35, rh=60, duration=bad_dur
        )


def test_initial_t_singleton_rejected():
    with pytest.raises(ValueError, match="Both t_re and t_sk must be provided"):
        ridge_regression_predict_t_re_t_sk(
            sex=Sex.male,
            age=70,
            height=1.8,
            weight=75,
            tdb=35,
            rh=60,
            duration=60,
            t_re=37.0,
        )
    with pytest.raises(ValueError, match="Both t_re and t_sk must be provided"):
        ridge_regression_predict_t_re_t_sk(
            sex=Sex.male,
            age=70,
            height=1.8,
            weight=75,
            tdb=35,
            rh=60,
            duration=60,
            t_sk=32.0,
        )


def test_wrong_type_duration():
    with pytest.raises(ValueError):
        ridge_regression_predict_t_re_t_sk(
            sex=Sex.male,
            age=70,
            height=1.8,
            weight=75,
            tdb=35,
            rh=60,
            duration=True,
            t_sk=32.0,
        )

    with pytest.raises(TypeError):
        ridge_regression_predict_t_re_t_sk(
            sex=Sex.male,
            age=70,
            height=1.8,
            weight=75,
            tdb=35,
            rh=60,
            duration=60.2,
            t_sk=32.0,
        )
