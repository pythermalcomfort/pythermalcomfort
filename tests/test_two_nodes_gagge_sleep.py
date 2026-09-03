import numpy as np
import pytest

from pythermalcomfort.classes_return import GaggeTwoNodesSleep
from pythermalcomfort.models import two_nodes_gagge_sleep
from pythermalcomfort.models.two_nodes_gagge_sleep import (
    _two_nodes_gagge_sleep_optimized,
)


def test_two_nodes_gagge_sleep_single_input() -> None:
    """Test the two_nodes_gagge_sleep function with scalar inputs."""
    result = two_nodes_gagge_sleep(18, 18, 0.05, 50, 1.4, thickness_quilt=1.76)

    # expected outputs
    expected = {
        "set": np.asarray([24.28]),
        "t_core": np.asarray([37.03]),
        "t_skin": np.asarray([33.67]),
        "wet": np.asarray([0.27]),
        "t_sens": np.asarray([1.12]),
        "disc": np.asarray([1.47]),
        "e_skin": np.asarray([28.75]),
        "met_shivering": np.asarray([0.0]),
        "alfa": np.asarray([0.13]),
        "skin_blood_flow": np.asarray([7.11]),
    }

    # compare each field with a reasonable tolerance
    for field, exp in expected.items():
        actual = getattr(result, field)
        np.testing.assert_allclose(actual, exp, atol=0.01, rtol=0.005)


def test_two_nodes_gagge_sleep_long_duration() -> None:
    """Test the two_nodes_gagge_sleep function with a longer duration input."""
    duration = 481
    ta = np.repeat(18, duration)
    tr = np.repeat(18, duration)
    vel = np.repeat(0.05, duration)
    rh = np.repeat(50, duration)
    clo_a = np.repeat(1.4, duration)
    thickness1 = np.repeat(1.76, duration)

    result = two_nodes_gagge_sleep(ta, tr, vel, rh, clo_a, thickness1)

    # Assert return type and shape
    assert isinstance(result, GaggeTwoNodesSleep)
    assert result.set.shape == (duration,)
    assert result.t_core.shape == (duration,)
    assert result.t_skin.shape == (duration,)

    # for field in fields:
    #     print(f"{getattr(result, field)[-1]:.2f},")

    first_row_expected = [
        24.29,
        37.03,
        33.67,
        0.27,
        1.13,
        1.48,
        28.76,
        0.00,
        0.14,
        7.11,
    ]
    last_row_expected = [
        22.23,
        36.23,
        31.06,
        0.06,
        -0.73,
        -0.73,
        5.21,
        0.00,
        0.25,
        2.98,
    ]

    fields = [
        "set",
        "t_core",
        "t_skin",
        "wet",
        "t_sens",
        "disc",
        "e_skin",
        "met_shivering",
        "alfa",
        "skin_blood_flow",
    ]

    # check first row
    for field, exp in zip(fields, first_row_expected, strict=False):
        np.testing.assert_allclose(
            getattr(result, field)[0],
            exp,
            atol=0.01,
            rtol=0.005,
            err_msg=f"first {field} mismatch",
        )

    # check last row
    for field, exp in zip(fields, last_row_expected, strict=False):
        np.testing.assert_allclose(
            getattr(result, field)[-1],
            exp,
            atol=0.01,
            rtol=0.005,
            err_msg=f"last {field} mismatch",
        )


def test_two_nodes_gagge_sleep_numba_full_time_series_parity() -> None:
    """Pin the complete optimized time series to the pre-Numba implementation."""
    result = two_nodes_gagge_sleep(
        tdb=[18, 19, 20],
        tr=[18, 18.5, 19],
        v=[0.05, 0.1, 0.15],
        rh=[50, 55, 60],
        clo=[1.4, 1.3, 1.2],
        thickness_quilt=[1.76, 1.6, 1.5],
    )

    expected = {
        "set": [
            24.285471240487254,
            23.69753060178574,
            23.55973896722113,
        ],
        "t_core": [
            37.03223590335804,
            37.026301386399254,
            37.022232367376795,
        ],
        "t_skin": [
            33.670550615860265,
            33.56972917281989,
            33.522385202322575,
        ],
        "wet": [
            0.2709382101612528,
            0.11868202497357694,
            0.08468947408201234,
        ],
        "t_sens": [
            1.1283814676194304,
            0.2030740970076956,
            -0.00007100712419982713,
        ],
        "disc": [
            1.479755896275013,
            0.29422228310400694,
            0.02764862771183254,
        ],
        "e_skin": [
            28.76139367298244,
            12.562622189484204,
            9.061735937484093,
        ],
        "met_shivering": [0, 0, 0],
        "alfa": [
            0.13861663831083165,
            0.14512664326204494,
            0.14635629215748827,
        ],
        "skin_blood_flow": [
            7.1093443630662945,
            6.624666007608543,
            6.5398921420592,
        ],
    }

    for field, reference in expected.items():
        np.testing.assert_allclose(
            getattr(result, field),
            reference,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"full time-series {field} mismatch",
        )

    assert _two_nodes_gagge_sleep_optimized.nopython_signatures


def test_two_nodes_gagge_sleep_scalar_shape_is_preserved() -> None:
    """The compiled array kernel must not change the scalar public API."""
    result = two_nodes_gagge_sleep(18, 18, 0.05, 50, 1.4, 1.76)

    for field in result.__dataclass_fields__:
        assert np.ndim(getattr(result, field)) == 0


def test_length_mismatch_raises_value_error() -> None:
    """Test that length mismatch in input lists raises ValueError."""
    with pytest.raises(ValueError) as exc:
        two_nodes_gagge_sleep(
            [18, 18],
            [18],  # length mismatch
            [0.05, 0.05],
            [50, 50],
            [1.4, 1.4],
            [1.76, 1.76],
        )

    assert "must have the same length" in str(exc.value)


def test_unexpected_kwargs_raises_type_error() -> None:
    """Test that unexpected keyword arguments raise TypeError."""
    with pytest.raises(TypeError) as exc:
        two_nodes_gagge_sleep(18, 18, 0.05, 50, 1.4, 1.76, foo=123)
    assert "Unexpected keyword arguments" in str(exc.value)


def test_invalid_kwarg_type_raises_type_error() -> None:
    """Test that a non-numeric type for tdb raises TypeError."""
    with pytest.raises(TypeError) as exc:
        two_nodes_gagge_sleep("string", 18, 0.05, 50, 1.4, 1.76)
    msg = str(exc.value)
    assert "tdb" in msg


def test_tickness_quilt_negative() -> None:
    """Test that a negative thickness_quilt raises ValueError."""
    with pytest.raises(ValueError):
        two_nodes_gagge_sleep(18, 18, 0.05, 50, 1.4, thickness_quilt=-1.76)
