from __future__ import annotations

import numpy as np
import pytest

from pythermalcomfort.classes_input import WorkIntensity
from pythermalcomfort.classes_return import WorkCapacity
from pythermalcomfort.models import work_capacity_hothaps


def _expected_capacity(
    wbgt: float | list[float],
    divisor: float,
    exponent: float,
) -> np.ndarray:
    """Compute the expected work capacity based on WBGT, divisor, and exponent.

    Parameters
    ----------
    wbgt: float
        Wet Bulb Globe Temperature value(s).
    divisor:
        Divisor parameter for the capacity formula.
    exponent:
        Exponent parameter for the capacity formula.

    Returns
    -------
        Numpy array of capacity values, clipped between 0 and 100.

    """
    wbgt_array = np.asarray(wbgt)
    cap = 100 * (0.1 + 0.9 / (1 + (wbgt_array / divisor) ** exponent))
    return cap.clip(0, 100)


def test_scalar_heavy() -> None:
    """Test that a scalar heavy work intensity produces the expected capacity."""
    result = work_capacity_hothaps(30.0, work_intensity=WorkIntensity.HEAVY.value)
    assert isinstance(result, WorkCapacity)
    assert isinstance(result.capacity, float)
    expected = _expected_capacity(30.0, divisor=30.94, exponent=16.64)
    assert pytest.approx(expected, rel=1e-3) == result.capacity


def test_scalar_moderate() -> None:
    """Test that a scalar moderate work intensity produces the expected capacity."""
    result = work_capacity_hothaps(30.0, work_intensity=WorkIntensity.MODERATE.value)
    assert isinstance(result, WorkCapacity)
    assert isinstance(result.capacity, float)
    expected = _expected_capacity(30.0, divisor=32.93, exponent=17.81)
    assert pytest.approx(expected, rel=1e-3) == result.capacity


def test_scalar_light() -> None:
    """Test that a scalar light work intensity produces the expected capacity."""
    result = work_capacity_hothaps(30.0, work_intensity=WorkIntensity.LIGHT.value)
    assert isinstance(result, WorkCapacity)
    assert isinstance(result.capacity, float)
    expected = _expected_capacity(30.0, divisor=34.64, exponent=22.72)
    assert pytest.approx(expected, rel=1e-3) == result.capacity


def test_list_input() -> None:
    """Test that the function handles list inputs correctly."""
    wbgts = [20.0, 40.0]
    result = work_capacity_hothaps(wbgts, work_intensity="heavy")
    assert isinstance(result.capacity, np.ndarray)
    exp_list = _expected_capacity(wbgts, divisor=30.94, exponent=16.64).tolist()
    assert all(
        pytest.approx(e, rel=1e-3) == r
        for e, r in zip(exp_list, result.capacity, strict=False)
    )


def test_low_wbgt_clamped_to_100() -> None:
    """Test that the function clamps low WBGT values to 100% capacity."""
    result = work_capacity_hothaps(0.0, work_intensity="light")
    assert result.capacity == pytest.approx(100.0, rel=1e-6)


def test_high_wbgt_approaches_10_percent() -> None:
    """Test that the function returns approximately 10% capacity for high WBGT."""
    result = work_capacity_hothaps(100.0, work_intensity="moderate")
    assert result.capacity == pytest.approx(10.0, rel=1e-3)


def test_invalid_intensity_raises() -> None:
    """Test that the function raises ValueError for invalid work intensity."""
    with pytest.raises(ValueError):
        work_capacity_hothaps(30.0, work_intensity="invalid")


def test_work_intensity_accepts_enum() -> None:
    """A WorkIntensity member gives the same result as its string value (#470)."""
    assert (
        work_capacity_hothaps(wbgt=30, work_intensity=WorkIntensity.MODERATE).capacity
        == work_capacity_hothaps(wbgt=30, work_intensity="moderate").capacity
    )
