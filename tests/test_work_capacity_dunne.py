import numpy as np
import pytest

from pythermalcomfort.classes_input import WorkIntensity
from pythermalcomfort.classes_return import WorkCapacity
from pythermalcomfort.models import work_capacity_dunne


def test_heavy_scalar_at_25() -> None:
    """Verify that a scalar WBGT of 25 with heavy intensity yields 100% capacity."""
    wc = work_capacity_dunne(25, "heavy")
    assert isinstance(wc, WorkCapacity)
    assert wc.capacity == pytest.approx(100)


def test_heavy_array_and_clipping() -> None:
    """Verify heavy intensity for an array of WBGT values and proper clipping at [0, 100]."""
    wbgt = [20, 25, 28, 30, 35]
    wc = work_capacity_dunne(wbgt, "heavy")
    expected_base = np.clip(
        100 - 25 * np.maximum(0, np.asarray(wbgt) - 25) ** (2 / 3),
        0,
        100,
    )
    assert isinstance(wc, WorkCapacity)
    assert np.allclose(wc.capacity, expected_base)


def test_moderate_and_light_intensity() -> None:
    """Verify moderate and light intensities scale the base capacity correctly."""
    wbgt = 30
    base = np.clip(100 - 25 * (5) ** (2 / 3), 0, 100)
    wc_med = work_capacity_dunne(wbgt, "moderate")
    wc_light = work_capacity_dunne(wbgt, "light")
    assert wc_med.capacity == pytest.approx(np.clip(base * 2, 0, 100))
    assert wc_light.capacity == pytest.approx(np.clip(base * 4, 0, 100))


def test_lower_bound_clipping() -> None:
    """Verify that capacity is clipped to 0 when WBGT is extremely high."""
    wc = work_capacity_dunne(100, "heavy")
    assert wc.capacity == pytest.approx(0)


def test_upper_bound_clipping() -> None:
    """Verify that capacity is clipped to 100 when WBGT is low and intensity is light."""
    wc = work_capacity_dunne(20, "light")
    assert wc.capacity == pytest.approx(100)


def test_invalid_intensity_raises() -> None:
    """Verify that providing an invalid intensity string raises a ValueError."""
    with pytest.raises(ValueError):
        work_capacity_dunne(25, "invalid")


def test_work_intensity_accepts_enum() -> None:
    """A WorkIntensity member gives the same result as its string value (#470)."""
    assert (
        work_capacity_dunne(wbgt=30, work_intensity=WorkIntensity.MODERATE).capacity
        == work_capacity_dunne(wbgt=30, work_intensity="moderate").capacity
    )
