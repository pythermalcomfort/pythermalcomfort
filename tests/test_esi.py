import pytest

from pythermalcomfort.models import esi
from tests.conftest import is_equal


def test_esi() -> None:
    """Test that the function calculates the ESI correctly for given inputs."""
    result = esi(tdb=30.2, rh=42.2, sol_radiation_global=766)
    assert is_equal(result.esi, 26.2, 0.1)


def test_esi_list_input() -> None:
    """Test that the function calculates the ESI correctly for list inputs."""
    result = esi(tdb=[30.2, 27.0], rh=[42.2, 68.8], sol_radiation_global=[766, 289])
    assert is_equal(result.esi, [26.2, 25.6], 0.1)


@pytest.mark.parametrize(
    ("tdb", "rh", "sol"),
    [
        (30, -5, 500),  # negative relative humidity
        (30, 120, 500),  # RH above 100%
        (30, 50, -10),  # negative solar radiation
    ],
)
def test_esi_invalid_numeric_ranges(tdb, rh, sol) -> None:
    """Test that the function raises ValueError for invalid input ranges."""
    with pytest.raises(ValueError):
        esi(tdb=tdb, rh=rh, sol_radiation_global=sol)


@pytest.mark.parametrize(
    ("tdb", "rh", "sol", "expected"),
    [
        # ESI = 0.63 tdb - 0.03 rh + 0.002 sol + 0.0054 tdb rh - 0.073 / (0.1 + sol)
        (30, 0, 500, 19.9),  # minimum RH: 18.9 + 1.0 - 0.0001 = 19.8999
        (30, 100, 500, 33.1),  # maximum RH: 18.9 - 3.0 + 1.0 + 16.2 - 0.0001 = 33.0999
        (30, 50, 0, 24.8),  # minimum solar radiation: 18.9 - 1.5 + 8.1 - 0.73 = 24.77
    ],
)
def test_esi_boundary_conditions(
    tdb: float,
    rh: float,
    sol: float,
    expected: float,
) -> None:
    """Test that the function handles boundary conditions correctly."""
    result = esi(tdb=tdb, rh=rh, sol_radiation_global=sol)
    assert is_equal(result.esi, expected, 0.1)


@pytest.mark.parametrize(
    ("tdb", "rh", "sol"),
    [("30.0", 45, 500), (30, "45", 500), (30, 45, "500.0")],
)
def test_esi_invalid_type(tdb: float, rh: float, sol: float) -> None:
    """Test that the function raises TypeError for invalid input types."""
    with pytest.raises(TypeError):
        esi(tdb=tdb, rh=rh, sol_radiation_global=sol)
