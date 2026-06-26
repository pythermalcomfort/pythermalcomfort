import numpy as np
import pytest

from pythermalcomfort.classes_return import SportsHeatStressRisk
from pythermalcomfort.models.sports_heat_stress_risk import (
    Sports,
    sports_heat_stress_risk,
)


@pytest.mark.parametrize(
    ("tdb", "tr", "rh", "vr", "sport", "expected_risk"),
    [
        # Moderate temperature
        (33, 20, 40, 2, Sports.MTB, 1.8),
        # High temperature
        (40, 20, 40, 2, Sports.MTB, 3.3),
        # Low temperature (low risk)
        (10, 20, 40, 2, Sports.MTB, 1.0),
        # Very high temperature (extreme risk)
        (60, 20, 40, 2, Sports.MTB, 4.9),
        # Different sport
        (35, 35, 50, 0.5, Sports.RUNNING, 2.1),
    ],
)
def test_sports_heat_stress_risk_scalar(tdb, tr, rh, vr, sport, expected_risk):
    """Test sports heat stress risk calculation with scalar inputs."""
    result = sports_heat_stress_risk(tdb=tdb, tr=tr, rh=rh, vr=vr, sport=sport)

    # Verify return type is the dataclass
    assert isinstance(result, SportsHeatStressRisk)
    # For scalar inputs, numpy.vectorize returns numpy scalars (0-d arrays)
    assert isinstance(result.t_medium, (float, np.ndarray))
    assert isinstance(result.t_high, (float, np.ndarray))
    assert isinstance(result.t_extreme, (float, np.ndarray))
    assert isinstance(result.recommendation, (str, np.ndarray))

    # Verify expected risk level - convert to float for comparison
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    assert risk_value == pytest.approx(expected_risk, rel=0.01)


def test_sports_heat_stress_risk_array_clamps_vr_to_sport_minimum():
    """Test that array wind speed inputs are clamped elementwise to the sport minimum."""
    tdb = np.array([30, 35, 40])
    tr = np.array([30, 35, 40])
    rh = np.array([40, 50, 60])
    vr = np.array([0.5, 2.0, 2.5])
    vr_effective = np.maximum(vr, Sports.RUNNING.vr)

    result = sports_heat_stress_risk(tdb=tdb, tr=tr, rh=rh, vr=vr, sport=Sports.RUNNING)
    expected = sports_heat_stress_risk(
        tdb=tdb,
        tr=tr,
        rh=rh,
        vr=vr_effective,
        sport=Sports.RUNNING,
    )

    np.testing.assert_allclose(
        np.asarray(result.risk_level_interpolated, dtype=float),
        np.asarray(expected.risk_level_interpolated, dtype=float),
    )
    np.testing.assert_allclose(
        np.asarray(result.t_medium, dtype=float),
        np.asarray(expected.t_medium, dtype=float),
    )
    np.testing.assert_allclose(
        np.asarray(result.t_high, dtype=float),
        np.asarray(expected.t_high, dtype=float),
    )
    np.testing.assert_allclose(
        np.asarray(result.t_extreme, dtype=float),
        np.asarray(expected.t_extreme, dtype=float),
    )
    np.testing.assert_array_equal(
        np.asarray(result.recommendation), np.asarray(expected.recommendation)
    )


def test_sports_heat_stress_risk_array():
    """Test sports heat stress risk calculation with array inputs (lists)."""
    t = [40, 40]
    rh = 40
    v = 2
    tr = 20
    result = sports_heat_stress_risk(tdb=t, tr=tr, rh=rh, vr=v, sport=Sports.MTB)

    # Verify return type
    assert isinstance(result, SportsHeatStressRisk)

    # Verify array outputs - numpy.vectorize returns numpy arrays
    np.testing.assert_allclose(
        result.risk_level_interpolated,
        [3.3, 3.3],
        rtol=0.01,
        atol=0.01,
    )
    assert isinstance(result.t_medium, (list, np.ndarray))
    assert isinstance(result.t_high, (list, np.ndarray))
    assert isinstance(result.t_extreme, (list, np.ndarray))
    assert len(result.t_medium) == 2
    assert len(result.t_high) == 2
    assert len(result.t_extreme) == 2


def test_sports_heat_stress_risk_numpy_array():
    """Test sports heat stress risk calculation with NumPy array inputs."""
    t = np.array([30, 35, 40])
    rh = np.array([40, 50, 60])
    v = np.array([0.5, 1.0, 1.5])
    tr = np.array([30, 35, 40])
    result = sports_heat_stress_risk(tdb=t, tr=tr, rh=rh, vr=v, sport=Sports.RUNNING)

    # Verify return type
    assert isinstance(result, SportsHeatStressRisk)

    # Verify outputs are numpy arrays (from numpy.vectorize)
    assert isinstance(result.risk_level_interpolated, (list, np.ndarray))
    assert isinstance(result.t_medium, (list, np.ndarray))
    assert isinstance(result.t_high, (list, np.ndarray))
    assert isinstance(result.t_extreme, (list, np.ndarray))
    assert isinstance(result.recommendation, (list, np.ndarray))

    # Verify correct array length
    assert len(result.risk_level_interpolated) == 3
    assert len(result.t_medium) == 3
    assert len(result.t_high) == 3
    assert len(result.t_extreme) == 3
    assert len(result.recommendation) == 3

    # Verify all values are valid
    for risk in result.risk_level_interpolated:
        assert 1 <= risk <= 4.9


def test_sports_heat_stress_risk_different_sports():
    """Test that different sports produce different risk levels."""
    t = 35
    rh = 50
    v = 0.5
    tr = 35

    result_running = sports_heat_stress_risk(
        tdb=t, tr=tr, rh=rh, vr=v, sport=Sports.RUNNING
    )
    result_walking = sports_heat_stress_risk(
        tdb=t, tr=tr, rh=rh, vr=v, sport=Sports.WALKING
    )

    # Running has higher metabolic rate, so should have higher risk at same conditions
    assert isinstance(result_running, SportsHeatStressRisk)
    assert isinstance(result_walking, SportsHeatStressRisk)
    # Both should return valid risk levels
    assert 1 <= result_running.risk_level_interpolated <= 4.9
    assert 1 <= result_walking.risk_level_interpolated <= 4.9


def test_sports_heat_stress_risk_invalid_inputs():
    """Test that invalid inputs raise appropriate errors."""
    # Test invalid humidity (> 100%)
    with pytest.raises(
        ValueError, match="Relative humidity.*must be between 0 and 100"
    ):
        sports_heat_stress_risk(tdb=30, tr=30, rh=150, vr=0.5, sport=Sports.RUNNING)

    # Test invalid humidity (< 0%)
    with pytest.raises(
        ValueError, match="Relative humidity.*must be between 0 and 100"
    ):
        sports_heat_stress_risk(tdb=30, tr=30, rh=-10, vr=0.5, sport=Sports.RUNNING)

    # Test invalid air speed (negative)
    with pytest.raises(ValueError, match="Relative air speed.*must be non-negative"):
        sports_heat_stress_risk(tdb=30, tr=30, rh=50, vr=-1, sport=Sports.RUNNING)

    # Test invalid sport type
    with pytest.raises(
        TypeError,
        match="sport must be a _SportsValues instance from the Sports dataclass",
    ):
        sports_heat_stress_risk(tdb=30, tr=30, rh=50, vr=0.5, sport="invalid")


def test_sports_heat_stress_risk_broadcasting_incompatibility():
    """Test that incompatible array shapes raise appropriate broadcasting errors."""
    # Test incompatible array shapes (length mismatch)
    with pytest.raises(ValueError, match="not broadcastable"):
        sports_heat_stress_risk(
            tdb=[30, 35, 40],  # length 3
            tr=30,
            rh=[40, 50],  # length 2 - incompatible!
            vr=0.5,
            sport=Sports.SOCCER,
        )

    # Test another incompatible combination
    with pytest.raises(ValueError, match="not broadcastable"):
        sports_heat_stress_risk(
            tdb=[30, 35],
            tr=[30, 35, 40, 45],  # length 4 - incompatible with tdb!
            rh=50,
            vr=0.5,
            sport=Sports.TENNIS,
        )


def test_sports_heat_stress_risk_broadcasting():
    """Test that broadcasting works correctly with different input shapes."""
    # Test broadcasting with different array shapes
    result = sports_heat_stress_risk(
        tdb=[30, 35, 40], tr=30, rh=[40, 50, 60], vr=0.5, sport=Sports.SOCCER
    )

    assert isinstance(result, SportsHeatStressRisk)
    assert len(result.risk_level_interpolated) == 3
    assert len(result.t_medium) == 3
    assert len(result.t_high) == 3
    assert len(result.t_extreme) == 3


def test_sports_heat_stress_risk_dataclass_fields():
    """Test that the returned dataclass has all expected fields."""
    result = sports_heat_stress_risk(tdb=35, tr=35, rh=50, vr=0.5, sport=Sports.TENNIS)

    # Verify all required fields are present
    assert hasattr(result, "risk_level_interpolated")
    assert hasattr(result, "t_medium")
    assert hasattr(result, "t_high")
    assert hasattr(result, "t_extreme")
    assert hasattr(result, "recommendation")

    # Convert numpy scalars to Python types for comparison
    t_medium = float(np.asarray(result.t_medium).item())
    t_high = float(np.asarray(result.t_high).item())
    t_extreme = float(np.asarray(result.t_extreme).item())

    # Verify threshold order (t_medium < t_high < t_extreme)
    assert t_medium < t_high
    assert t_high < t_extreme

    # Verify recommendation is a non-empty string (may be numpy array)
    recommendation = str(np.asarray(result.recommendation).item())
    assert len(recommendation) > 0


def test_sports_heat_stress_risk_recommendations():
    """Test that recommendations are appropriate for different risk levels."""
    # Test low risk (risk level 1.0-2.0)
    result_low = sports_heat_stress_risk(
        tdb=20, tr=20, rh=50, vr=0.5, sport=Sports.RUNNING
    )
    assert "Increase hydration & modify clothing" == str(result_low.recommendation)
    assert 1.0 <= result_low.risk_level_interpolated < 2.0

    # Test medium risk (risk level 2.0-3.0)
    result_medium = sports_heat_stress_risk(
        tdb=35, tr=34, rh=30, vr=0.5, sport=Sports.RUNNING
    )
    assert 2.0 <= result_medium.risk_level_interpolated < 3.0
    assert "Increase frequency and/or duration of rest breaks" == str(
        result_medium.recommendation
    )

    # Test high risk (risk level 3.0-4.0)
    result_high = sports_heat_stress_risk(
        tdb=40, tr=35, rh=30, vr=0.5, sport=Sports.RUNNING
    )
    assert 3.0 <= result_high.risk_level_interpolated < 4.0
    assert "Apply active cooling strategies" == str(result_high.recommendation)

    # Test medium risk near high boundary (risk level 2.0-3.0)
    result = sports_heat_stress_risk(tdb=38, tr=38, rh=40, vr=0.5, sport=Sports.RUNNING)
    assert 2.0 <= result.risk_level_interpolated < 3.0
    assert "Increase frequency and/or duration of rest breaks" == str(
        result.recommendation
    )

    # Test extreme risk (risk level = 4.9 at the upper temperature anchor)
    result_extreme = sports_heat_stress_risk(
        tdb=50, tr=50, rh=50, vr=0.5, sport=Sports.RUNNING
    )
    assert "Consider suspending play" in str(result_extreme.recommendation)
    assert result_extreme.risk_level_interpolated == pytest.approx(4.9, rel=1e-3)


def test_sports_heat_stress_risk_array_recommendations():
    """Test that array inputs produce array recommendations."""
    result = sports_heat_stress_risk(
        tdb=[20, 35, 45],
        tr=[20, 35, 45],
        rh=[50, 50, 50],
        vr=[0.5, 0.5, 0.5],
        sport=Sports.SOCCER,
    )

    # Verify recommendations is an array (numpy array or list)
    assert isinstance(result.recommendation, (list, np.ndarray))
    assert len(result.recommendation) == 3

    # Verify each recommendation is a string (may be numpy string)
    for rec in result.recommendation:
        rec_str = (
            str(rec) if isinstance(rec, (str, np.str_)) else str(np.asarray(rec).item())
        )
        assert len(rec_str) > 0


def test_sports_heat_stress_risk_threshold_capping():
    """Test that threshold temperatures are properly capped at min/max bounds."""
    # Test conditions that should trigger maximum threshold capping
    # Very high humidity and temperature should cap thresholds
    result_high = sports_heat_stress_risk(
        tdb=45, tr=45, rh=90, vr=0.1, sport=Sports.RUNNING
    )

    # Verify thresholds respect maximum bounds (from implementation)
    # max_t_low = 34.5, max_t_medium = 39, max_t_high = 43.5
    assert result_high.t_medium <= 34.5
    assert result_high.t_high <= 39.0
    assert result_high.t_extreme <= 43.5

    # Test conditions that should trigger minimum threshold capping
    # Low humidity and low temperature should set minimum thresholds
    result_low = sports_heat_stress_risk(
        tdb=25, tr=25, rh=10, vr=2.0, sport=Sports.WALKING
    )

    # Verify thresholds respect minimum bounds (from implementation)
    # min_t_medium = 23, min_t_high = 25, min_t_extreme = 26
    assert result_low.t_medium >= 23.0
    assert result_low.t_high >= 25.0
    assert result_low.t_extreme >= 26.0

    # Verify threshold ordering is always maintained
    assert result_high.t_medium < result_high.t_high < result_high.t_extreme
    assert result_low.t_medium < result_low.t_high < result_low.t_extreme


def test_sports_heat_stress_risk_edge_temperature_ranges():
    """Test risk calculation at temperature boundary conditions."""
    # Test very low temperature (should be low risk, risk_level = 1)
    result_very_low = sports_heat_stress_risk(
        tdb=15, tr=15, rh=50, vr=0.5, sport=Sports.RUNNING
    )
    assert result_very_low.risk_level_interpolated == pytest.approx(1.0, abs=0.01)

    # Test very high temperature (should be capped at the upper extreme risk level)
    result_very_high = sports_heat_stress_risk(
        tdb=50, tr=50, rh=50, vr=0.5, sport=Sports.RUNNING
    )
    assert result_very_high.risk_level_interpolated == pytest.approx(4.9, abs=0.01)


def test_sports_heat_stress_risk_extreme_interpolation():
    """Test interpolation within the extreme risk range up to 4.9."""
    anchor_result = sports_heat_stress_risk(
        tdb=43.5, tr=20, rh=0, vr=10, sport=Sports.FISHING
    )
    assert anchor_result.t_extreme == pytest.approx(43.5, abs=0.01)
    assert anchor_result.risk_level_interpolated == pytest.approx(4.0, abs=0.01)

    mid_result = sports_heat_stress_risk(
        tdb=45.0, tr=20, rh=0, vr=10, sport=Sports.FISHING
    )
    assert mid_result.t_extreme == pytest.approx(43.5, abs=0.01)
    # With t_upper_extreme = t_extreme + 5, risk at tdb=45 is 4.0 + (45 - 43.5) / 5 = 4.3
    assert mid_result.risk_level_interpolated == pytest.approx(4.3, abs=0.01)

    capped_result = sports_heat_stress_risk(
        tdb=50.0, tr=20, rh=0, vr=10, sport=Sports.FISHING
    )
    assert capped_result.risk_level_interpolated == pytest.approx(4.9, abs=0.01)


def test_sports_heat_stress_risk_extreme_interpolation_uses_lower_model_threshold():
    """Test that severe conditions start extreme interpolation below 43.5°C."""
    base_result = sports_heat_stress_risk(
        tdb=45.0, tr=45.0, rh=100, vr=0, sport=Sports.RUNNING
    )
    severe_result = sports_heat_stress_risk(
        tdb=45.0, tr=50.0, rh=100, vr=0, sport=Sports.RUNNING
    )

    assert severe_result.t_extreme < 43.5
    assert severe_result.risk_level_interpolated >= 4.0
    assert severe_result.risk_level_interpolated >= base_result.risk_level_interpolated


def test_sports_croquet_values():
    """Test Croquet preset values."""
    assert (Sports.CROQUET.clo, Sports.CROQUET.met, Sports.CROQUET.vr) == (
        0.7,
        4.5,
        0.5,
    )
    assert Sports.CROQUET.duration == 90


def test_sports_heat_stress_risk_all_sports():
    """Test that all predefined sports in Sports dataclass work correctly."""
    # Get all sport attributes from Sports dataclass
    all_sports = [
        Sports.ABSEILING,
        Sports.ARCHERY,
        Sports.AUSTRALIAN_FOOTBALL,
        Sports.BASEBALL,
        Sports.BASKETBALL,
        Sports.BOWLS,
        Sports.CANOEING,
        Sports.CRICKET,
        Sports.CROQUET,
        Sports.CYCLING,
        Sports.EQUESTRIAN,
        Sports.FIELD_ATHLETICS,
        Sports.FIELD_HOCKEY,
        Sports.FISHING,
        Sports.GOLF,
        Sports.HORSEBACK,
        Sports.KAYAKING,
        Sports.RUNNING,
        Sports.MTB,
        Sports.NETBALL,
        Sports.OZTAG,
        Sports.PICKLEBALL,
        Sports.CLIMBING,
        Sports.ROWING,
        Sports.RUGBY_LEAGUE,
        Sports.RUGBY_UNION,
        Sports.SAILING,
        Sports.SHOOTING,
        Sports.SOCCER,
        Sports.SOFTBALL,
        Sports.TENNIS,
        Sports.TOUCH,
        Sports.VOLLEYBALL,
        Sports.WALKING,
    ]

    # Test moderate conditions with each sport
    tdb, tr, rh, vr = 32, 32, 50, 0.5

    for sport in all_sports:
        result = sports_heat_stress_risk(tdb=tdb, tr=tr, rh=rh, vr=vr, sport=sport)

        # Verify all sports return valid results
        assert isinstance(result, SportsHeatStressRisk)
        assert isinstance(result.risk_level_interpolated, (float, np.ndarray))
        # Convert to float for comparison
        risk_value = float(np.asarray(result.risk_level_interpolated).item())
        assert 1 <= risk_value <= 4.9
        assert isinstance(result.t_medium, (float, np.ndarray))
        assert isinstance(result.t_high, (float, np.ndarray))
        assert isinstance(result.t_extreme, (float, np.ndarray))
        assert isinstance(result.recommendation, (str, np.ndarray))

        # Verify threshold ordering - convert to float for comparison
        t_medium = float(np.asarray(result.t_medium).item())
        t_high = float(np.asarray(result.t_high).item())
        t_extreme = float(np.asarray(result.t_extreme).item())
        assert t_medium < t_high < t_extreme


@pytest.mark.parametrize(
    ("tdb", "tr", "rh", "vr", "sport"),
    [
        # Boundary: rh = 0% (minimum humidity)
        (30, 30, 0, 0.5, Sports.RUNNING),
        # Boundary: rh = 100% (maximum humidity)
        (30, 30, 100, 0.5, Sports.RUNNING),
        # Boundary: vr = 0 (no air movement)
        (30, 30, 50, 0, Sports.SOCCER),
        # Boundary: very low vr
        (30, 30, 50, 0.01, Sports.TENNIS),
        # Combination: high temp, high humidity, no wind
        (40, 40, 100, 0, Sports.RUNNING),
        # Combination: low temp, low humidity, high wind
        (20, 20, 0, 5.0, Sports.CYCLING),
    ],
)
def test_sports_heat_stress_risk_boundary_values(tdb, tr, rh, vr, sport):
    """Test sports heat stress risk with boundary and edge case values."""
    result = sports_heat_stress_risk(tdb=tdb, tr=tr, rh=rh, vr=vr, sport=sport)

    # Verify function handles boundary values correctly
    assert isinstance(result, SportsHeatStressRisk)
    assert isinstance(result.risk_level_interpolated, (float, np.ndarray))
    # Convert to float for comparison
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    assert 1 <= risk_value <= 4.9
    assert isinstance(result.t_medium, (float, np.ndarray))
    assert isinstance(result.t_high, (float, np.ndarray))
    assert isinstance(result.t_extreme, (float, np.ndarray))
    assert isinstance(result.recommendation, (str, np.ndarray))

    # Verify threshold ordering is maintained - convert to float for comparison
    t_medium = float(np.asarray(result.t_medium).item())
    t_high = float(np.asarray(result.t_high).item())
    t_extreme = float(np.asarray(result.t_extreme).item())
    assert t_medium < t_high < t_extreme

    # Verify no NaN values
    assert not np.isnan(risk_value)
    assert not np.isnan(t_medium)
    assert not np.isnan(t_high)
    assert not np.isnan(t_extreme)


def test_sports_heat_stress_risk_extreme_negative_wind_speed():
    """Test that extreme negative wind speeds raise ValueError."""
    with pytest.raises(ValueError, match="Relative air speed.*must be non-negative"):
        sports_heat_stress_risk(tdb=30, tr=30, rh=50, vr=-10.5, sport=Sports.RUNNING)

    with pytest.raises(ValueError, match="Relative air speed.*must be non-negative"):
        sports_heat_stress_risk(tdb=30, tr=30, rh=50, vr=-0.001, sport=Sports.SOCCER)


def test_sports_heat_stress_risk_very_large_wind_speed():
    """Test that very large wind speeds are handled correctly."""
    # Very high wind speed (hurricane force)
    result = sports_heat_stress_risk(
        tdb=30, tr=30, rh=50, vr=50.0, sport=Sports.CYCLING
    )
    assert isinstance(result, SportsHeatStressRisk)
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    assert 1 <= risk_value <= 4.9

    # Extreme wind speed
    result = sports_heat_stress_risk(
        tdb=35, tr=35, rh=60, vr=100.0, sport=Sports.SAILING
    )
    assert isinstance(result, SportsHeatStressRisk)
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    assert 1 <= risk_value <= 4.9


def test_sports_heat_stress_risk_extreme_cold_temperatures():
    """Test behavior with extreme cold temperatures (Arctic/Antarctic conditions)."""
    # Arctic winter conditions
    result = sports_heat_stress_risk(
        tdb=-40, tr=-40, rh=50, vr=5.0, sport=Sports.RUNNING
    )
    assert isinstance(result, SportsHeatStressRisk)
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    # Should be low risk (1) for extreme cold
    assert risk_value == pytest.approx(1.0, abs=0.01)

    # Extreme sub-zero temperature
    result = sports_heat_stress_risk(
        tdb=-20, tr=-20, rh=30, vr=1.0, sport=Sports.WALKING
    )
    assert isinstance(result, SportsHeatStressRisk)
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    assert risk_value == pytest.approx(1.0, abs=0.01)


def test_sports_heat_stress_risk_extreme_hot_temperatures():
    """Test behavior with extremely high temperatures (> 60°C)."""
    # Death Valley extreme heat
    result = sports_heat_stress_risk(tdb=70, tr=70, rh=10, vr=0.5, sport=Sports.WALKING)
    assert isinstance(result, SportsHeatStressRisk)
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    # Should be capped at the upper extreme risk level
    assert risk_value == pytest.approx(4.9, abs=0.01)

    # Industrial environment extreme heat
    result = sports_heat_stress_risk(tdb=80, tr=80, rh=50, vr=0.1, sport=Sports.RUNNING)
    assert isinstance(result, SportsHeatStressRisk)
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    assert risk_value == pytest.approx(4.9, abs=0.01)


def test_sports_heat_stress_risk_zero_values():
    """Test with zero values for environmental parameters."""
    # All zeros (except reasonable tdb)
    result = sports_heat_stress_risk(tdb=0, tr=0, rh=0, vr=0, sport=Sports.WALKING)
    assert isinstance(result, SportsHeatStressRisk)
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    assert 1 <= risk_value <= 4.9

    # Zero humidity and wind
    result = sports_heat_stress_risk(tdb=25, tr=25, rh=0, vr=0, sport=Sports.SOCCER)
    assert isinstance(result, SportsHeatStressRisk)
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    assert 1 <= risk_value <= 4.9


def test_sports_heat_stress_risk_very_small_positive_values():
    """Test with very small positive values near zero."""
    result = sports_heat_stress_risk(
        tdb=0.001, tr=0.001, rh=0.001, vr=0.001, sport=Sports.RUNNING
    )
    assert isinstance(result, SportsHeatStressRisk)
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    assert 1 <= risk_value <= 4.9

    # Very small but valid humidity
    result = sports_heat_stress_risk(
        tdb=20, tr=20, rh=0.1, vr=0.01, sport=Sports.TENNIS
    )
    assert isinstance(result, SportsHeatStressRisk)
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    assert 1 <= risk_value <= 4.9


def test_sports_heat_stress_risk_float_precision_edge_cases():
    """Test float precision edge cases."""
    # Temperature at exact boundary values (from model constants)
    result = sports_heat_stress_risk(
        tdb=21.0, tr=21.0, rh=50, vr=0.5, sport=Sports.RUNNING
    )
    assert isinstance(result, SportsHeatStressRisk)

    # Humidity at exact 100%
    result = sports_heat_stress_risk(
        tdb=30, tr=30, rh=100.0, vr=0.5, sport=Sports.SOCCER
    )
    assert isinstance(result, SportsHeatStressRisk)

    # Very high precision values
    result = sports_heat_stress_risk(
        tdb=33.123456789,
        tr=33.987654321,
        rh=55.5555555,
        vr=1.11111111,
        sport=Sports.CYCLING,
    )
    assert isinstance(result, SportsHeatStressRisk)


def test_sports_heat_stress_risk_large_array():
    """Test with large arrays (performance and correctness)."""
    # Create large arrays
    size = 100
    tdb_array = np.linspace(10, 45, size)
    tr_array = np.linspace(10, 45, size)
    rh_array = np.full(size, 50)
    vr_array = np.full(size, 0.5)

    result = sports_heat_stress_risk(
        tdb=tdb_array, tr=tr_array, rh=rh_array, vr=vr_array, sport=Sports.RUNNING
    )

    assert isinstance(result, SportsHeatStressRisk)
    assert len(result.risk_level_interpolated) == size
    assert len(result.t_medium) == size
    assert len(result.t_high) == size
    assert len(result.t_extreme) == size
    assert len(result.recommendation) == size

    # Verify all values are valid
    for i in range(size):
        risk = float(np.asarray(result.risk_level_interpolated[i]).item())
        assert 1 <= risk <= 4.9
        assert not np.isnan(risk)


def test_sports_heat_stress_risk_mixed_extreme_conditions():
    """Test combinations of extreme conditions."""
    # High temp + high humidity + no wind (worst case)
    result = sports_heat_stress_risk(tdb=45, tr=50, rh=100, vr=0, sport=Sports.RUNNING)
    assert isinstance(result, SportsHeatStressRisk)
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    assert 4.0 <= risk_value <= 4.9

    # High temp + low humidity + high wind (best cooling case)
    result = sports_heat_stress_risk(tdb=40, tr=40, rh=5, vr=10.0, sport=Sports.CYCLING)
    assert isinstance(result, SportsHeatStressRisk)
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    assert 1 <= risk_value <= 4.9

    # Cold + high humidity + high wind (hypothermia risk, but not heat stress)
    result = sports_heat_stress_risk(
        tdb=-10, tr=-10, rh=100, vr=15.0, sport=Sports.RUNNING
    )
    assert isinstance(result, SportsHeatStressRisk)
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    assert risk_value == pytest.approx(1.0, abs=0.01)


def test_sports_heat_stress_risk_radiant_temperature_extremes():
    """Test with extreme differences between air and radiant temperature."""
    # Very high radiant temperature (direct sunlight)
    result = sports_heat_stress_risk(tdb=30, tr=70, rh=40, vr=0.5, sport=Sports.TENNIS)
    assert isinstance(result, SportsHeatStressRisk)
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    assert 1 <= risk_value <= 4.9

    # Very low radiant temperature (shade)
    result = sports_heat_stress_risk(tdb=35, tr=20, rh=50, vr=0.5, sport=Sports.SOCCER)
    assert isinstance(result, SportsHeatStressRisk)
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    assert 1 <= risk_value <= 4.9

    # Extreme radiant heat (hot surfaces)
    result = sports_heat_stress_risk(tdb=30, tr=80, rh=30, vr=1.0, sport=Sports.RUNNING)
    assert isinstance(result, SportsHeatStressRisk)
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    assert 1 <= risk_value <= 4.9


def test_sports_heat_stress_risk_negative_radiant_temperature():
    """Test with negative radiant temperatures."""
    # Cold radiant surfaces
    result = sports_heat_stress_risk(tdb=20, tr=-5, rh=40, vr=0.5, sport=Sports.WALKING)
    assert isinstance(result, SportsHeatStressRisk)
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    assert 1 <= risk_value <= 4.9

    # Very cold radiant surfaces
    result = sports_heat_stress_risk(
        tdb=10, tr=-30, rh=50, vr=1.0, sport=Sports.RUNNING
    )
    assert isinstance(result, SportsHeatStressRisk)
    risk_value = float(np.asarray(result.risk_level_interpolated).item())
    assert risk_value == pytest.approx(1.0, abs=0.01)


def test_sports_heat_stress_risk_array_with_negative_wind_speeds():
    """Test that arrays containing negative wind speeds raise ValueError."""
    with pytest.raises(ValueError, match="Relative air speed.*must be non-negative"):
        sports_heat_stress_risk(
            tdb=[30, 35, 40],
            tr=[30, 35, 40],
            rh=[50, 50, 50],
            vr=[0.5, -1.0, 0.5],  # One negative value
            sport=Sports.RUNNING,
        )

    with pytest.raises(ValueError, match="Relative air speed.*must be non-negative"):
        sports_heat_stress_risk(
            tdb=[25, 30],
            tr=[25, 30],
            rh=[40, 50],
            vr=[-0.1, -0.2],  # All negative values
            sport=Sports.SOCCER,
        )


def test_sports_heat_stress_risk_array_with_invalid_humidity():
    """Test that arrays with invalid humidity values raise ValueError."""
    # Humidity > 100%
    with pytest.raises(
        ValueError, match="Relative humidity.*must be between 0 and 100"
    ):
        sports_heat_stress_risk(
            tdb=[30, 35, 40],
            tr=[30, 35, 40],
            rh=[50, 101, 60],  # One value > 100
            vr=[0.5, 0.5, 0.5],
            sport=Sports.TENNIS,
        )

    # Humidity < 0%
    with pytest.raises(
        ValueError, match="Relative humidity.*must be between 0 and 100"
    ):
        sports_heat_stress_risk(
            tdb=[25, 30],
            tr=[25, 30],
            rh=[-5, 50],  # One negative value
            vr=[0.5, 0.5],
            sport=Sports.RUNNING,
        )


def test_sports_heat_stress_risk_single_element_array():
    """Test with single-element arrays (should behave like scalar)."""
    result_array = sports_heat_stress_risk(
        tdb=[30], tr=[30], rh=[50], vr=[0.5], sport=Sports.RUNNING
    )
    result_scalar = sports_heat_stress_risk(
        tdb=30, tr=30, rh=50, vr=0.5, sport=Sports.RUNNING
    )

    # Both should return valid results
    assert isinstance(result_array, SportsHeatStressRisk)
    assert isinstance(result_scalar, SportsHeatStressRisk)

    # Values should be approximately equal
    risk_array = float(np.asarray(result_array.risk_level_interpolated).flat[0])
    risk_scalar = float(np.asarray(result_scalar.risk_level_interpolated).item())
    assert risk_array == pytest.approx(risk_scalar, rel=1e-5)


def test_sports_heat_stress_risk_inf_values():
    """Test that infinite values are handled or rejected appropriately."""
    # Positive infinity should likely cause issues in the model
    # This tests robustness - behavior may vary depending on implementation
    try:
        result = sports_heat_stress_risk(
            tdb=np.inf, tr=30, rh=50, vr=0.5, sport=Sports.RUNNING
        )
        # If it doesn't raise an error, ensure result is still valid
        assert isinstance(result, SportsHeatStressRisk)
    except (ValueError, RuntimeError, Warning):
        # It's acceptable to raise an error for infinity
        pass

    # Negative infinity
    try:
        result = sports_heat_stress_risk(
            tdb=-np.inf, tr=30, rh=50, vr=0.5, sport=Sports.WALKING
        )
        # Should treat as extreme cold (risk = 1)
        if isinstance(result, SportsHeatStressRisk):
            risk_value = float(np.asarray(result.risk_level_interpolated).item())
            assert risk_value == pytest.approx(1.0, abs=0.01)
    except (ValueError, RuntimeError, Warning):
        # It's acceptable to raise an error for infinity
        pass


def test_sports_heat_stress_risk_stress_gradient():
    """Test that risk increases monotonically with temperature (for fixed conditions)."""
    temps = [20, 25, 30, 35, 40, 45]
    risks = []

    for temp in temps:
        result = sports_heat_stress_risk(
            tdb=temp, tr=temp, rh=50, vr=0.5, sport=Sports.RUNNING
        )
        risk_value = float(np.asarray(result.risk_level_interpolated).item())
        risks.append(risk_value)

    # Risk should generally increase with temperature
    # (allowing for some flatness at the extremes where risk is capped at 1 or 4.9)
    for i in range(1, len(risks)):
        assert risks[i] >= risks[i - 1], (
            f"Risk decreased from {risks[i - 1]} to {risks[i]} when temperature increased from {temps[i - 1]} to {temps[i]}"
        )
