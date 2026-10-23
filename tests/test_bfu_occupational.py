import numpy as np
import pytest

from pythermalcomfort.models import bfu_occupational


def test_foster_fans_scalar():
    result = bfu_occupational(
        tdb=30.0,
        rh=60.0,
        posture="standing",
        activity="walk",
        speed=1.5,
        round_output=False,
        met=125 / 58.15,
        clo=0.28,
    )

    assert result.delta_storage < 0  # fan reduces heat storage
    assert result.heat_storage_no_fan > result.heat_storage_fan

    if result.delta_storage < -50:
        expected_interpretation = 1
    elif result.delta_storage <= 50:
        expected_interpretation = 0
    else:
        expected_interpretation = -1
    assert result.fan_benefit == expected_interpretation


def test_foster_fans_array():
    tdb = np.array([30.0, 32.0])
    rh = np.array([60.0, 55.0])
    speed = np.array([0.5, 0.0])

    result = bfu_occupational(
        tdb=tdb,
        rh=rh,
        posture=["standing", "seated"],
        activity=["walk", "rest"],
        speed=speed,
        round_output=False,
        met=125 / 58.15,
        clo=0.28,
    )

    for attribute in (
        result.heat_storage_fan,
        result.heat_storage_no_fan,
        result.delta_storage,
        result.dry_heat_fan,
        result.dry_heat_no_fan,
        result.evaporative_heat_fan,
        result.evaporative_heat_no_fan,
        result.respiratory_heat,
        result.sweat_efficiency_fan,
        result.sweat_efficiency_no_fan,
        result.fan_benefit,
    ):
        assert isinstance(attribute, np.ndarray)
        assert attribute.shape == tdb.shape

    expected_interpretation = np.select(
        [result.delta_storage < -50, result.delta_storage <= 50],
        [1, 0],
        default=-1,
    )
    np.testing.assert_array_equal(result.fan_benefit, expected_interpretation)


def test_foster_fans_invalid_airflow():
    with pytest.raises(ValueError):
        bfu_occupational(tdb=30.0, rh=60.0, v=-0.3, met=125 / 58.15, clo=0.28)


@pytest.mark.parametrize(
    ("posture", "canonical"),
    [
        ("seated", "seated"),
        ("SEATED", "seated"),
        ("sitting", "seated"),
        ("stand", "standing"),
    ],
)
def test_posture_aliases(posture, canonical):
    result = bfu_occupational(
        30, 60, posture=posture, round_output=False, met=125 / 58.15, clo=0.28
    )
    expected = bfu_occupational(
        30, 60, posture=canonical, round_output=False, met=125 / 58.15, clo=0.28
    )
    assert result.heat_storage_fan == pytest.approx(expected.heat_storage_fan)
    assert result.heat_storage_no_fan == pytest.approx(expected.heat_storage_no_fan)


def test_seated_and_standing_use_different_coefficients():
    seated = bfu_occupational(
        30, 60, posture="seated", round_output=False, met=125 / 58.15, clo=0.28
    )
    standing = bfu_occupational(
        30, 60, posture="standing", round_output=False, met=125 / 58.15, clo=0.28
    )
    assert seated.heat_storage_fan != pytest.approx(standing.heat_storage_fan)


@pytest.mark.parametrize(
    ("activity", "canonical"),
    [
        ("walk", "walk"),
        ("WALK", "walk"),
        ("WaLk", "walk"),
        ("rest", "rest"),
        ("REST", "rest"),
        ("cycle", "cycle"),
        ("CYCLE", "cycle"),
        ("CyClE", "cycle"),
    ],
)
def test_activity_case_normalization(activity, canonical):
    result = bfu_occupational(
        30, 60, activity=activity, round_output=False, met=125 / 58.15, clo=0.28
    )
    expected = bfu_occupational(
        30, 60, activity=canonical, round_output=False, met=125 / 58.15, clo=0.28
    )
    assert result.heat_storage_fan == pytest.approx(expected.heat_storage_fan)


def test_two_dimensional_posture_and_activity_inputs():
    postures = [["seated", "standing"], ["sitting", "stand"]]
    activities = [["walk", "rest"], ["CYCLE", "WALK"]]
    result = bfu_occupational(
        [[30], [35]],
        [50, 60],
        posture=postures,
        activity=activities,
        round_output=False,
        met=125 / 58.15,
        clo=0.28,
    )
    assert result.heat_storage_fan.shape == (2, 2)
    for i in range(2):
        for j in range(2):
            expected = bfu_occupational(
                [30, 35][i],
                [50, 60][j],
                posture=postures[i][j],
                activity=activities[i][j],
                round_output=False,
                met=125 / 58.15,
                clo=0.28,
            )
            assert result.heat_storage_fan[i, j] == pytest.approx(
                expected.heat_storage_fan
            )
            assert result.heat_storage_no_fan[i, j] == pytest.approx(
                expected.heat_storage_no_fan
            )


@pytest.mark.parametrize("rh", [-1, 101])
def test_invalid_humidity(rh):
    with pytest.raises(ValueError, match="rh"):
        bfu_occupational(30, rh, met=125 / 58.15, clo=0.28)


def test_invalid_posture():
    with pytest.raises(ValueError, match="posture"):
        bfu_occupational(30, 60, posture="unknown", met=125 / 58.15, clo=0.28)


def test_equal_air_speeds_have_no_fan_effect():
    result = bfu_occupational(30, 60, v=0.2, v_no_fan=0.2, met=125 / 58.15, clo=0.28)
    assert result.delta_storage == pytest.approx(0)
    assert result.fan_benefit == 0


@pytest.mark.parametrize("missing", ["met", "clo"])
def test_clothing_and_metabolic_rate_are_required(missing):
    inputs = dict(tdb=30, rh=60, met=2, clo=0.28)
    del inputs[missing]
    with pytest.raises(TypeError, match=missing):
        bfu_occupational(**inputs)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("met", None),
        ("met", -1),
        ("clo", -0.1),
        ("v", -1),
        ("v_no_fan", -1),
        ("speed", -1),
        ("max_sweating", -1),
        ("body_surface_area", 0),
        ("tdb", np.nan),
        ("tr", np.inf),
        ("rh", np.nan),
        ("round_output", "yes"),
    ],
)
def test_invalid_inputs(name, value):
    inputs = dict(tdb=30, rh=60, met=2, clo=0.28)
    inputs[name] = value
    with pytest.raises((ValueError, TypeError)):
        bfu_occupational(**inputs)


def test_posture_enum_and_categorical_only_broadcasting():
    from pythermalcomfort.utilities import Postures

    result = bfu_occupational(
        tdb=30,
        rh=60,
        met=2,
        clo=0.28,
        posture=[Postures.sitting, Postures.standing],
        round_output=False,
    )
    assert result.delta_storage.shape == (2,)
    expected = bfu_occupational(
        tdb=30,
        rh=60,
        met=2,
        clo=0.28,
        posture=Postures.sitting,
        round_output=False,
    )
    assert result.delta_storage[0] == pytest.approx(expected.delta_storage)
    assert result.delta_storage[0] != pytest.approx(result.delta_storage[1])


def test_sweat_only_broadcasting_and_heat_balance():
    from pythermalcomfort.utilities import met_to_w_m2

    result = bfu_occupational(
        tdb=35,
        rh=50,
        met=2,
        clo=0.28,
        max_sweating=[0, 500],
        round_output=False,
    )
    assert result.delta_storage.shape == (2,)
    assert result.evaporative_heat_fan[0] == 0
    np.testing.assert_allclose(
        result.heat_storage_fan
        + result.dry_heat_fan
        + result.evaporative_heat_fan
        + result.respiratory_heat,
        2 * met_to_w_m2,
    )
    np.testing.assert_allclose(
        result.delta_storage,
        result.heat_storage_fan - result.heat_storage_no_fan,
    )


def test_empirical_sweating_and_zero_evaporative_capacity():
    with np.errstate(divide="raise", invalid="raise"):
        result = bfu_occupational(
            tdb=[30, 45],
            rh=[50, 100],
            met=2,
            clo=0.28,
            max_sweating=None,
            round_output=False,
        )
    assert np.isfinite(result.delta_storage).all()
    assert np.all(
        (result.sweat_efficiency_fan >= 0) & (result.sweat_efficiency_fan <= 1)
    )


def test_incompatible_shapes():
    with pytest.raises(ValueError, match="broadcast"):
        bfu_occupational(
            tdb=[30, 35], rh=60, met=2, clo=0.28, max_sweating=[0, 100, 500]
        )


def test_legacy_numeric_regression_and_benefit_categories():
    # Baseline: PR commit 88d831c, 125 W/m², 1 L/h, BSA 1.81 m².
    # Reusing p_sat replaces the source approximation; allow 0.5 W/m² for it.
    result = bfu_occupational(
        tdb=[30, 36, 40, 45],
        rh=[50, 60, 80, 20],
        met=125 / 58.15,
        clo=0.28,
        round_output=False,
    )
    np.testing.assert_allclose(
        result.heat_storage_fan,
        [-334.8153, -160.9678, 200.5802, 21.0010],
        atol=0.5,
        rtol=0,
    )
    np.testing.assert_array_equal(result.fan_benefit, [1, 1, 0, -1])


def test_cycling_cadence_is_not_used_as_walking_speed():
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = bfu_occupational(
            tdb=35,
            rh=50,
            met=2,
            clo=0.28,
            activity="cycle",
            speed=60,
            posture="sitting",
            round_output=False,
        )
    assert np.isfinite(result.heat_storage_fan)
    assert np.isfinite(result.heat_storage_no_fan)


def test_nude_insulation_matches_limit_of_light_clothing():
    # Supplementary Eq. 13 and the interpolation in Eq. 14 meet at zero clo.
    result = bfu_occupational(
        tdb=35,
        rh=50,
        met=2,
        clo=[0, 1e-8],
        round_output=False,
    )
    assert result.dry_heat_fan[0] == pytest.approx(result.dry_heat_fan[1], abs=1e-4)
    assert result.dry_heat_no_fan[0] == pytest.approx(
        result.dry_heat_no_fan[1], abs=1e-4
    )


def test_no_evaporation_above_skin_vapour_pressure():
    # Eq. 1 uses the smaller of sweat supply and the evaporative capacity.
    result = bfu_occupational(tdb=45, rh=100, met=2, clo=0.28)
    assert result.evaporative_heat_fan == 0
    assert result.evaporative_heat_no_fan == 0


def test_metabolic_rate_changes_absolute_storage_not_fan_difference():
    low = bfu_occupational(tdb=35, rh=50, met=1, clo=0.28, round_output=False)
    high = bfu_occupational(tdb=35, rh=50, met=3, clo=0.28, round_output=False)
    assert high.heat_storage_fan != pytest.approx(low.heat_storage_fan)
    assert high.delta_storage == pytest.approx(low.delta_storage)


@pytest.mark.parametrize("activity", ["wak", "other", "cycling", "", None, 123])
def test_unknown_activity_is_rejected(activity):
    with pytest.raises(ValueError, match="activity.*walk.*rest.*cycle"):
        bfu_occupational(tdb=30, rh=60, met=2, clo=0.28, activity=activity)


@pytest.mark.parametrize(
    "activity", [["walk", "wak"], [["WALK", "rest"], ["cycle", "other"]]]
)
def test_unknown_activity_in_arrays_is_rejected(activity):
    with pytest.raises(ValueError, match="activity.*walk.*rest.*cycle"):
        bfu_occupational(tdb=30, rh=60, met=2, clo=0.28, activity=activity)


def test_input_dataclass_rejects_unknown_activity():
    from pythermalcomfort.classes_input import BFUOccupationalInputs

    with pytest.raises(ValueError, match="activity.*walk.*rest.*cycle"):
        BFUOccupationalInputs(tdb=30, rh=60, met=2, clo=0.28, v=3.5, activity="wak")
