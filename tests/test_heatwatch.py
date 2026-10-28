import numpy as np
import pytest

from pythermalcomfort.classes_return import HeatWatch
from pythermalcomfort.models import two_nodes_gagge
from pythermalcomfort.models.heatwatch import Persona, heatwatch

OLDER_ADULT = Persona(
    age=75, met=1.2, clo=0.45, posture="sitting", illness=True, medication=True
)
PREGNANT_ADULT = Persona(age=30, met=1.5, clo=0.36, pregnancy=True)
OUTDOOR_WORKER = Persona(age=30, met=1.8, clo=0.55)


@pytest.mark.parametrize(
    ("age", "w_max", "alpha"),
    [
        (10, 0.85, 0.6),
        (17, 0.85, 0.6),
        (18, 0.85, 0.7),
        (39, 0.85, 0.7),
        (40, 0.8, 0.55),
        (59, 0.8, 0.55),
        (60, 0.75, 0.5),
        (69, 0.75, 0.5),
        (70, 0.7, 0.4),
        (79, 0.7, 0.4),
        (80, 0.65, 0.3),
        (95, 0.65, 0.3),
    ],
)
def test_persona_age_bands(age, w_max, alpha):
    persona = Persona(age=age, met=1.2, clo=0.5)
    assert persona.w_max == pytest.approx(w_max)
    assert persona.alpha == pytest.approx(alpha)


def test_persona_health_conditions_and_acclimatisation():
    assert OLDER_ADULT.w_max == pytest.approx(0.7 * 0.95**2)
    assert OLDER_ADULT.alpha == pytest.approx(0.4)

    all_conditions = Persona(
        age=30,
        met=1.2,
        clo=0.5,
        obesity=True,
        medication=True,
        pregnancy=True,
        mobility_impairment=True,
        homelessness=True,
        illness=True,
    )
    # homelessness changes alpha but not w_max
    assert all_conditions.w_max == pytest.approx(0.85 * 0.95**5)
    assert all_conditions.alpha == pytest.approx(0.5)

    assert Persona(age=30, met=1.2, clo=0.5, obesity=True).alpha == pytest.approx(0.6)
    # alpha caps never raise the age value
    assert Persona(age=85, met=1.2, clo=0.5, illness=True).alpha == pytest.approx(0.3)

    acclimatised = Persona(age=30, met=1.2, clo=0.5, acclimatised=True)
    assert acclimatised.w_max == pytest.approx(0.85 / 0.9)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"age": 30.5}, TypeError),
        ({"age": -1}, ValueError),
        ({"met": 0}, ValueError),
        ({"clo": -0.1}, ValueError),
        ({"posture": "walking"}, ValueError),
        ({"illness": 1}, TypeError),
    ],
)
def test_persona_validation(kwargs, error):
    values = {"age": 30, "met": 1.2, "clo": 0.5} | kwargs
    with pytest.raises(error):
        Persona(**values)


def test_heatwatch_rejects_other_persona_types():
    with pytest.raises(TypeError):
        heatwatch(tdb=30, tr=30, v=0.2, rh=50, persona={"age": 30})


@pytest.mark.parametrize("persona", [OLDER_ADULT, PREGNANT_ADULT, OUTDOOR_WORKER])
def test_heatwatch_matches_formula(persona):
    tdb = np.array([18, 21, 23.5, 27, 30, 34, 38, 42, 45])
    rh = np.array([50, 80, 20, 60, 5, 50, 30, 80, 10])
    result = heatwatch(
        tdb=tdb, tr=tdb + 2, v=0.3, rh=rh, persona=persona, round_output=False
    )

    gagge = two_nodes_gagge(
        tdb=tdb,
        tr=tdb + 2,
        v=0.3,
        rh=rh,
        met=persona.met,
        clo=persona.clo,
        position=persona.posture,
        max_skin_blood_flow=90,
        w_max=persona.w_max,
        round_output=False,
    )
    d = gagge.disc / 4.7
    b = (gagge.m_bl - 6.3) / (90 - 6.3)
    combined = np.maximum(d, persona.alpha * d + (1 - persona.alpha) * b)
    expected = 6 * np.clip(combined, 0, 1)

    np.testing.assert_allclose(result.hhrs, expected)
    np.testing.assert_allclose(result.disc, gagge.disc)
    np.testing.assert_allclose(result.m_bl, gagge.m_bl)


def test_heatwatch_reference_values():
    tdb = [20, 22.5, 26, 30, 34, 38, 42]
    result = heatwatch(tdb=tdb, tr=tdb, v=0.2, rh=50, persona=OLDER_ADULT)
    np.testing.assert_allclose(result.hhrs, [0.0, 0.1, 0.6, 2.2, 4.2, 6.0, 6.0])
    assert list(result.risk_category) == [
        "Minimal",
        "Minimal",
        "Minimal",
        "Moderate",
        "Severe",
        "Extreme",
        "Extreme",
    ]


def test_blood_flow_only_raises_the_score():
    # In hot, dry air during work, skin blood flow raises the score above the
    # relative skin wettedness term.
    kwargs = {"tdb": 40, "tr": 40, "v": 0.2, "rh": 5, "round_output": False}
    result = heatwatch(persona=OUTDOOR_WORKER, **kwargs)
    assert result.hhrs == pytest.approx(3.19, abs=0.01)
    assert result.hhrs > 6 * result.disc / 4.7

    # In hot, humid air the score equals the relative skin wettedness term.
    humid = heatwatch(tdb=34, tr=34, v=0.2, rh=80, persona=OUTDOOR_WORKER)
    assert humid.hhrs == pytest.approx(6 * humid.disc / 4.7, abs=0.05)


def test_cool_shade_scores_minimal():
    tdb = [10, 15, 18, 20]
    result = heatwatch(tdb=tdb, tr=tdb, v=0.2, rh=95, persona=OLDER_ADULT)
    assert list(result.risk_category) == ["Minimal"] * 4


def test_radiant_heat_raises_the_score_in_cool_air():
    # Full sun at 19 °C air adds about 30 K to the mean radiant temperature. In still
    # air the score rises above Low but stays below Severe, and a light breeze removes
    # most of the radiant load.
    kwargs = {"tdb": 19, "tr": 50, "rh": 50, "persona": OLDER_ADULT}
    still = heatwatch(v=0.2, round_output=False, **kwargs)
    assert 2 < still.hhrs < 4
    breeze = heatwatch(v=1.0, **kwargs)
    assert breeze.risk_category == "Minimal"


def test_score_is_capped_at_six():
    result = heatwatch(tdb=45, tr=60, v=0.2, rh=90, persona=OLDER_ADULT)
    assert result.hhrs == 6.0
    assert result.risk_category == "Extreme"


def test_category_follows_rounded_score():
    from pythermalcomfort._internal.validation import _mapping
    from pythermalcomfort.models.heatwatch import _RISK_CATEGORIES

    scores = [0, 1.0, 1.1, 2.0, 2.1, 3.0, 4.0, 5.0, 5.1, 6.0]
    expected = [
        "Minimal",
        "Minimal",
        "Low",
        "Low",
        "Moderate",
        "Moderate",
        "High",
        "Severe",
        "Extreme",
        "Extreme",
    ]
    assert list(_mapping(scores, _RISK_CATEGORIES)) == expected


def test_scalar_and_array_inputs_agree():
    scalar = heatwatch(tdb=33, tr=35, v=0.4, rh=45, persona=PREGNANT_ADULT)
    array = heatwatch(
        tdb=[33, 33], tr=[35, 35], v=0.4, rh=[45, 45], persona=PREGNANT_ADULT
    )

    assert isinstance(scalar, HeatWatch)
    assert isinstance(scalar.risk_category, str)
    for name in ("hhrs", "disc", "m_bl"):
        assert np.ndim(scalar[name]) == 0
        assert not isinstance(scalar[name], np.ndarray)
    np.testing.assert_allclose(array.hhrs, [scalar.hhrs, scalar.hhrs])
    assert list(array.risk_category) == [scalar.risk_category] * 2


def test_posture_changes_the_score():
    standing = Persona(age=75, met=1.2, clo=0.45, posture="standing")
    sitting = Persona(age=75, met=1.2, clo=0.45, posture="sitting")
    kwargs = {"tdb": 34, "tr": 34, "v": 0.2, "rh": 50, "round_output": False}
    assert heatwatch(persona=standing, **kwargs).hhrs != pytest.approx(
        heatwatch(persona=sitting, **kwargs).hhrs
    )
