import math

import numpy as np
import pytest

from pythermalcomfort.models import phs
from pythermalcomfort.utilities import met_to_w_m2
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_phs(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the Predicted Heat Strain (PHS) correctly for
    various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.PHS.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        inputs["model"] = "7933-2004"
        outputs = entry["outputs"]
        result = phs(**inputs)

        validate_result(result, outputs, tolerance)


weight = 75
height = 1.8
a_dubois = 0.202 * (weight**0.425) * (height**0.725)


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        (
            {
                "tdb": 40,
                "tr": 40,
                "rh": 35,
                "v": 0.3,
                "met": 300 / met_to_w_m2 / a_dubois,
                "clo": 0.5,
                "posture": "standing",
                "wme": 0,
                "model": "7933-2023",
                "duration": 480,
                "limit_inputs": False,
                "acclimatized": 100,
                "round_output": False,
            },
            {
                "sweat_loss_g": 6538,
                "t_cr": 37.6,
                "d_lim_loss_95": 280,
            },
        ),
        (
            {
                "tdb": 35,
                "tr": 35,
                "rh": 60,
                "v": 0.1,
                "met": 300 / met_to_w_m2 / a_dubois,
                "clo": 0.5,
                "posture": "standing",
                "wme": 0,
                "model": "7933-2023",
                "duration": 480,
                "limit_inputs": False,
                "acclimatized": 0,
                "round_output": False,
            },
            {
                "sweat_loss_g": 6345,
                "t_cr": 40.8,
                "d_lim_loss_95": 250,
                "d_lim_t_re": 62,
            },
        ),
        pytest.param(
            {
                "tdb": 30,
                "tr": 54.2,
                "rh": 35,
                "v": 0.1,
                "met": 300 / met_to_w_m2 / a_dubois,
                "clo": 0.8,
                "posture": "standing",
                "wme": 0,
                "a_p": 0.3,
                "f_r": 0.85,
                "model": "7933-2023",
                "duration": 480,
                "limit_inputs": False,
                "acclimatized": 0,
                "round_output": False,
            },
            {
                "sweat_loss_g": 6419,
                "t_cr": 38.7,
                "d_lim_loss_95": 280,
                "d_lim_t_re": 149,
            },
            id="high-radiant-temp",
            marks=pytest.mark.xfail(
                reason="Known discrepancy t_cr and d_lim_t_re",
                strict=True,
            ),
        ),
        (
            {
                "tdb": 30,
                "tr": 30,
                "rh": 45,
                "v": 1.0,
                "met": 450 / met_to_w_m2 / a_dubois,
                "clo": 0.5,
                "posture": "standing",
                "wme": 0,
                "model": "7933-2023",
                "duration": 480,
                "limit_inputs": False,
                "acclimatized": 0,
                "round_output": False,
            },
            {
                "sweat_loss_g": 4593,
                "t_cr": 38.0,
                "d_lim_loss_95": 400,
            },
        ),
        (
            {
                "tdb": 35,
                "tr": 74.6,
                "rh": 30,
                "v": 1.0,
                "met": 250 / met_to_w_m2 / a_dubois,
                "clo": 1,
                "posture": "sitting",
                "wme": 0,
                "a_p": 0.2,
                "f_r": 0.85,
                "model": "7933-2023",
                "duration": 480,
                "limit_inputs": False,
                "acclimatized": 100,
                "round_output": False,
            },
            {
                "sweat_loss_g": 5813,
                "t_cr": 37.5,
                "d_lim_loss_95": 310,
            },
        ),
        pytest.param(
            {
                "tdb": [35, 35],
                "tr": 74.6,
                "rh": 30,
                "v": 1.0,
                "met": 250 / met_to_w_m2 / a_dubois,
                "clo": 1,
                "posture": ["sitting", "sitting"],
                "wme": 0,
                "a_p": 0.2,
                "f_r": 0.85,
                "model": "7933-2023",
                "duration": 480,
                "limit_inputs": False,
                "acclimatized": 100,
                "round_output": False,
            },
            {
                "sweat_loss_g": [5813, 5813],
                "t_cr": [37.5, 37.5],
                "d_lim_loss_95": [310, 310],
            },
            id="array-input",
        ),
    ],
)
def test_2023_standard(inputs, expected) -> None:
    """Test the 2023 PHS model with various inputs."""
    result = phs(**inputs)
    assert result.sweat_loss_g == pytest.approx(expected["sweat_loss_g"], rel=0.025)
    assert result.t_cr == pytest.approx(expected["t_cr"], abs=0.3)
    assert result.d_lim_loss_95 == pytest.approx(expected["d_lim_loss_95"], abs=10)
    if "d_lim_t_re" in expected:
        assert result.d_lim_t_re == pytest.approx(expected["d_lim_t_re"], abs=10)


def test_value_acclimatized() -> None:
    """Test that the function raises ValueError for invalid acclimatized values."""
    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.58,
            clo=0.5,
            posture="standing",
            acclimatized=101,
        )

    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.58,
            clo=0.5,
            posture="standing",
            acclimatized=-1,
        )


def test_value_weight() -> None:
    """Test that the function raises a ValueError for invalid weight values."""
    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.58,
            clo=0.5,
            posture="standing",
            weight=1001,
        )

    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.58,
            clo=0.5,
            posture="standing",
            weight=0,
        )


def test_value_drink() -> None:
    """Test that drink input is within valid range."""
    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.58,
            clo=0.5,
            posture="standing",
            drink=0.5,
        )
    with pytest.raises(ValueError):
        phs(
            tdb=40,
            tr=40,
            rh=33.85,
            v=0.3,
            met=2.58,
            clo=0.5,
            posture="standing",
            drink=2,
        )


@pytest.mark.parametrize("model", ["7933-2004", "7933-2023"])
def test_applicability_limit_is_on_tr_minus_tdb_not_tr(model: str) -> None:
    """ISO 7933 Annex A, Table A.1 limits (tr - tdb) to (0, 60), not raw tr.

    tr=10, tdb=40 gives tr - tdb = -30 (out of range) even though the raw tr
    value of 10 would have passed the old, incorrect (0, 60) check on tr
    alone. See #225.
    """
    result = phs(
        tdb=40,
        tr=10,
        rh=50,
        v=0.5,
        met=2.0,
        clo=0.5,
        posture="standing",
        model=model,
        round_output=False,
    )
    assert math.isnan(result.t_cr)


def test_applicability_limit_metabolic_rate_is_standard_specific() -> None:
    """ISO 7933 Annex A, Table A.1: the metabolic rate range is 56-250 W/m2 in the 2023
    standard but 100-450 W/m2 in the 2004 standard, not the 2004 range for both.

    met=1.5 (~87.2 W/m2) falls inside the 2023 range and outside the 2004 one. See #225.
    """
    kwargs = dict(
        tdb=30,
        tr=40,
        rh=50,
        v=0.5,
        met=1.5,
        clo=0.5,
        posture="standing",
        round_output=False,
    )
    assert not math.isnan(phs(model="7933-2023", **kwargs).t_cr)
    assert math.isnan(phs(model="7933-2004", **kwargs).t_cr)


def test_2023_forces_skin_temperature_to_equilibrium_on_minute_one() -> None:
    """ISO 7933:2023 Annex E forces t_sk to its equilibrium value on minute 1, removing
    the exponential lag for that one step; this special case is not present in the 2004
    Annex E reference code. See #217.

    Starting from an artificial t_sk=20 far from equilibrium, after 1 minute the 2023
    model should already be close to equilibrium (>30 degC) while the 2004 model, still
    exponentially averaging, remains much closer to the artificial starting value.
    """
    kwargs = dict(
        tdb=40,
        tr=40,
        rh=35,
        v=0.3,
        met=2.58,
        clo=0.5,
        posture="standing",
        wme=0,
        duration=1,
        limit_inputs=False,
        round_output=False,
        t_sk=20.0,
    )
    result_2023 = phs(model="7933-2023", **kwargs)
    result_2004 = phs(model="7933-2004", **kwargs)
    assert result_2023.t_sk > 30
    assert result_2004.t_sk < result_2023.t_sk


PHS_OUTPUTS = [
    "t_re",
    "t_sk",
    "t_cr",
    "t_cr_eq",
    "t_sk_t_cr_wg",
    "d_lim_loss_50",
    "d_lim_loss_95",
    "d_lim_t_re",
    "sweat_rate_watt",
    "sweat_loss_g",
    "evap_load_wm2_min",
]
PHS_BASE = dict(tdb=40, tr=40, rh=33.85, v=0.3, met=2.5, clo=0.5, posture="standing")


# Before #524 was fixed, each of these calls never returned.
@pytest.mark.parametrize(
    ("name", "bad"),
    [
        (name, bad)
        for name in [
            "tdb",
            "tr",
            "v",
            "rh",
            "met",
            "clo",
            "wme",
            "t_sk",
            "t_cr",
            "t_re",
            "t_cr_eq",
            "sweat_rate_watt",
            "evap_load_wm2_min",
        ]
        for bad in [np.nan, np.inf]
    ]
    + [("t_sk_t_cr_wg", np.nan)],
)
@pytest.mark.parametrize("model", ["7933-2004", "7933-2023"])
def test_phs_returns_nan_for_non_finite_point_inputs(name, bad, model) -> None:
    """A non-finite value gives NaN for that point only, and the call returns (#524)."""
    reference = phs(**PHS_BASE, model=model, limit_inputs=False, round_output=False)
    default = {
        "wme": 0,
        "t_sk": 34.1,
        "t_cr": 36.8,
        "t_re": 36.8,
        "t_cr_eq": 36.8,
        "t_sk_t_cr_wg": 0.3,
        "sweat_rate_watt": 0,
        "evap_load_wm2_min": 0,
    }
    inputs = {**PHS_BASE, name: [PHS_BASE.get(name, default.get(name)), bad]}
    inputs["posture"] = ["standing", "standing"]
    result = phs(**inputs, model=model, limit_inputs=False, round_output=False)
    scalar = phs(
        **{**PHS_BASE, name: bad}, model=model, limit_inputs=False, round_output=False
    )

    for output in PHS_OUTPUTS:
        assert getattr(result, output)[0] == getattr(reference, output)
        assert np.isnan(getattr(result, output)[1])
        assert np.isnan(getattr(scalar, output))


@pytest.mark.parametrize(
    "name", ["weight", "height", "i_mst", "a_p", "walk_sp", "theta", "f_r"]
)
def test_phs_returns_nan_for_non_finite_scalar_inputs(name) -> None:
    """A NaN optional input that applies to every point gives NaN everywhere (#524)."""
    result = phs(**PHS_BASE, **{name: np.nan})
    for output in PHS_OUTPUTS:
        assert np.isnan(getattr(result, output))


def test_phs_returns_nan_when_the_temperature_loop_does_not_converge() -> None:
    """A tiny weight makes the core temperature diverge, so the iteration cap returns
    NaN instead of looping forever (#524)."""
    result = phs(**PHS_BASE, weight=1e-300, limit_inputs=False)
    assert np.isnan(result.t_re)


@pytest.mark.parametrize("name", ["height", "i_mst"])
@pytest.mark.parametrize("value", [0, -1])
def test_phs_rejects_zero_height_and_i_mst(name, value) -> None:
    """The kernel divides by the body surface area and by i_mst (#523)."""
    with pytest.raises(ValueError, match=name):
        phs(**PHS_BASE, **{name: value}, limit_inputs=False)


@pytest.mark.parametrize("model", ["7933-2004", "7933-2023"])
def test_phs_zero_clo_returns_the_skin_temperature_limit(model) -> None:
    """At clo=0 and v=0.3 the dynamic clothing insulation is exactly 0, which raised a
    ZeroDivisionError that the parallel kernel lost, leaving uninitialised memory in
    the output (#523). The clothing surface is then the skin, so the result is the
    limit of a vanishingly small clo.
    """
    kwargs = dict(PHS_BASE, model=model, limit_inputs=False, round_output=False)
    scalar = phs(**{**kwargs, "clo": 0})
    thin = phs(**{**kwargs, "clo": 1e-9})
    reference = phs(**kwargs)
    result = phs(**{**kwargs, "clo": [0.5, 0, 0.5], "posture": ["standing"] * 3})

    for output in PHS_OUTPUTS:
        assert getattr(scalar, output) == pytest.approx(getattr(thin, output), rel=1e-6)
        assert getattr(result, output)[1] == getattr(scalar, output)
        assert getattr(result, output)[0] == getattr(reference, output)
        assert getattr(result, output)[2] == getattr(reference, output)

    # clo=0 is outside the applicability limits, so limit_inputs=True still masks it
    assert np.isnan(phs(**{**PHS_BASE, "clo": 0}).t_re)
