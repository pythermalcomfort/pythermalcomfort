import numpy as np
import pytest

from pythermalcomfort.models import two_nodes_gagge
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_two_nodes(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the two nodes Gagge model correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.TWO_NODES.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = two_nodes_gagge(**inputs)

        validate_result(result, outputs, tolerance)


@pytest.mark.parametrize(
    "w_max", [0.01, 0.06, 1.2, [0.5, 0.05], np.nan, [0.5, np.nan], True, [True, 0.5]]
)
def test_two_nodes_rejects_w_max_out_of_range(w_max) -> None:
    """Reject w_max that would break the wettedness cap (#489).

    Below 0.06 p_rsw is negative at the cap, at exactly 0.06 disc divides by zero
    when met <= 1, NaN disables the cap, and True would silently mean a cap of 1.
    """
    with pytest.raises(ValueError, match="w_max"):
        two_nodes_gagge(tdb=40, tr=40, v=0.1, rh=50, met=1, clo=0.5, w_max=w_max)


@pytest.mark.parametrize("w_max", [False, 0, 0.0601, 0.5, 1])
def test_two_nodes_accepts_w_max_in_range(w_max) -> None:
    """False and 0 still mean that w_max is computed from v and clo."""
    two_nodes_gagge(tdb=40, tr=40, v=0.1, rh=50, met=2, clo=0.5, w_max=w_max)


NUMERIC_INPUTS = {
    "tdb": 30,
    "tr": 30,
    "v": 0.2,
    "rh": 50,
    "met": 1.2,
    "clo": 0.5,
    "wme": 0,
    "body_surface_area": 1.8258,
    "p_atm": 101325,
}


@pytest.mark.parametrize("name", list(NUMERIC_INPUTS))
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_two_nodes_returns_nan_for_non_finite_inputs(name, bad) -> None:
    """A non-finite value gives NaN for that point only, instead of StopIteration."""
    inputs = {**NUMERIC_INPUTS, name: [NUMERIC_INPUTS[name], bad]}
    result = two_nodes_gagge(**inputs)
    reference = two_nodes_gagge(**NUMERIC_INPUTS)

    assert result.disc[0] == reference.disc
    assert np.isnan(result.disc[1])
    assert np.isnan(result.t_core[1])
    assert np.isnan(result.set[1])


@pytest.mark.parametrize("name", list(NUMERIC_INPUTS))
def test_two_nodes_calculate_ce_returns_nan_for_non_finite_inputs(name) -> None:
    """The SET-only path used by cooling_effect handles a NaN the same way."""
    inputs = {**NUMERIC_INPUTS, name: [NUMERIC_INPUTS[name], np.nan]}
    result = two_nodes_gagge(**inputs, calculate_ce=True)
    reference = two_nodes_gagge(**NUMERIC_INPUTS, calculate_ce=True)

    assert result.set[0] == reference.set
    assert np.isnan(result.set[1])
