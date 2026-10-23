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


@pytest.mark.parametrize("w_max", [0.01, 1.2, [0.5, 0.05]])
def test_two_nodes_rejects_w_max_out_of_range(w_max) -> None:
    """w_max below 0.06 would make p_rsw negative at the wettedness cap (#489)."""
    with pytest.raises(ValueError, match="w_max"):
        two_nodes_gagge(tdb=40, tr=40, v=0.1, rh=50, met=2, clo=0.5, w_max=w_max)


@pytest.mark.parametrize("w_max", [False, 0, 0.06, 0.5, 1])
def test_two_nodes_accepts_w_max_in_range(w_max) -> None:
    """False and 0 still mean that w_max is computed from v and clo."""
    two_nodes_gagge(tdb=40, tr=40, v=0.1, rh=50, met=2, clo=0.5, w_max=w_max)
