from pythermalcomfort.models import at
from tests.conftest import Urls, is_equal, retrieve_reference_table, validate_result


def test_at(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the AT correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.AT.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = at(**inputs)

        validate_result(result, outputs, tolerance)


def test_at_list_input() -> None:
    """Test that the function calculates the AT correctly for list inputs."""
    result = at(tdb=[25, 25, 25], rh=[30, 30, 30], v=[0.1, 0.1, 0.1])
    assert is_equal(result.at, [24.1, 24.1, 24.1], 0.1)


def test_at_q() -> None:
    """Test that the function calculates the AT correctly for given inputs.

    BoM formula with radiation: AT = tdb + 0.348 e - 0.70 v + 0.70 q / (v + 10) - 4.25,
    with e = 9.47 hPa at 25 °C and 30% RH, gives 25 + 3.30 - 0.07 + 6.93 - 4.25 = 30.9.
    """
    result = at(tdb=25, rh=30, v=0.1, q=100)
    assert is_equal(result.at, 30.9, 0.1)
