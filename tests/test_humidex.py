import math

import pytest

from pythermalcomfort.models import humidex
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_humidex(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the Humidex correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.HUMIDEX.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = humidex(**inputs)

        validate_result(result, outputs, tolerance)

    with pytest.raises(TypeError):
        humidex("25", 50)

    with pytest.raises(TypeError):
        humidex(25, "50")

    with pytest.raises(ValueError):
        humidex(tdb=25, rh=110)

    with pytest.raises(ValueError):
        humidex(25, -10)


def test_humidex_masterson() -> None:
    """Test the Masterson model for Humidex calculations."""
    # TODO move this to shared test
    # I got these values from
    # https://publications.gc.ca/collections/collection_2018/eccc/En57-23-1-79-eng.pdf
    result = humidex(tdb=21, rh=100, model="masterson")
    assert math.isclose(result.humidex, 29.3, abs_tol=0.1)
    assert result.discomfort == "Little or no discomfort"

    result = humidex(tdb=34, rh=100, model="masterson")
    assert math.isclose(result.humidex, 58.9, abs_tol=0.01)

    result = humidex(tdb=43, rh=20, model="masterson")
    assert math.isclose(result.humidex, 47.1, abs_tol=0.01)

    result = humidex(tdb=30, rh=30, model="masterson")
    assert math.isclose(result.humidex, 31.5, abs_tol=0.01)

    result = humidex(tdb=31, rh=55, model="masterson")
    assert math.isclose(result.humidex, 39.3, abs_tol=0.01)


@pytest.mark.parametrize("model", ["rana", "masterson"])
@pytest.mark.parametrize(("tdb", "rh"), [(30, math.nan), (math.nan, 50)])
def test_humidex_returns_nan_for_a_nan_input(model, tdb, rh) -> None:
    """A NaN input gives a NaN humidex and discomfort, not "Heat stroke probable"."""
    result = humidex(tdb=[30, tdb], rh=[60, rh], model=model)

    assert result.discomfort[0] == "Evident discomfort"
    assert math.isnan(result.humidex[1])
    assert math.isnan(result.discomfort[1])
