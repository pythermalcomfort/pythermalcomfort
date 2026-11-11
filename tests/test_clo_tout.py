import numpy as np
import pytest

from pythermalcomfort.models import clo_tout
from pythermalcomfort.utilities import Units
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_clo_tout(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the clothing thermal insulation (Clo) for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.CLO_TOUT.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = clo_tout(
            tout=inputs["tout"],
            units=inputs.get("units", Units.SI.value),
        )

        validate_result(result, outputs, tolerance)


@pytest.mark.parametrize("units", ["SI", "IP"])
def test_clo_tout_returns_nan_for_a_nan_input(units) -> None:
    """A NaN tout gives NaN, not the 0.46 of the warmest range."""
    tout = [27, 0, np.nan] if units == "SI" else [80.6, 32, np.nan]
    result = clo_tout(tout=tout, units=units)

    np.testing.assert_equal(result.clo_tout, [0.46, 0.82, np.nan])
