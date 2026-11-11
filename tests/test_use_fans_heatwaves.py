import numpy as np
import pytest

from pythermalcomfort.models import use_fans_heatwaves
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_use_fans_heatwaves(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the use of fans during heatwaves correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.USE_FANS_HEATWAVES.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = use_fans_heatwaves(**inputs)

        validate_result(result, outputs, tolerance)


@pytest.mark.parametrize("name", ["tdb", "met", "wme", "body_surface_area", "p_atm"])
def test_use_fans_heatwaves_returns_nan_for_a_nan_input(name) -> None:
    """A NaN gives NaN outputs and flags for that point only."""
    inputs = {
        "tdb": 39,
        "tr": 39,
        "v": 0.8,
        "rh": 40,
        "met": 1.2,
        "clo": 0.5,
        "wme": 0,
        "body_surface_area": 1.8258,
        "p_atm": 101325,
    }
    reference = use_fans_heatwaves(**inputs)
    inputs[name] = [inputs[name], np.nan]
    result = use_fans_heatwaves(**inputs)

    for output in (
        "t_core",
        "heat_strain",
        "heat_strain_blood_flow",
        "heat_strain_w",
        "heat_strain_sweating",
    ):
        assert getattr(result, output)[0] == getattr(reference, output)
        assert np.isnan(getattr(result, output)[1])
