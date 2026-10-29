import numpy as np

from pythermalcomfort.models import (
    set_tmp,
)
from tests.conftest import Urls, retrieve_reference_table, validate_result

tdb = []
tr = []
v = []
rh = []
met = []
clo = []
set_exp = []


def test_set_url(get_test_url, retrieve_data) -> None:
    """Test the SET model with various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.SET.name,
    )
    tolerance = reference_table["tolerance"]
    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        inputs["round_output"] = True
        inputs["limit_inputs"] = False
        result = set_tmp(**inputs)

        validate_result(result, outputs, tolerance)


def test_set_npnan() -> None:
    """Test that the function returns np.nan when inputs are np.nan."""
    np.testing.assert_equal(
        set_tmp(
            [41, 20, 20, 20, 20, 39],
            [20, 41, 20, 20, 20, 39],
            [0.1, 0.1, 2.1, 0.1, 0.1, 0.1],
            50,
            [1.1, 1.1, 1.1, 0.7, 1.1, 3.9],
            [0.5, 0.5, 0.5, 0.5, 2.1, 1.9],
        ).set,
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    )


def test_set_continuous_at_critical_wettedness() -> None:
    """SET should be continuous when skin wettedness reaches w_max.

    At the cap, p_rsw must invert w = 0.06 + 0.94 * p_rsw, i.e. (w_max - 0.06) / 0.94 as in
    Gagge et al. (1986).
    """
    t = np.round(np.arange(32.5, 34.0, 0.001), 3)
    result = set_tmp(
        tdb=t,
        tr=t,
        v=0.1,
        rh=70,
        met=1.1,
        clo=0.5,
        round_output=False,
        limit_inputs=False,
    ).set
    max_step = np.max(np.abs(np.diff(result)))
    assert max_step < 0.01
    assert np.all(np.diff(result) > 0)  # monotonic, so cooling_effect has a single root
