import warnings

import numpy as np
import pytest

from pythermalcomfort.models.vertical_tmp_grad_ppd import vertical_tmp_grad_ppd
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_vertical_tmp_grad_ppd(get_test_url, retrieve_data) -> None:
    """Test that the function calculates the output correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.VERTICAL_TMP_GRAD_PPD.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = vertical_tmp_grad_ppd(**inputs)

        validate_result(result, outputs, tolerance)

    # Test for ValueError
    np.isclose(
        vertical_tmp_grad_ppd(25, 25, 0.3, 50, 1.2, 0.5, 7).ppd_vg,
        np.nan,
        equal_nan=True,
    )


def test_vertical_tmp_grad_ppd_limit_inputs_false_suppresses_warning_and_nan() -> None:
    """limit_inputs=False bypasses range checks: no UserWarning, no NaN mask."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = vertical_tmp_grad_ppd(
            50,
            25,
            0.1,
            50,
            1.2,
            0.5,
            7,
            limit_inputs=False,
        )
    assert not np.isnan(result.ppd_vg)


def test_vertical_tmp_grad_ppd_limit_inputs_true_still_warns_and_nans() -> None:
    """limit_inputs=True (default) keeps current behavior: warning + NaN mask."""
    with pytest.warns(UserWarning):
        result = vertical_tmp_grad_ppd(50, 25, 0.1, 50, 1.2, 0.5, 7)
    assert np.isnan(result.ppd_vg)


def test_vertical_tmp_grad_ppd_is_never_negative() -> None:
    """ppd_vg is 0 when the logistic model is below the 34.5 % baseline."""
    # without the clamp these inputs give -4.9 % and -7.9 %
    result = vertical_tmp_grad_ppd(
        tdb=[24, 24, 25],
        tr=[24, 24, 25],
        vr=0.1,
        rh=50,
        met=1.2,
        clo=0.5,
        vertical_tmp_grad=[1, 0, 7],
    )
    np.testing.assert_allclose(result.ppd_vg, [0.0, 0.0, 12.6])
    np.testing.assert_array_equal(result.acceptability, [True, True, False])
