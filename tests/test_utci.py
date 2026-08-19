import numpy as np
import pytest

from pythermalcomfort.models import utci
from pythermalcomfort.models.utci import _utci_optimized
from pythermalcomfort.utilities import p_sat
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_utci(get_test_url, retrieve_data) -> None:
    """Test that the UTCI function calculates correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.UTCI.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = utci(**inputs)

        validate_result(result, outputs, tolerance)


def test_utci_optimized() -> None:
    """Test that the optimized UTCI function calculates correctly for various inputs."""
    np.testing.assert_equal(
        np.around(_utci_optimized([25, 27], 1, 1, 1.5), 2),
        [24.73, 26.57],
    )


def test_utci_ip_uses_si_thresholds_for_stress_category() -> None:
    """Test that IP stress categories use the underlying SI UTCI value."""
    result = utci(tdb=77, tr=77, v=3.28084, rh=50, units="IP")

    assert result.utci == 76.3
    assert result.stress_category == "no thermal stress"


def test_utci_ip_vector_stress_category() -> None:
    """Test that IP vector stress categories use SI UTCI values."""
    result = utci(
        tdb=[77, 104],
        tr=[77, 104],
        v=[3.28084, 3.28084],
        rh=[50, 50],
        units="IP",
    )

    np.testing.assert_allclose(result.utci, [76.3, 110.5])
    np.testing.assert_array_equal(
        result.stress_category,
        ["no thermal stress", "very strong heat stress"],
    )


def test_utci_stress_category_uses_rounded_si_value_by_default() -> None:
    """Test that default SI category mapping preserves rounded-output behavior."""
    rounded = utci(tdb=26.27, tr=26.27, v=1, rh=50)
    unrounded = utci(tdb=26.27, tr=26.27, v=1, rh=50, round_output=False)

    assert rounded.utci == 26.0
    assert rounded.stress_category == "no thermal stress"
    assert unrounded.utci > 26.0
    assert unrounded.stress_category == "moderate heat stress"


def test_utci_ip_stress_category_uses_unrounded_si_value_when_not_rounding() -> None:
    """Test that unrounded IP output maps categories from unrounded SI values."""
    result = utci(
        tdb=79.286,
        tr=79.286,
        v=3.28084,
        rh=50,
        units="IP",
        round_output=False,
    )

    assert result.utci > 78.8
    assert result.stress_category == "moderate heat stress"


def test_utci_ip_out_of_range_stress_category_is_nan() -> None:
    """Test that out-of-range IP inputs produce NaN stress categories."""
    result = utci(tdb=1000, tr=1000, v=3.28084, rh=50, units="IP")

    assert np.isnan(result.utci).item()
    assert np.isnan(result.stress_category).item()

    vector_result = utci(
        tdb=[77, 1000],
        tr=[77, 1000],
        v=[3.28084, 3.28084],
        rh=[50, 50],
        units="IP",
    )

    np.testing.assert_allclose(vector_result.utci, [76.3, np.nan], equal_nan=True)
    assert vector_result.stress_category[0] == "no thermal stress"
    assert np.isnan(vector_result.stress_category[1])


def test_utci_saturation_vapour_pressure_matches_p_sat() -> None:
    """Test that UTCI's vapour pressure matches the independent p_sat formulation.

    Regression test for #372, where np.log1p was used instead of np.log in the
    Hardy equation, inflating the saturation vapour pressure by ~1%. Saturation
    vapour pressure grows exponentially with temperature, so a given relative
    error only becomes detectable in hot, humid conditions: at 25 C / 50% RH it
    shifts UTCI by ~0.03 C, but at 40 C / 80% RH it reaches ~0.7 C.
    """
    tdb = 40
    tr = 40
    v = 1
    rh = 80
    pa = p_sat(tdb) * (rh / 100) / 1000
    expected = _utci_optimized(tdb, v, tr - tdb, pa)
    actual = utci(tdb=tdb, tr=tr, v=v, rh=rh, round_output=False).utci
    assert actual == pytest.approx(expected, abs=0.1)
