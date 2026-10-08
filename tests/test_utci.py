import numpy as np
import pytest

from pythermalcomfort.models import utci
from pythermalcomfort.models.utci import _utci_optimized
from pythermalcomfort.psychrometrics import p_sat
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


@pytest.mark.parametrize("tdb,rh", [(40, 80), (50, 100)])
@pytest.mark.parametrize("units", ["SI", "IP"])
def test_utci_rejects_excess_vapour_pressure(tdb, rh, units) -> None:
    """Default limits must not classify a humid heat extrapolation as cold stress."""
    temperature = tdb if units == "SI" else tdb * 1.8 + 32
    wind = 1 if units == "SI" else 3.281
    with pytest.warns(UserWarning, match="pa"):
        result = utci(tdb=temperature, tr=temperature, v=wind, rh=rh, units=units)
    assert np.isnan(result.utci)
    assert np.isnan(result.stress_category)


@pytest.mark.parametrize("rh", [-1, 101])
def test_utci_rejects_relative_humidity_outside_physical_range(rh) -> None:
    with pytest.warns(UserWarning) as records:
        result = utci(tdb=25, tr=25, v=1, rh=rh)
    assert any("'rh'" in str(record.message) for record in records)
    assert np.isnan(result.utci)
    assert np.isnan(result.stress_category)


@pytest.mark.parametrize("rh", [0, 100])
def test_utci_accepts_relative_humidity_endpoints(rh) -> None:
    limited = utci(tdb=25, tr=25, v=1, rh=rh, round_output=False)
    unlimited = utci(tdb=25, tr=25, v=1, rh=rh, limit_inputs=False, round_output=False)
    assert np.isfinite(limited.utci)
    assert limited.utci == unlimited.utci
    assert limited.stress_category == unlimited.stress_category


def test_utci_vapour_pressure_limit_is_inclusive() -> None:
    # Hardy's equation at 40 C: this RH gives exactly 5.0 kPa in float64.
    rh_at_limit = 67.70206528209202
    rh = [rh_at_limit - 1e-10, rh_at_limit, rh_at_limit + 1e-10]
    with pytest.warns(UserWarning, match="pa"):
        limited = utci(tdb=40, tr=40, v=1, rh=rh, round_output=False)
    unlimited = utci(tdb=40, tr=40, v=1, rh=rh, limit_inputs=False, round_output=False)
    np.testing.assert_array_equal(np.isnan(limited.utci), [False, False, True])
    np.testing.assert_array_equal(limited.utci[:2], unlimited.utci[:2])
    np.testing.assert_array_equal(
        limited.stress_category[:2], unlimited.stress_category[:2]
    )
    assert np.isnan(limited.stress_category[2])


@pytest.mark.parametrize("units", ["SI", "IP"])
def test_utci_humidity_guard_broadcasts_without_masking_valid_neighbors(units) -> None:
    temperature = np.array([[25.0], [50.0]])
    if units == "IP":
        temperature = temperature * 1.8 + 32
    wind = 1 if units == "SI" else 3.281
    inputs = dict(tdb=temperature, tr=temperature, v=wind, rh=[0, 50, 101], units=units)
    with pytest.warns(UserWarning):
        limited = utci(**inputs)
    unlimited = utci(**inputs, limit_inputs=False)
    invalid = np.array([[False, False, True], [False, True, True]])
    np.testing.assert_array_equal(np.isnan(limited.utci), invalid)
    np.testing.assert_array_equal(limited.utci[~invalid], unlimited.utci[~invalid])
    np.testing.assert_array_equal(
        limited.stress_category[~invalid], unlimited.stress_category[~invalid]
    )
    assert all(np.isnan(category) for category in limited.stress_category[invalid])


def test_utci_explicit_extrapolation_preserves_original_polynomial() -> None:
    result = utci(tdb=50, tr=50, v=1, rh=100, limit_inputs=False, round_output=False)
    assert result.utci == pytest.approx(-326.0792322091629)
    assert result.stress_category == "extreme cold stress"


def test_utci_nan_humidity_propagates_without_masking_valid_neighbors() -> None:
    result = utci(tdb=25, tr=25, v=1, rh=[50, np.nan])
    np.testing.assert_array_equal(result.utci, [24.6, np.nan])
    assert result.stress_category[0] == "no thermal stress"
    assert np.isnan(result.stress_category[1])


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
    # This pressure-formula regression intentionally evaluates beyond 5 kPa.
    actual = utci(
        tdb=tdb, tr=tr, v=v, rh=rh, limit_inputs=False, round_output=False
    ).utci
    assert actual == pytest.approx(expected, abs=0.1)
