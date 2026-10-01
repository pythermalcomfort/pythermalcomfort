import numpy as np
import pytest

from pythermalcomfort.models import solar_gain
from tests.conftest import Urls, retrieve_reference_table, validate_result


def test_solar_gain(get_test_url, retrieve_data) -> None:
    """Test that the solar gain function calculates correctly for various inputs."""
    reference_table = retrieve_reference_table(
        get_test_url,
        retrieve_data,
        Urls.SOLAR_GAIN.name,
    )
    tolerance = reference_table["tolerance"]

    for entry in reference_table["data"]:
        inputs = entry["inputs"]
        outputs = entry["outputs"]
        result = solar_gain(**inputs)

        validate_result(result, outputs, tolerance)


def test_solar_gain_regression_values() -> None:
    """Pin erf/delta_mrt against the reference (pre-numba) implementation.

    Values were captured from the implementation before its table-lookup
    kernel was rewritten for numba (posture strings -> integer codes, table
    lists -> np.array, plain-Python loop -> njit/prange). Confirmed to match
    bit-for-bit across a wide random sweep (all 3 postures) plus exact grid
    boundary points. The supine value was updated when the floor-reflected
    term was changed to use the true solar altitude.
    """
    cases = [
        (0, 120, 800, 0.5, 0.5, 0.5, "sitting", 43.2839, 10.3649),
        (30, 60, 600, 0.6, 0.4, 0.6, "standing", 52.8099, 12.1402),
        (45, 90, 500, 0.7, 0.3, 0.7, "supine", 61.447, 14.1257),
        (90, 0, 1000, 1.0, 1.0, 1.0, "sitting", 326.6804, 78.2281),
        (15, 165, 250, 0.2, 0.1, 0.9, "standing", 8.9043, 2.047),
    ]
    for alt, sharp, rad, trans, svv, bes, posture, exp_erf, exp_d_mrt in cases:
        result = solar_gain(
            sol_altitude=alt,
            sharp=sharp,
            sol_radiation_dir=rad,
            sol_transmittance=trans,
            f_svv=svv,
            f_bes=bes,
            posture=posture,
            round_output=False,
        )
        assert np.isclose(result.erf, exp_erf, atol=1e-3)
        assert np.isclose(result.delta_mrt, exp_d_mrt, atol=1e-3)


def test_solar_gain_supine_reflected_uses_solar_altitude() -> None:
    """Supine: the transposed angles are used only for the fp lookup.

    The floor-reflected term depends on the irradiance on the horizontal floor,
    I_dir * sin(sol_altitude) + I_diff, so it must use the true solar altitude
    and not the body-relative altitude used to read the fp table. Reference
    values are the supine cases of the CBE Thermal Comfort Tool ERF tests.
    """
    cases = [
        # sol_altitude, sharp, sol_radiation_dir, sol_transmittance, f_svv, f_bes,
        # expected erf, expected delta_mrt
        (45, 0, 700, 0.8, 0.2, 0.5, 60.9, 14.0),
        (45, 45, 700, 0.8, 0.2, 0.5, 65.8, 15.1),
        (45, 45, 800, 0.5, 0.5, 0.5, 70.9, 16.3),
    ]
    for alt, sharp, rad, trans, svv, bes, exp_erf, exp_d_mrt in cases:
        result = solar_gain(
            sol_altitude=alt,
            sharp=sharp,
            sol_radiation_dir=rad,
            sol_transmittance=trans,
            f_svv=svv,
            f_bes=bes,
            asw=0.7,
            posture="supine",
        )
        assert result.erf == pytest.approx(exp_erf)
        assert result.delta_mrt == pytest.approx(exp_d_mrt)


def test_solar_gain_out_of_range_returns_nan() -> None:
    """Test that out-of-domain sol_altitude/sharp warn and return NaN.

    The fp lookup table only covers sol_altitude in [0, 90] and sharp in
    [0, 180]; outside that range there is no valid span to look up. Regression
    test for a case where this used to silently wrap to a plausible-looking
    but wrong value instead of failing loudly.
    """
    with pytest.warns(UserWarning, match="sol_altitude"):
        result = solar_gain(
            sol_altitude=200,
            sharp=90,
            sol_radiation_dir=800,
            sol_transmittance=0.5,
            f_svv=0.5,
            f_bes=0.5,
        )
    assert np.isnan(result.erf)
    assert np.isnan(result.delta_mrt)

    with pytest.warns(UserWarning, match="sharp"):
        result = solar_gain(
            sol_altitude=45,
            sharp=300,
            sol_radiation_dir=800,
            sol_transmittance=0.5,
            f_svv=0.5,
            f_bes=0.5,
        )
    assert np.isnan(result.erf)
    assert np.isnan(result.delta_mrt)

    # mixed valid/invalid array: only the invalid element becomes NaN
    with pytest.warns(UserWarning, match="sol_altitude"):
        result = solar_gain(
            sol_altitude=[45, 200],
            sharp=[90, 90],
            sol_radiation_dir=800,
            sol_transmittance=0.5,
            f_svv=0.5,
            f_bes=0.5,
        )
    assert not np.isnan(result.erf[0])
    assert np.isnan(result.erf[1])


def test_solar_gain_array() -> None:
    """Test that the solar gain function works with arrays.

    The inputs are the two sitting rows of test_solar_gain_regression_values, so
    the array call must return the same values as those pinned scalar results.
    """
    result = solar_gain(
        sol_altitude=[0, 90],
        sharp=[120, 0],
        sol_radiation_dir=[800, 1000],
        sol_transmittance=[0.5, 1.0],
        f_svv=[0.5, 1.0],
        f_bes=[0.5, 1.0],
        posture="sitting",
        round_output=False,
    )
    assert np.allclose(result.erf, [43.2839, 326.6804], atol=1e-3)
    assert np.allclose(result.delta_mrt, [10.3649, 78.2281], atol=1e-3)
