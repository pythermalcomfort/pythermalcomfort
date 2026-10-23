import numpy as np
import pytest

from pythermalcomfort.models import bfu_rest


def test_bfu_rest_scalar():
    result = bfu_rest(tdb=36.0, rh=60.0, population="YNG")

    assert not hasattr(result, "population")
    assert result.delta_storage < 0  # fan improves evaporative margin


def test_bfu_rest_array():
    tdb = np.array([35.0, 40.0])
    rh = np.array([50.0, 65.0])

    result = bfu_rest(tdb=tdb, rh=rh, population="OLD", round_output=False)

    for attribute in (
        result.delta_storage,
        result.e_req_fan,
        result.e_max_fan,
        result.e_req_no_fan,
        result.e_max_no_fan,
    ):
        assert isinstance(attribute, np.ndarray)
        assert attribute.shape == tdb.shape


def test_bfu_rest_invalid_population():
    with pytest.raises(ValueError):
        bfu_rest(tdb=35.0, rh=50.0, population="INVALID")


@pytest.mark.parametrize("population", ["YNG", "OLD", "MEDS"])
def test_population_populations_and_case_normalization(population):
    result = bfu_rest(36, 60, population=population.lower(), round_output=False)
    expected = bfu_rest(36, 60, population=population, round_output=False)
    assert not hasattr(result, "population")
    assert result.delta_storage == pytest.approx(expected.delta_storage)


@pytest.mark.parametrize("rh", [-1, 101])
def test_invalid_humidity(rh):
    with pytest.raises(ValueError, match="rh"):
        bfu_rest(36, rh)


def test_broadcasting_and_rounding():
    raw = bfu_rest([[35], [40]], [50, 65], round_output=False)
    rounded = bfu_rest([[35], [40]], [50, 65])
    assert raw.delta_storage.shape == (2, 2)
    np.testing.assert_array_equal(
        rounded.delta_storage, np.around(raw.delta_storage, 1)
    )


def test_incompatible_shapes():
    with pytest.raises(ValueError, match="broadcast"):
        bfu_rest([35, 40], [50, 60, 70])


def test_population_enum():
    from pythermalcomfort.utilities import BFURestPopulation

    for population in BFURestPopulation:
        actual = bfu_rest(tdb=36, rh=60, population=population)
        expected = bfu_rest(tdb=36, rh=60, population=population.value)
        assert actual.delta_storage == expected.delta_storage


@pytest.mark.parametrize("name", ["tdb", "rh"])
@pytest.mark.parametrize("value", [None, np.nan, np.inf])
def test_nonfinite_or_missing_inputs(name, value):
    inputs = dict(tdb=36, rh=60)
    inputs[name] = value
    with pytest.raises((ValueError, TypeError)):
        bfu_rest(**inputs)


def test_storage_difference_matches_evaporative_balance():
    result = bfu_rest(tdb=[35, 40], rh=[50, 65], round_output=False)
    np.testing.assert_allclose(
        result.delta_storage,
        (result.e_req_fan - result.e_max_fan)
        - (result.e_req_no_fan - result.e_max_no_fan),
    )


def test_input_dataclasses_validate_during_initialization():
    from pythermalcomfort.classes_input import BFUOccupationalInputs, BFURestInputs

    with pytest.raises(ValueError, match="rh"):
        BFURestInputs(tdb=36, rh=101)
    with pytest.raises(ValueError, match="v_no_fan"):
        BFUOccupationalInputs(tdb=30, rh=60, met=2, clo=0.28, v=3.5, v_no_fan=-1)
