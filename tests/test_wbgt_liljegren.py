from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import FrozenInstanceError, fields
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from pythermalcomfort.classes_input import WBGTLiljegrenInputs
from pythermalcomfort.classes_return import WBGTLiljegren
from pythermalcomfort.models import wbgt, wbgt_liljegren
from tests.conftest import is_equal

# Reference inputs and values from the released lwbgt 0.3.0 fixtures:
# https://github.com/zyf0717/lwbgt/blob/v0.3.0/tests/python/_fixtures.py
# Its test_api.py freezes the Singapore WBGT float32 bits as 59020242.
SINGAPORE = dict(
    tdb=32.1,
    rh=68.0,
    v=2.8,
    sol_radiation_global=742.0,
    latitude=1.3521,
    longitude=103.8198,
    year=2024,
    month=4,
    day=15,
    hour=14,
    minute=30,
    gmt_offset_hours=8,
    averaging_minutes=60,
    p_atm=100840,
    wind_height=10,
    urban=1,
    vertical_temperature_difference=-0.4,
)
FAILURE = dict(
    tdb=60,
    rh=100,
    v=0.129,
    sol_radiation_global=1000,
    latitude=0,
    longitude=0,
    year=2024,
    month=3,
    day=20,
    hour=12,
    p_atm=30000,
)


@pytest.fixture
def backend():
    return pytest.importorskip("lwbgt")


@pytest.fixture
def fake_backend(monkeypatch):
    """Capture adapter records without requiring the optional package."""
    batches = []

    def calculate(records):
        batch = list(records)
        batches.append(batch)
        return [
            SimpleNamespace(
                wbgt_c=30.123,
                globe_temperature_c=40.456,
                natural_wet_bulb_c=27.789,
                psychrometric_wet_bulb_c=26.123,
                estimated_wind_speed_m_s=2.12345,
                status=0,
            )
            for _ in batch
        ]

    module = SimpleNamespace(
        Input=SimpleNamespace, calculate_batch=Mock(side_effect=calculate)
    )
    monkeypatch.setitem(sys.modules, "lwbgt", module)
    return module, batches


def test_reference_scalar(backend):
    result = wbgt_liljegren(**SINGAPORE, round_output=False)
    expected = dict(
        wbgt=32.50229263305664,
        tg=45.79418182373047,
        twb=28.7620792388916,
        tpsy=26.952478408813477,
        v_2m=2.199441909790039,
        status=0,
    )
    for name, value in expected.items():
        assert is_equal(result[name], value, 1e-4)
        assert np.isscalar(result[name])
    assert isinstance(result, WBGTLiljegren)
    with pytest.raises(FrozenInstanceError):
        result.wbgt = 0
    assert "WBGTLiljegren" in str(result)


def test_reference_list_day_and_night(backend):
    result = wbgt_liljegren(
        **(
            SINGAPORE
            | dict(
                tdb=[32.1, 27],
                rh=[68, 88],
                v=[2.8, 1.1],
                sol_radiation_global=[742, 0],
                hour=[14, 2],
                p_atm=[100840, 100970],
                vertical_temperature_difference=[-0.4, 0.2],
            )
        ),
        round_output=False,
    )
    assert is_equal(result.wbgt, [32.50229263305664, 25.700485229492188], 1e-4)
    assert is_equal(result.v_2m, [2.199441909790039, 0.678737223148346], 1e-5)
    assert is_equal(result.status, [0, 0], 0)


def test_documented_examples(backend):
    site = {
        name: SINGAPORE[name]
        for name in (
            "latitude",
            "longitude",
            "year",
            "month",
            "day",
            "minute",
            "gmt_offset_hours",
            "averaging_minutes",
            "wind_height",
            "urban",
        )
    }
    scalar = wbgt_liljegren(
        32.1,
        68,
        2.8,
        742,
        hour=14,
        p_atm=100840,
        vertical_temperature_difference=-0.4,
        **site,
    )
    array = wbgt_liljegren(
        [32.1, 27],
        [68, 88],
        [2.8, 1.1],
        [742, 0],
        hour=[14, 2],
        p_atm=[100840, 100970],
        vertical_temperature_difference=[-0.4, 0.2],
        **site,
    )
    assert is_equal(scalar.wbgt, 32.5, 0)
    assert is_equal(array.wbgt, [32.5, 25.7], 0)


@pytest.mark.filterwarnings("error")
def test_native_failure_preserves_valid_partial_results(backend, caplog):
    with caplog.at_level(logging.INFO, logger="pythermalcomfort.models.wbgt_liljegren"):
        result = wbgt_liljegren(
            **FAILURE,
            minute=0,
            gmt_offset_hours=0,
            averaging_minutes=0,
            wind_height=2,
            round_output=False,
        )
    assert not caplog.records
    assert is_equal(result.status, -1, 0)
    assert np.isnan([result.wbgt, result.tg, result.twb]).all()
    assert is_equal(result.tpsy, 60, 1e-4)
    assert is_equal(result.v_2m, 0.129, 1e-6)


def test_two_metre_wind_and_wbgt_consistency(backend):
    inputs = SINGAPORE | dict(
        wind_height=2, urban=None, vertical_temperature_difference=None
    )
    result = wbgt_liljegren(**inputs, round_output=False)
    assert is_equal(result.v_2m, inputs["v"], 1e-6)
    assert is_equal(result.wbgt, 32.10398483276367, 1e-4)
    measured = wbgt(
        result.twb, result.tg, inputs["tdb"], with_solar_load=True, round_output=False
    )
    assert is_equal(result.wbgt, measured.wbgt, 1e-4)


def test_utc_equivalence(backend):
    local = wbgt_liljegren(**SINGAPORE, round_output=False)
    utc = wbgt_liljegren(
        **(SINGAPORE | dict(hour=6, gmt_offset_hours=0)), round_output=False
    )
    for field in fields(local):
        assert is_equal(local[field.name], utc[field.name], 1e-5)


def test_averaging_interval_is_centered_before_timestamp(backend):
    averaged = wbgt_liljegren(**SINGAPORE, round_output=False)
    midpoint = wbgt_liljegren(
        **(SINGAPORE | dict(minute=0, averaging_minutes=0)), round_output=False
    )
    for field in fields(averaged):
        assert is_equal(averaged[field.name], midpoint[field.name], 1e-5)


@pytest.mark.parametrize(
    "name,value",
    [
        ("year", 1950),
        ("year", 2049),
        ("rh", 100),
        ("v", 0),
        ("sol_radiation_global", 0),
        ("latitude", -90),
        ("latitude", 90),
        ("longitude", -180),
        ("longitude", 180),
        ("hour", 0),
        ("hour", 23),
        ("minute", 0),
        ("minute", 59),
        ("gmt_offset_hours", -12),
        ("gmt_offset_hours", 14),
        ("averaging_minutes", 0),
        ("averaging_minutes", 1440),
    ],
)
def test_native_boundaries(backend, name, value):
    # Convergence is weather-dependent; boundary inputs must remain safe to evaluate.
    result = wbgt_liljegren(**(SINGAPORE | {name: value}))
    assert result.status in (0, -1, -2)
    assert all(not np.isinf(result[field.name]) for field in fields(result))


def test_zero_humidity_reports_solver_failure(backend):
    result = wbgt_liljegren(**(SINGAPORE | dict(rh=0)))
    assert np.isnan(result.wbgt)
    assert is_equal(result.status, -1, 0)


def test_default_pressure_is_1013_25_hpa(fake_backend):
    _, batches = fake_backend
    inputs = {name: value for name, value in SINGAPORE.items() if name != "p_atm"}
    assert is_equal(WBGTLiljegrenInputs(**inputs).p_atm, 101325, 0)
    wbgt_liljegren(**inputs)
    assert is_equal(batches[0][0].pressure_hpa, 1013.25, 0)


@pytest.mark.parametrize("explicit_none", [False, True])
@pytest.mark.parametrize(
    "name,message",
    [
        ("minute", "minute=0 min"),
        ("p_atm", "p_atm=101325 Pa (1013.25 hPa)"),
        ("gmt_offset_hours", "gmt_offset_hours=0 h (UTC)"),
        ("averaging_minutes", "averaging_minutes=0 min (instantaneous)"),
        ("wind_height", "wind_height=2 m"),
    ],
)
def test_assumed_input_is_logged(fake_backend, caplog, explicit_none, name, message):
    inputs = SINGAPORE | dict(wind_height=2)
    if explicit_none:
        inputs[name] = None
    else:
        del inputs[name]
    with caplog.at_level(logging.INFO, logger="pythermalcomfort.models.wbgt_liljegren"):
        wbgt_liljegren(**inputs)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO
    assert (
        caplog.records[0].getMessage() == f"wbgt_liljegren assumed defaults: {message}"
    )


def test_defaults_logged_once_for_a_batch(fake_backend, caplog):
    inputs = {
        name: value
        for name, value in SINGAPORE.items()
        if name
        not in (
            "minute",
            "p_atm",
            "gmt_offset_hours",
            "averaging_minutes",
            "wind_height",
            "urban",
            "vertical_temperature_difference",
        )
    }
    with caplog.at_level(logging.INFO, logger="pythermalcomfort.models.wbgt_liljegren"):
        wbgt_liljegren(**(inputs | dict(tdb=[30, 32, 34])))
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO
    message = caplog.records[0].getMessage()
    assert "p_atm=101325 Pa (1013.25 hPa)" in message
    assert "wind_height=2 m" in message
    assert "gmt_offset_hours=0 h (UTC)" in message
    assert "minute=0 min" in message
    assert "averaging_minutes=0 min (instantaneous)" in message
    assert "urban" not in message
    assert "vertical_temperature_difference" not in message
    assert "round_output" not in message


@pytest.mark.parametrize("omit_settings", [False, True])
def test_unused_wind_settings_are_quiet_at_two_metres(
    fake_backend, caplog, omit_settings
):
    inputs = SINGAPORE | dict(wind_height=2)
    if omit_settings:
        del inputs["urban"]
        del inputs["vertical_temperature_difference"]
    with caplog.at_level(logging.INFO, logger="pythermalcomfort.models.wbgt_liljegren"):
        wbgt_liljegren(**inputs)
    assert not caplog.records


@pytest.mark.parametrize("omit_pressure", [False, True])
def test_active_wind_adjustment_logs_once_for_evaluated_records(
    fake_backend, caplog, omit_pressure
):
    inputs = SINGAPORE | dict(tdb=[30, 32, np.nan], wind_height=[2, 10, 10])
    if omit_pressure:
        del inputs["p_atm"]
    with caplog.at_level(logging.INFO, logger="pythermalcomfort.models.wbgt_liljegren"):
        wbgt_liljegren(**inputs)
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO
    message = caplog.records[0].getMessage()
    assert (
        "wind-height adjustment uses urban (and vertical_temperature_difference "
        "at night) "
        "for 1 of 2 evaluated records (wind_height != 2 m)"
    ) in message
    assert ("p_atm=101325 Pa (1013.25 hPa)" in message) == omit_pressure


@pytest.mark.parametrize("temperature_difference", [None, -0.4])
def test_wind_adjustment_not_logged_for_skipped_records(
    fake_backend, caplog, temperature_difference
):
    with caplog.at_level(logging.INFO, logger="pythermalcomfort.models.wbgt_liljegren"):
        wbgt_liljegren(
            **(
                SINGAPORE
                | dict(
                    tdb=[30, np.nan],
                    wind_height=[2, 10],
                    vertical_temperature_difference=temperature_difference,
                )
            )
        )
    assert not caplog.records


@pytest.mark.parametrize("explicit_none", [False, True])
def test_default_temperature_difference_for_wind_adjustment(
    fake_backend, caplog, explicit_none
):
    _, batches = fake_backend
    inputs = SINGAPORE | dict(tdb=[30, 32, np.nan], wind_height=[2, 10, 10])
    if explicit_none:
        inputs["vertical_temperature_difference"] = None
    else:
        del inputs["vertical_temperature_difference"]
    normalized = WBGTLiljegrenInputs(**inputs)._arrays
    assert is_equal(
        normalized["vertical_temperature_difference"], [0, -0.052, -0.052], 0
    )
    with caplog.at_level(logging.INFO, logger="pythermalcomfort.models.wbgt_liljegren"):
        wbgt_liljegren(**inputs)
    assert is_equal(
        [record.vertical_temperature_difference_c for record in batches[0]],
        [0, -0.052],
        0,
    )
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.INFO
    message = caplog.records[0].getMessage()
    assert "vertical_temperature_difference=-0.052 °C" in message
    assert "for 1 of 2 evaluated records" in message


def test_assumptions_and_wind_adjustment_do_not_log_warnings(fake_backend, caplog):
    inputs = {name: value for name, value in SINGAPORE.items() if name != "p_atm"}
    with caplog.at_level(
        logging.WARNING, logger="pythermalcomfort.models.wbgt_liljegren"
    ):
        wbgt_liljegren(**(inputs | dict(wind_height=2)))
        wbgt_liljegren(**inputs)
    assert not caplog.records


def test_default_temperature_difference_matches_explicit_nighttime_result(backend):
    inputs = SINGAPORE | dict(hour=2, v=1.1, sol_radiation_global=0)
    del inputs["vertical_temperature_difference"]
    implicit = wbgt_liljegren(**inputs, round_output=False)
    explicit = wbgt_liljegren(
        **inputs, vertical_temperature_difference=-0.052, round_output=False
    )
    assert is_equal(implicit.status, 0, 0)
    for field in fields(implicit):
        assert is_equal(implicit[field.name], explicit[field.name], 0)


def test_explicit_defaults_are_quiet_and_match_native_results(backend, caplog):
    inputs = dict(
        tdb=[30, 32],
        rh=60,
        v=2,
        sol_radiation_global=700,
        latitude=0,
        longitude=0,
        year=2024,
        month=3,
        day=20,
        hour=12,
    )
    implicit = wbgt_liljegren(**inputs, round_output=False)
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="pythermalcomfort.models.wbgt_liljegren"):
        explicit = wbgt_liljegren(
            **inputs,
            minute=0,
            p_atm=101325,
            gmt_offset_hours=0,
            averaging_minutes=0,
            wind_height=2,
            urban=0,
            vertical_temperature_difference=0,
            round_output=False,
        )
    assert not caplog.records
    for field in fields(implicit):
        assert is_equal(implicit[field.name], explicit[field.name], 0)


def test_defaults_not_logged_for_invalid_or_missing_records(fake_backend, caplog):
    inputs = {name: value for name, value in SINGAPORE.items() if name != "p_atm"}
    with caplog.at_level(logging.INFO, logger="pythermalcomfort.models.wbgt_liljegren"):
        with pytest.raises(ValueError, match="rh"):
            wbgt_liljegren(**(inputs | dict(rh=-1)))
        wbgt_liljegren(**(inputs | dict(tdb=[np.nan, np.nan])))
    assert not caplog.records


def test_broadcast_mapping_and_input_immutability(fake_backend):
    module, batches = fake_backend
    temperatures = np.array([[30.0], [32.0]])
    humidity = np.array([50, 60, 70])
    result = wbgt_liljegren(
        **(SINGAPORE | dict(tdb=temperatures, rh=humidity)), round_output=False
    )
    module.calculate_batch.assert_called_once()
    assert len(batches[0]) == 6
    assert is_equal(
        [r.air_temperature_c for r in batches[0]], [30, 30, 30, 32, 32, 32], 0
    )
    assert is_equal(
        [r.relative_humidity_percent for r in batches[0]], [50, 60, 70, 50, 60, 70], 0
    )
    record = batches[0][0]
    assert is_equal(record.pressure_hpa, 1008.4, 1e-12)
    assert is_equal(record.latitude_deg_north, 1.3521, 0)
    assert is_equal(record.longitude_deg_east, 103.8198, 0)
    assert is_equal(record.solar_w_m2, 742, 0)
    assert is_equal(record.wind_height_m, 10, 0)
    assert is_equal(record.wind_speed_m_s, 2.8, 0)
    assert is_equal(record.vertical_temperature_difference_c, -0.4, 0)
    for name, value in dict(
        year=2024,
        month=4,
        day=15,
        hour=14,
        minute=30,
        gmt_offset_hours=8,
        averaging_minutes=60,
        urban=1,
    ).items():
        assert getattr(record, name) == value
        assert type(getattr(record, name)) is int
    for field in fields(result):
        assert result[field.name].shape == (2, 3)
    assert is_equal(temperatures, [[30], [32]], 0)
    assert is_equal(humidity, [50, 60, 70], 0)


def test_rounding_leaves_wind_and_status_untouched(fake_backend):
    rounded = wbgt_liljegren(**SINGAPORE)
    raw = wbgt_liljegren(**SINGAPORE, round_output=False)
    for name in ("wbgt", "tg", "twb", "tpsy"):
        assert is_equal(rounded[name], np.round(raw[name], 1), 0)
    assert is_equal(rounded.v_2m, raw.v_2m, 0)
    assert is_equal(rounded.status, raw.status, 0)


@pytest.mark.parametrize("name", WBGTLiljegrenInputs._numeric_fields)
def test_missing_records_never_reach_backend(fake_backend, name):
    _, batches = fake_backend
    result = wbgt_liljegren(**(SINGAPORE | {name: [SINGAPORE[name], np.nan]}))
    assert len(batches[0]) == 1
    for field in fields(result):
        assert np.isfinite(result[field.name][0])
        assert np.isnan(result[field.name][1])


@pytest.mark.parametrize(
    "temperature", [np.nan, [np.nan, np.nan], [], np.empty((0, 3))]
)
def test_no_valid_records_skips_native_call(fake_backend, temperature):
    module, _ = fake_backend
    result = wbgt_liljegren(**(SINGAPORE | dict(tdb=temperature)))
    module.calculate_batch.assert_not_called()
    for field in fields(result):
        assert np.shape(result[field.name]) == np.shape(temperature)
        assert np.isnan(result[field.name]).all()


@pytest.mark.filterwarnings("error")
def test_partial_failure_and_nonfinite_outputs(fake_backend, caplog):
    module, _ = fake_backend
    good = dict(
        wbgt_c=30.1,
        globe_temperature_c=40,
        natural_wet_bulb_c=27,
        psychrometric_wet_bulb_c=26,
        estimated_wind_speed_m_s=2,
        status=0,
    )
    module.calculate_batch.side_effect = None
    module.calculate_batch.return_value = [
        SimpleNamespace(**good),
        SimpleNamespace(**(good | dict(psychrometric_wet_bulb_c=-9999))),
        SimpleNamespace(
            **(good | dict(wbgt_c=-9999, globe_temperature_c=np.inf, status=-1))
        ),
    ]
    with caplog.at_level(logging.INFO, logger="pythermalcomfort.models.wbgt_liljegren"):
        result = wbgt_liljegren(**(SINGAPORE | dict(tdb=[32, 32, 32], wind_height=2)))
    assert not caplog.records
    assert is_equal(result.status, [0, -2, -1], 0)
    assert np.isnan(result.tpsy[1])
    assert np.isnan(result.wbgt[2])
    assert np.isnan(result.tg[2])
    assert is_equal(result.wbgt[:2], [30.1, 30.1], 0)
    assert is_equal(result.twb, [27, 27, 27], 0)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("invalid", [-9999, np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    "native_name,field_name",
    [
        ("wbgt_c", "wbgt"),
        ("globe_temperature_c", "tg"),
        ("natural_wet_bulb_c", "twb"),
        ("psychrometric_wet_bulb_c", "tpsy"),
        ("estimated_wind_speed_m_s", "v_2m"),
    ],
)
def test_invalid_component_cannot_report_success(
    fake_backend, caplog, native_name, field_name, invalid
):
    module, _ = fake_backend
    native_result = SimpleNamespace(
        wbgt_c=30,
        globe_temperature_c=40,
        natural_wet_bulb_c=27,
        psychrometric_wet_bulb_c=26,
        estimated_wind_speed_m_s=2,
        status=0,
    )
    setattr(native_result, native_name, invalid)
    module.calculate_batch.side_effect = None
    module.calculate_batch.return_value = [native_result]
    with caplog.at_level(logging.INFO, logger="pythermalcomfort.models.wbgt_liljegren"):
        result = wbgt_liljegren(**(SINGAPORE | dict(wind_height=2)))
    assert not caplog.records
    assert is_equal(result.status, -2, 0)
    assert np.isnan(result[field_name])
    for field in fields(result):
        if field.name != field_name:
            assert np.isfinite(result[field.name])


@pytest.mark.parametrize("native_status", [-1, -3, np.nan, np.inf, -np.inf])
def test_failure_status_always_invalidates_wbgt(fake_backend, native_status):
    module, _ = fake_backend
    module.calculate_batch.side_effect = None
    module.calculate_batch.return_value = [
        SimpleNamespace(
            wbgt_c=30,
            globe_temperature_c=40,
            natural_wet_bulb_c=27,
            psychrometric_wet_bulb_c=26,
            estimated_wind_speed_m_s=2,
            status=native_status,
        )
    ]
    result = wbgt_liljegren(**SINGAPORE)
    expected_status = native_status if np.isfinite(native_status) else -2
    assert is_equal(result.status, expected_status, 0)
    assert np.isnan(result.wbgt)
    assert is_equal(result.tg, 40, 0)


@pytest.mark.parametrize(
    "name,value",
    [
        ("rh", -1),
        ("rh", 101),
        ("v", -1),
        ("sol_radiation_global", -1),
        ("p_atm", 0),
        ("wind_height", 0),
        ("tdb", -273.15),
        ("latitude", -91),
        ("latitude", 91),
        ("longitude", -181),
        ("longitude", 181),
        ("year", 1949),
        ("year", 2050),
        ("month", 0),
        ("month", 13),
        ("day", 0),
        ("day", 32),
        ("hour", -1),
        ("hour", 24),
        ("minute", -1),
        ("minute", 60),
        ("gmt_offset_hours", -13),
        ("gmt_offset_hours", 15),
        ("averaging_minutes", -1),
        ("averaging_minutes", 1441),
        ("urban", -1),
        ("urban", 2),
    ],
)
def test_invalid_ranges_rejected_before_native_call(fake_backend, name, value):
    module, _ = fake_backend
    with pytest.raises(ValueError, match=name):
        wbgt_liljegren(**(SINGAPORE | {name: value}))
    module.calculate_batch.assert_not_called()


@pytest.mark.parametrize("name", WBGTLiljegrenInputs._numeric_fields)
@pytest.mark.parametrize("value", [np.inf, -np.inf, 1e100])
def test_nonfinite_and_overflow_inputs(fake_backend, name, value):
    module, _ = fake_backend
    with pytest.raises(ValueError, match=name):
        wbgt_liljegren(**(SINGAPORE | {name: value}))
    module.calculate_batch.assert_not_called()


@pytest.mark.parametrize("name", WBGTLiljegrenInputs._integer_fields)
def test_integer_fields_are_not_truncated(fake_backend, name):
    with pytest.raises(ValueError, match="whole numbers"):
        wbgt_liljegren(**(SINGAPORE | {name: SINGAPORE[name] + 0.5}))


@pytest.mark.skipif(
    np.finfo(np.longdouble).max <= np.finfo(float).max,
    reason="longdouble has no wider range than float64 on this platform",
)
def test_extended_precision_overflow_is_rejected_without_warning(fake_backend):
    temperature = np.array([np.longdouble("1e400")])
    with np.errstate(over="raise"), pytest.raises(ValueError, match="tdb"):
        wbgt_liljegren(**(SINGAPORE | dict(tdb=temperature)))


@pytest.mark.parametrize(
    "value", ["32", ["32"], np.array(["32"]), [None], 32 + 1j, True, None]
)
def test_invalid_numeric_elements(fake_backend, value):
    with pytest.raises(TypeError, match="tdb"):
        wbgt_liljegren(**(SINGAPORE | dict(tdb=value)))


@pytest.mark.parametrize("value", [0, 1, None, "True", [True]])
def test_invalid_round_output(fake_backend, value):
    with pytest.raises(TypeError, match="round_output"):
        wbgt_liljegren(**SINGAPORE, round_output=value)


@pytest.mark.parametrize(
    "date", [dict(year=2023, month=2, day=29), dict(month=4, day=31)]
)
def test_invalid_calendar_date(fake_backend, date):
    with pytest.raises(ValueError, match="calendar date"):
        wbgt_liljegren(**(SINGAPORE | date))


def test_leap_day_and_numpy_integer_fields(fake_backend):
    _, batches = fake_backend
    wbgt_liljegren(**(SINGAPORE | dict(year=np.int64(2024), month=2, day=np.array(29))))
    assert batches[0][0].day == 29


def test_averaging_midpoint_outside_native_window(fake_backend):
    with pytest.raises(ValueError, match="UTC averaging midpoint"):
        wbgt_liljegren(
            **(
                SINGAPORE
                | dict(
                    day=1, hour=0, minute=0, gmt_offset_hours=14, averaging_minutes=1440
                )
            )
        )


def test_mismatched_shapes(fake_backend):
    with pytest.raises(ValueError, match="broadcast"):
        wbgt_liljegren(**(SINGAPORE | dict(tdb=[30, 31], rh=[50, 60, 70])))


def test_wind_conversion_requires_explicit_urban(fake_backend):
    with pytest.raises(ValueError, match="urban"):
        wbgt_liljegren(
            **(SINGAPORE | dict(urban=None, vertical_temperature_difference=None))
        )


def test_default_two_metre_settings(fake_backend):
    _, batches = fake_backend
    wbgt_liljegren(**FAILURE)
    record = batches[0][0]
    assert record.urban == 0
    assert is_equal(record.wind_height_m, 2, 0)
    assert is_equal(record.vertical_temperature_difference_c, 0, 0)


def test_native_loader_errors_are_not_hidden(fake_backend):
    module, _ = fake_backend
    module.calculate_batch.side_effect = RuntimeError(
        "cannot load bundled lwbgt runtime"
    )
    with pytest.raises(RuntimeError, match="cannot load"):
        wbgt_liljegren(**SINGAPORE)


def test_import_and_wbgt_without_optional_dependency():
    # A fresh interpreter catches accidental eager imports even when lwbgt is installed.
    code = """
import importlib.abc
import sys

class BlockLwbgt(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'lwbgt' or fullname.startswith('lwbgt.'):
            raise ModuleNotFoundError('blocked for test', name='lwbgt')

sys.meta_path.insert(0, BlockLwbgt())
from pythermalcomfort.models import wbgt, wbgt_liljegren
assert 'lwbgt' not in sys.modules
assert abs(wbgt(25, 32).wbgt - 27.1) < 1e-10
try:
    wbgt_liljegren(30, 50, 2, 500, latitude=0, longitude=0,
                   year=2024, month=3, day=20, hour=12)
except ImportError as exc:
    assert "pip install 'pythermalcomfort[lwbgt]'" in str(exc)
else:
    raise AssertionError('missing backend did not raise ImportError')
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
