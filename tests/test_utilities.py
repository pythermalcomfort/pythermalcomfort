import importlib
import inspect

import numpy as np
import pytest

import pythermalcomfort.utilities as utilities
from pythermalcomfort.utilities import Units, body_surface_area, units_converter


def test_ip_units_converter() -> None:
    """Test the units converter for IP and SI units."""
    assert (units_converter(tdb=77, tr=77, v=3.2, from_units=Units.IP.value)) == [
        25.0,
        25.0,
        0.975312404754648,
    ]
    assert (units_converter(pressure=1, area=1 / 0.09, from_units=Units.IP.value)) == [
        101325,
        1.0322474090590033,
    ]

    expected_result = [25.0, 3.047]
    assert np.allclose(
        units_converter(Units.IP.value, tdb=77, v=10),
        expected_result,
        atol=0.01,
    )

    # Test case 2: Conversion from SI to IP for temperature and velocity
    expected_result = [68, 6.562]
    assert np.allclose(
        units_converter(Units.SI.value, tdb=20, v=2),
        expected_result,
        atol=0.01,
    )

    # Test case 3: Conversion from IP to SI for area and pressure
    expected_result = [9.29, 1489477.5]
    assert np.allclose(
        units_converter(Units.IP.value, area=100, pressure=14.7),
        expected_result,
        atol=0.01,
    )

    # Test case 4: Conversion from SI to IP for area and pressure
    expected_result = [538.199, 1]
    assert np.allclose(
        units_converter(Units.SI.value, area=50, pressure=101325),
        expected_result,
        atol=0.01,
    )


def test_body_surface_area() -> None:
    """Test the body surface area calculations with various formulas."""
    assert body_surface_area(weight=80, height=1.8) == pytest.approx(1.9917, rel=1e-2)
    assert body_surface_area(70, 1.8, "dubois") == pytest.approx(1.88, rel=1e-2)
    assert body_surface_area(75, 1.75, "takahira") == pytest.approx(1.91, rel=1e-2)
    assert body_surface_area(80, 1.7, "fujimoto") == pytest.approx(1.872, rel=1e-2)
    assert body_surface_area(85, 1.65, "kurazumi") == pytest.approx(1.89, rel=1e-2)
    with pytest.raises(ValueError):
        body_surface_area(70, 1.8, "invalid_formula")


MOVED_PUBLIC_FUNCTION_CASES = [
    pytest.param(
        "mean_radiant_tmp",
        "environment",
        ([53.2, 55, 55], 30, [0.3, 0.3, 0.1]),
        {"d": 0.1, "standard": "ISO"},
        [74.8, 77.8, 71.9],
        0.1,
        id="mean_radiant_tmp",
    ),
    pytest.param(
        "operative_tmp",
        "environment",
        ([25, 20], 30, 0.3),
        {},
        [26.83, 23.66],
        0.01,
        id="operative_tmp",
    ),
    pytest.param(
        "running_mean_outdoor_temperature",
        "environment",
        ([20, 21, 22],),
        {},
        20.9,
        1e-8,
        id="running_mean_outdoor_temperature",
    ),
    pytest.param(
        "transpose_sharp_altitude",
        "environment",
        (120, 75),
        {},
        [13.064, 7.435],
        1e-3,
        id="transpose_sharp_altitude",
    ),
    pytest.param(
        "f_svv",
        "environment",
        (30, 10, 3.3),
        {},
        0.2709762313,
        1e-8,
        id="f_svv",
    ),
    pytest.param(
        "v_relative",
        "environment",
        ([1, 2], 2),
        {},
        [1.3, 2.3],
        1e-8,
        id="v_relative",
    ),
    pytest.param(
        "p_sat",
        "psychrometrics",
        (25,),
        {},
        3169.21647014,
        1e-5,
        id="p_sat",
    ),
    pytest.param(
        "p_sat_torr",
        "psychrometrics",
        (25,),
        {},
        23.75744972,
        1e-8,
        id="p_sat_torr",
    ),
    pytest.param(
        "antoine",
        "psychrometrics",
        (25,),
        {},
        3.16735278,
        1e-8,
        id="antoine",
    ),
    pytest.param(
        "psy_ta_rh",
        "psychrometrics",
        (25, 50),
        {},
        [
            3169.21647014,
            1584.60823507,
            0.0098816,
            17.99814747,
            13.8515836,
            50259.78815634,
        ],
        1e-5,
        id="psy_ta_rh",
    ),
    pytest.param(
        "hr_to_rh",
        "psychrometrics",
        (0.01, 25),
        {},
        50.58961491,
        1e-8,
        id="hr_to_rh",
    ),
    pytest.param(
        "wet_bulb_tmp",
        "psychrometrics",
        (25, 50),
        {},
        17.99814747,
        1e-8,
        id="wet_bulb_tmp",
    ),
    pytest.param(
        "dew_point_tmp",
        "psychrometrics",
        (25, 50),
        {},
        13.8515836,
        1e-8,
        id="dew_point_tmp",
    ),
    pytest.param(
        "enthalpy_air",
        "psychrometrics",
        (25, 0.01),
        {},
        50561.25,
        1e-8,
        id="enthalpy_air",
    ),
    pytest.param(
        "clo_dynamic_ashrae",
        "clothing",
        (1, 2),
        {},
        0.8,
        1e-8,
        id="clo_dynamic_ashrae",
    ),
    pytest.param(
        "clo_dynamic_iso",
        "clothing",
        (1, 1.2, 0.2),
        {},
        0.95486298,
        1e-8,
        id="clo_dynamic_iso",
    ),
    pytest.param(
        "clo_intrinsic_insulation_ensemble",
        "clothing",
        ([0.2, 0.3],),
        {},
        0.5785,
        1e-8,
        id="clo_intrinsic_insulation_ensemble",
    ),
    pytest.param(
        "clo_area_factor",
        "clothing",
        (1,),
        {},
        1.28,
        1e-8,
        id="clo_area_factor",
    ),
    pytest.param(
        "clo_insulation_air_layer",
        "clothing",
        (0.2, 0.1, 0.7),
        {},
        0.65224016,
        1e-8,
        id="clo_insulation_air_layer",
    ),
    pytest.param(
        "clo_total_insulation",
        "clothing",
        (1.7, 0.2, 0.1, 0.7, 1),
        {},
        1.59879185,
        1e-8,
        id="clo_total_insulation",
    ),
    pytest.param(
        "clo_correction_factor_environment",
        "clothing",
        (0.2, 0.1, 1),
        {},
        0.94046579,
        1e-8,
        id="clo_correction_factor_environment",
    ),
]


@pytest.mark.parametrize(
    ("function_name", "new_module", "args", "kwargs", "expected", "atol"),
    MOVED_PUBLIC_FUNCTION_CASES,
)
def test_moved_public_utility_shims(
    function_name: str,
    new_module: str,
    args: tuple,
    kwargs: dict,
    expected,
    atol: float,
) -> None:
    """Every old public utility keeps its signature, result, and warning."""
    module = importlib.import_module(f"pythermalcomfort.{new_module}")
    target = getattr(module, function_name)
    deprecated = getattr(utilities, function_name)

    assert inspect.signature(deprecated) == inspect.signature(target)
    assert deprecated.__doc__ == (
        f"Deprecated alias for pythermalcomfort.{new_module}.{function_name}(). "
        "Import from there instead; this path will be removed after two minor releases."
    )

    with pytest.warns(DeprecationWarning, match=f"pythermalcomfort.{new_module}"):
        result = deprecated(*args, **kwargs)

    if function_name == "psy_ta_rh":
        result = [
            result.p_sat,
            result.p_vap,
            result.hr,
            result.wet_bulb_tmp,
            result.dew_point_tmp,
            result.h,
        ]

    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=atol)


def test_internal_helpers_have_no_utility_shims() -> None:
    """Private implementation helpers are available only from _internal."""
    assert not hasattr(utilities, "validate_type")
    assert not hasattr(utilities, "_check_ashrae55_compliance")
    assert not hasattr(utilities, "adaptive_cooling_effect")
