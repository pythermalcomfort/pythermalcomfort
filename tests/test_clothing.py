import numpy as np
import pytest

from pythermalcomfort.clothing import (
    clo_area_factor,
    clo_correction_factor_environment,
    clo_dynamic_ashrae,
    clo_dynamic_iso,
    clo_insulation_air_layer,
    clo_intrinsic_insulation_ensemble,
    clo_total_insulation,
)


def test_intrinsic_insulation_ensemble() -> None:
    """Test the intrinsic insulation ensemble function."""
    assert clo_intrinsic_insulation_ensemble([0.5, 0.5]) == 0.835 + 0.161
    assert clo_intrinsic_insulation_ensemble([1, 1]) == 2 * 0.835 + 0.161
    assert clo_intrinsic_insulation_ensemble(2) == 2 * 0.835 + 0.161
    assert clo_intrinsic_insulation_ensemble([0]) == 0.161


def test_clo_area_factor() -> None:
    """Test the clothing area factor function."""
    assert clo_area_factor(1) == 1.28
    assert np.allclose(clo_area_factor(i_cl=[1, 2]), np.asarray([1.28, 1.56]))


def test_clo_air_layer_insulation() -> None:
    """Test the clothing insulation air layer function."""
    assert np.isclose(
        clo_insulation_air_layer(vr=1, v_walk=1, i_a_static=0.71),
        0.365,
        atol=0.001,
    )
    assert np.isclose(
        clo_insulation_air_layer(vr=0.2, v_walk=1, i_a_static=0.71),
        0.532,
        atol=0.001,
    )
    assert np.allclose(
        clo_insulation_air_layer(vr=[0.2, 1], v_walk=1, i_a_static=0.71),
        [0.532, 0.365],
        atol=0.001,
    )


def test_clo_total_insulation() -> None:
    """Test the total clothing insulation function."""
    assert np.allclose(
        clo_total_insulation(
            i_t=[1.21, 1.26, 1.56],
            vr=0.15,
            v_walk=0,
            i_a_static=0.5,
            i_cl=[0.61, 0.71, 1.01],
        ),
        [1.21, 1.26, 1.56],
        atol=0.001,
    )

    # compare the normal_clothing results with the figure in the standard
    assert np.allclose(
        clo_total_insulation(
            i_t=[1.21, 1.26, 1.56],
            vr=2,
            v_walk=[1, 0.5, 0.25],
            i_a_static=0.5,
            i_cl=[0.61, 0.71, 1.01],
        ),
        [1.21 * 0.5, 1.26 * 0.565, 1.56 * 0.62],
        atol=0.005,
    )

    # test that the nude function works as expected
    assert np.allclose(
        clo_total_insulation(
            i_t=0,
            vr=0.15,
            v_walk=0,
            i_a_static=[0.71, 0.61, 0.5],
            i_cl=0,
        ),
        [0.71, 0.61, 0.5],
        atol=0.001,
    )

    # compare the nude results with the figure in the standard
    assert np.allclose(
        clo_total_insulation(
            i_t=0,
            vr=[0.5, 2, 3],
            v_walk=0.5,
            i_a_static=[0.71, 0.61, 0.5],
            i_cl=0,
        ),
        [0.71 * 0.7, 0.61 * 0.4, 0.50 * 0.32],
        atol=0.004,
    )

    # test that the low_clothing function works as expected
    assert np.allclose(
        clo_total_insulation(
            i_t=[1.2, 0.6],
            vr=0.15,
            v_walk=0,
            i_a_static=[0.6, 0.6],
            i_cl=[0.6, 0],
        ),
        [1.2, 0.6],
        atol=0.001,
    )

    clo = 0.3
    i_a = 0.7
    assert np.isclose(
        clo_total_insulation(
            i_t=clo + i_a,
            vr=0.26,
            v_walk=0.06,
            i_a_static=i_a,
            i_cl=clo,
        ),
        0.79,
        atol=0.01,
    )


def test_clo_correction_factor_environment() -> None:
    """Test the clothing correction factor for environment function."""
    assert np.allclose(
        clo_correction_factor_environment(
            vr=0.15,
            v_walk=0,
            i_cl=[0.61, 0.71, 1.01],
        ),
        [1, 1, 1],
        atol=0.001,
    )

    # compare the normal_clothing results with the figure in the standard
    assert np.allclose(
        clo_correction_factor_environment(
            vr=2,
            v_walk=[1, 0.5, 0.25],
            i_cl=[0.61, 0.71, 1.01],
        ),
        [0.503, 0.564, 0.618],
        atol=0.001,
    )

    # test that the nude function works as expected
    assert np.allclose(
        clo_correction_factor_environment(
            vr=0.15,
            v_walk=0,
            i_cl=0,
        ),
        [1],
        atol=0.001,
    )

    # compare the nude results with the figure in the standard
    assert np.allclose(
        clo_correction_factor_environment(
            vr=[0.5, 2, 3],
            v_walk=0.5,
            i_cl=0,
        ),
        [0.698, 0.394, 0.320],
        atol=0.001,
    )

    # test that the low_clothing function works as expected
    assert np.allclose(
        clo_correction_factor_environment(
            vr=0.15,
            v_walk=0,
            i_cl=[0.6, 0],
        ),
        [1, 1],
        atol=0.001,
    )


def test_clo_dynamic_ashrae() -> None:
    """Test the dynamic clothing insulation function for ASHRAE standards."""
    assert clo_dynamic_ashrae(clo=1, met=1) == 1
    assert clo_dynamic_ashrae(clo=1, met=0.5) == 1
    assert clo_dynamic_ashrae(clo=2, met=0.5) == 2
    assert np.allclose(clo_dynamic_ashrae(1.0, 1.0), np.asarray(1))
    assert np.allclose(clo_dynamic_ashrae(1.0, 1.2), np.asarray(1))
    assert np.allclose(clo_dynamic_ashrae(1.0, 2.0), np.asarray(0.8))

    # Test invalid standard input
    with pytest.raises(ValueError):
        clo_dynamic_ashrae(1.0, 1.0, model="invalid")


def test_clo_dynamic_iso() -> None:
    """Test the dynamic clothing insulation function for ISO standards."""
    assert np.isclose(clo_dynamic_iso(clo=1, met=1, v=0.2), 0.99, atol=0.01)
    assert np.allclose(
        clo_dynamic_iso(clo=[1, 1.5], met=1, v=0.2),
        [0.99, 1.48],
        atol=0.01,
    )
    assert np.allclose(
        clo_dynamic_iso(clo=[1, 1.5], met=1, v=0.2),
        [0.99, 1.48],
        atol=0.01,
    )
    assert np.allclose(
        clo_dynamic_iso(
            clo=[0.95, 1.07, 0.88, 0.59, 0.83, 0.66, 1.02, 0.71, 1.1, 0.68, 0.3],
            met=[1.71, 1.11, 1.21, 1.77, 1.48, 1.5, 1.33, 1.33, 1.26, 1.47, 1.27],
            v=[0.03, 0.08, 0.04, 0.03, 0.15, 0.15, 0.06, 0.03, 0.25, 0.05, 0.12],
        ),
        [0.85, 1.06, 0.86, 0.52, 0.76, 0.61, 0.97, 0.68, 1.03, 0.63, 0.17],
        atol=0.01,
    )

    # Test invalid standard input
    with pytest.raises(ValueError):
        clo_dynamic_iso(1.0, 1.0, v=0.2, model="invalid")
