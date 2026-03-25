import pytest
from range_scene import *

class MockResult:
    def __init__(self):
        self.metric1 = np.array([[1, 2], [3, 4]])
        self.metric2 = np.array([[5, 6], [7, 8]])

def mock_model(x=0, y=0, v=0, fixed_param=0):
    return MockResult()

def mock_model2(x=0, y=0, vr=0, fixed_param=0):
    return MockResult()

# ----------------------
# Axis test
# ----------------------
def test_x_y_axis_valid():
    scene = RangeScene(mock_model)
    # Normal setup
    scene.x(x=(0, 1)).y(y=(0, 2))
    assert scene.x_var == "x"
    assert scene.x_range == (0.0, 1.0)
    assert scene.y_var == "y"
    assert scene.y_range == (0.0, 2.0)

def test_x_axis_invalid_param():
    scene = RangeScene(mock_model)
    with pytest.raises(ValueError):
        scene.x(z=(0, 1))  # z does not exist

def test_y_axis_invalid_range_length():
    scene = RangeScene(mock_model)
    with pytest.raises(ValueError):
        scene.y(y=(0, 1, 2))  # length is not 2

def test_x_axis_non_numeric():
    scene = RangeScene(mock_model)
    with pytest.raises(ValueError):
        scene.x(x=("a", "b"))

def test_axis_min_ge_max():
    scene = RangeScene(mock_model)
    with pytest.raises(ValueError):
        scene.x(x=(5, 1))  # min >= max

def test_axis_multiple_parameters():
    scene = RangeScene(mock_model)
    with pytest.raises(ValueError):
        scene.x(x=(0, 1), y=(0, 1))

# ----------------------
# Fixed test
# ----------------------
def test_fixed_valid_scalar():
    scene = RangeScene(mock_model)
    scene.x(x=(0, 1)).y(y=(0, 1)).fixed(fixed_param=5)
    assert scene.constant_params["fixed_param"] == 5

def test_fixed_invalid_param():
    scene = RangeScene(mock_model)
    scene.x(x=(0, 1)).y(y=(0, 1))
    with pytest.raises(ValueError):
        scene.fixed(z=1)

def test_fixed_axis_conflict():
    scene = RangeScene(mock_model)
    scene.x(x=(0, 1)).y(y=(0, 1))
    with pytest.raises(ValueError):
        scene.fixed(x=5)

def test_fixed_non_scalar():
    scene = RangeScene(mock_model)
    scene.x(x=(0, 1)).y(y=(0, 1))
    with pytest.raises(TypeError):
        scene.fixed(fixed_param=[1, 2])

# ----------------------
# v/vr alias test
# ----------------------
def test_handle_wind_alias_convert_vr_to_v():
    scene = RangeScene(mock_model)
    scene.param_names = ["v"]  # model only accepts v
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = scene._handle_wind_alias({"vr": 8})
        assert "v" in result and result["v"] == 8
        assert "vr" not in result
        assert any("Automatically converting 'vr' -> 'v'" in str(warn.message) for warn in w)


def test_handle_wind_alias_convert_v_to_vr():
    scene = RangeScene(mock_model2)
    scene.param_names = ["vr"]  # model only accepts vr
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = scene._handle_wind_alias({"v": 12})
        assert "vr" in result and result["vr"] == 12
        assert "v" not in result
        assert any("Automatically converting 'v' -> 'vr'" in str(warn.message) for warn in w)

def test_handle_wind_alias_both_provided():
    scene = RangeScene(mock_model)
    with pytest.raises(ValueError):
        scene._handle_wind_alias({"v": 1, "vr": 2})

# ----------------------
# Copy test
# ----------------------
def test_copy_independent():
    scene = RangeScene(mock_model)
    scene.x(x=(0, 1)).y(y=(0, 1)).fixed(fixed_param=5)
    copy_scene = scene._copy()
    # copy does not affect original object
    copy_scene.constant_params["fixed_param"] = 10
    assert scene.constant_params["fixed_param"] == 5

def test_with_field_and_with_lines():
    scene = RangeScene(mock_model).x(x=(0, 1)).y(y=(0, 1))
    field_scene = scene.with_field(cmap="viridis")
    line_scene = scene.with_lines(colors="red")
    assert field_scene._field_style["cmap"] == "viridis"
    assert "cmap" not in scene._field_style
    assert line_scene._line_style["colors"] == "red"
    assert "colors" not in scene._line_style

# ----------------------
# Plot test
# ----------------------
def test_plot_basic():
    scene = RangeScene(mock_model).x(x=(0, 1)).y(y=(0, 1))
    fig, ax = scene.plot()
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)

def test_plot_with_metric_and_levels():
    scene = RangeScene(mock_model).x(x=(0, 1)).y(y=(0, 1))
    fig, ax = scene.plot(metric="metric2", levels=[1, 5, 10], resolution=2)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

def test_plot_requires_axes():
    scene = RangeScene(mock_model)
    with pytest.raises(ValueError):
        scene.plot()

# ----------------------
# Method chaining test
# ----------------------
def test_chainable_calls():
    scene = RangeScene(mock_model)
    new_scene = scene.x(x=(0, 1)).y(y=(0, 1)).fixed(fixed_param=5).with_field(cmap="coolwarm")
    assert isinstance(new_scene, RangeScene)
    assert new_scene._field_style["cmap"] == "coolwarm"
    assert new_scene.constant_params["fixed_param"] == 5

# ----------------------
# utci/pmv_ppd_iso plotting test
# ----------------------
def test_plot_utci_pmvp_iso():
    from pythermalcomfort.models import utci, pmv_ppd_iso
    import matplotlib.pyplot as plt

    # utci
    scene1 = RangeScene(utci).x(tdb=(0,40)).y(rh=(10,90)).fixed(tr=25, v=1)
    fig1, ax1 = scene1.with_field(cmap="coolwarm").with_lines(colors="black").plot(levels=[-20,-10,0,10,20,30,40], fill=True)
    assert isinstance(fig1, plt.Figure)
    plt.show()

    # pmv_ppd_iso
    scene2 = RangeScene(pmv_ppd_iso).x(tdb=(18,28)).y(rh=(30,70)).fixed(tr=25, v=0.1, met=1, clo=1)
    fig2, ax2 = scene2.with_field(cmap="YlOrRd").with_lines(colors="black").plot(metric="ppd", levels=[0,5,10,20,30,40,100], fill=True)
    assert isinstance(fig2, plt.Figure)
    plt.show()
