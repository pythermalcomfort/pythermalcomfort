import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PolyCollection

from pythermalcomfort.models import pmv_ppd_iso
from pythermalcomfort.plots.matplotlib import PsychrometricPlot, ThresholdPlotResult
from pythermalcomfort.psychrometrics import hr_to_rh, psy_ta_rh


def _new_plot() -> PsychrometricPlot:
    """Initialize a basic PsychrometricPlot."""
    return (
        PsychrometricPlot(pmv_ppd_iso)
        .set_params(vr=0.1, met=1.2, clo=0.5, tr=25.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )


def test_import_export() -> None:
    """Test import/export from pythermalcomfort.plots.matplotlib."""
    try:
        from pythermalcomfort.plots.matplotlib import PsychrometricPlot

        assert PsychrometricPlot is not None
    except ImportError as exc:
        pytest.fail(f"Failed to import PsychrometricPlot: {exc}")


def test_set_x_axis_accepts_any_model_temperature_param() -> None:
    """tdb and tr are both valid x-axis parameters; unknown params are rejected."""
    # tdb is always accepted
    plot = _new_plot()
    plot.set_x_axis("tdb", 10.0, 40.0, resolution=1.0)

    # tr is also a valid model parameter when it is not already in fixed params
    plot_tr = PsychrometricPlot(pmv_ppd_iso).set_params(vr=0.1, met=1.2, clo=0.5)
    plot_tr.set_x_axis("tr", 10.0, 40.0, resolution=1.0)

    with pytest.raises(ValueError):
        plot.set_x_axis("not_a_model_param", 10.0, 40.0, resolution=1.0)


def test_set_y_axis_only_accepts_hr() -> None:
    """Test set_y_axis strictly enforces 'hr'."""
    plot = _new_plot()
    with pytest.raises(ValueError, match="requires the y-axis to be 'hr'"):
        plot.set_y_axis("rh", 0.0, 30.0, resolution=1.0)

    # Valid input should not raise
    plot.set_y_axis("hr", 0.0, 30.0, resolution=1.0)


def test_set_y_axis_warns_on_kg_per_kg_range() -> None:
    """A pre-4.5.0 kg/kg range warns rather than silently rendering a blank chart.

    It warns rather than raises because humidity ratios below 1 g/kg are
    physically real in cold air, which this package supports.
    """
    plot = _new_plot()
    with pytest.warns(UserWarning, match="g/kg dry air rather than kg/kg"):
        plot.set_y_axis("hr", 0.0, 0.03, resolution=0.002)


def test_set_y_axis_accepts_cold_climate_range() -> None:
    """A sub-1 g/kg range is accepted; at -20 degC, 0.5 g/kg is about 80 % RH."""
    plot = _new_plot()
    plot.set_y_axis("hr", 0.0, 0.5, resolution=0.05)
    assert plot._y_axis.max_val == 0.5


def test_basic_plot_renders_and_preserves_limits() -> None:
    """Test a basic plot renders, masks invalid RH, and preserves requested axis limits."""
    plot = _new_plot()
    plot.set_x_axis("tdb", 10.0, 40.0, resolution=1.0)
    plot.set_y_axis("hr", 0.0, 30.0, resolution=2.0)

    result = plot.plot()

    # Verify the return object
    assert isinstance(result, ThresholdPlotResult)
    assert result.fig is not None
    assert result.ax is not None

    # Verify requested axis limits are preserved perfectly
    xlim = result.ax.get_xlim()
    ylim = result.ax.get_ylim()
    assert xlim == (10.0, 40.0)
    assert ylim == (0.0, 30.0)

    plt.close(result.fig)


def test_y_axis_has_default_humidity_ratio_label() -> None:
    """The chart labels its own y-axis instead of falling back to the bare 'hr'."""
    plot = _new_plot()
    plot.set_x_axis("tdb", 10.0, 40.0, resolution=1.0)
    plot.set_y_axis("hr", 0.0, 30.0, resolution=2.0)

    result = plot.plot()

    ylabel = result.ax.get_ylabel()
    assert ylabel != "hr"
    assert "Humidity ratio" in ylabel
    # The units have to be stated, and stated as g/kg dry air: the whole point
    # of #338 was that callers disagreed about them.
    assert "g" in ylabel
    assert "kg" in ylabel
    assert "dry" in ylabel
    assert "kg/kg" not in ylabel.replace(" ", "")

    plt.close(result.fig)


def test_grid_is_evaluated_as_g_per_kg() -> None:
    """A y value of 10 must reach the model as 10 g/kg, i.e. 0.010 kg/kg.

    This pins the conversion itself rather than inspecting rendered artists.
    An earlier version of this test scanned ``ax.collections``, which includes
    the white saturation mask; that mask is drawn up to the axis maximum
    whatever the units, so its assertion held even when the conversion was
    wrong.
    """
    plot = _new_plot()
    plot.set_x_axis("tdb", 20.0, 30.0, resolution=5.0)
    plot.set_y_axis("hr", 0.0, 20.0, resolution=10.0)

    tdb = np.array([[25.0]])
    hr_g_kg = np.array([[10.0]])
    actual = plot._evaluate_grid_output(x=tdb, y=hr_g_kg, output_name="pmv")

    expected_rh = hr_to_rh(10.0 / 1000.0, 25.0)
    # Grid evaluation asks for unrounded output, because bisecting a rounded
    # staircase parks a boundary on the edge of a quantisation plateau.
    expected = pmv_ppd_iso(
        tdb=25.0,
        tr=25.0,
        vr=0.1,
        rh=float(expected_rh),
        met=1.2,
        clo=0.5,
        round_output=False,
    ).pmv

    assert float(actual[0, 0]) == pytest.approx(float(expected), abs=1e-9)


def test_rh_curves_and_saturation_mask_render_in_g_per_kg() -> None:
    """RH iso-lines and the saturation mask must be drawn in g/kg, not kg/kg.

    test_grid_is_evaluated_as_g_per_kg only pins the conversion used to
    evaluate the model grid. The saturation mask and RH iso-lines are drawn
    by a separate code path that also calls psy_ta_rh, and could silently
    regress to kg/kg while that test still passes.
    """
    plot = _new_plot()
    x_min = 20.0
    plot.set_x_axis("tdb", x_min, 30.0, resolution=5.0)
    plot.set_y_axis("hr", 0.0, 30.0, resolution=5.0)

    result = plot.plot()
    ax = result.ax

    expected_50pct_hr = psy_ta_rh(x_min, 50.0).hr * 1000.0
    dotted_lines = [line for line in ax.lines if line.get_linestyle() == ":"]
    rh_50_line = next(
        line
        for line in dotted_lines
        if line.get_xdata()[0] == pytest.approx(x_min)
        and line.get_ydata()[0] == pytest.approx(expected_50pct_hr, abs=1e-6)
    )
    # Would be ~0.0076 kg/kg dry air if the conversion regressed.
    assert rh_50_line.get_ydata()[0] > 1.0

    expected_saturation_hr = psy_ta_rh(x_min, 100.0).hr * 1000.0
    white_polys = [
        c
        for c in ax.collections
        if isinstance(c, PolyCollection)
        and np.allclose(np.asarray(c.get_facecolor())[0][:3], 1.0)
    ]
    assert white_polys, "expected the saturation mask to be a white PolyCollection"
    mask_y = white_polys[0].get_paths()[0].vertices[:, 1]
    assert mask_y.min() == pytest.approx(expected_saturation_hr, abs=1e-6)

    plt.close(result.fig)


def test_rh_curves_and_labels_use_25_percent_intervals() -> None:
    """RH curves and labels are limited to 25, 50, 75, and 100 percent."""
    plot = _new_plot()
    plot.set_x_axis("tdb", 10.0, 40.0, resolution=1.0)
    plot.set_y_axis("hr", 0.0, 30.0, resolution=2.0)

    result = plot.plot()

    dotted_lines = [
        line for line in result.ax.lines if line.get_linestyle() in (":", "dotted")
    ]
    rh_labels = {
        text.get_text() for text in result.ax.texts if text.get_text().endswith("%")
    }

    assert len(dotted_lines) == 4
    assert rh_labels == {"25%", "50%", "75%", "100%"}
    assert all(
        text.get_fontsize() == plt.rcParams["font.size"] for text in result.ax.texts
    )

    plt.close(result.fig)


def test_rh_labels_are_placed_on_the_visible_part_of_each_curve() -> None:
    """An elevated y window must not size the label gap off hidden samples.

    Masking only ``hr <= max_val`` left the samples below ``min_val`` steering
    the label position and the gap width, which blanked up to a third of the
    curve actually on screen.
    """
    y_min, y_max = 15.0, 30.0
    plot = (
        PsychrometricPlot(pmv_ppd_iso)
        .set_x_axis("tdb", 10.0, 36.0)
        .set_y_axis("hr", y_min, y_max)
        .set_params(vr=0.1, met=1.2, clo=0.5)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )
    result = plot.plot()

    dotted = [
        line
        for line in result.ax.get_lines()
        if line.get_linestyle() in (":", "dotted")
    ]
    assert dotted

    for line in dotted:
        y = line.get_ydata()
        drawn = np.asarray(y, dtype=float)
        finite = drawn[np.isfinite(drawn)]
        assert finite.size
        # Nothing is plotted outside the window...
        assert finite.min() >= y_min - 1e-9
        assert finite.max() <= y_max + 1e-9
        # ...and the gap cut for the label stays a small share of the curve.
        blanked = int(np.isnan(drawn).sum())
        assert blanked / drawn.size < 0.15
