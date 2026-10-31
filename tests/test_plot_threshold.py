from __future__ import annotations

import warnings
from dataclasses import dataclass
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from pythermalcomfort.models import pmv_ppd_iso
from pythermalcomfort.plots.matplotlib.threshold import (
    OUT_OF_MODEL_LIMITS_COLOR,
    ThresholdPlot,
    ThresholdPlotResult,
)


@pytest.fixture(autouse=True)
def close_all_figures():
    yield
    plt.close("all")


def vectorized_attribute_model(tdb, rh):
    tdb_arr = np.asarray(tdb, dtype=float)
    rh_arr = np.asarray(rh, dtype=float)
    return SimpleNamespace(
        pmv=(tdb_arr - 25.0) / 10.0 - (rh_arr - 50.0) / 100.0,
        ppd=np.full_like(tdb_arr, 10.0, dtype=float),
    )


def vectorized_mapping_model(tdb, rh):
    tdb_arr = np.asarray(tdb, dtype=float)
    rh_arr = np.asarray(rh, dtype=float)
    return {
        "pmv": (tdb_arr - 25.0) / 10.0 - (rh_arr - 50.0) / 100.0,
        "ppd": np.full_like(tdb_arr, 10.0, dtype=float),
    }


def required_input_model(tdb, rh, met):
    tdb_arr = np.asarray(tdb, dtype=float)
    rh_arr = np.asarray(rh, dtype=float)
    met_arr = np.asarray(met, dtype=float)
    return SimpleNamespace(pmv=tdb_arr + rh_arr + met_arr)


def linked_input_model(tdb, tr, rh):
    tdb_arr = np.asarray(tdb, dtype=float)
    tr_arr = np.asarray(tr, dtype=float)
    rh_arr = np.asarray(rh, dtype=float)
    return SimpleNamespace(pmv=(tdb_arr + tr_arr + rh_arr) / 10.0)


def mismatched_payload_model(tdb, rh):
    _ = np.asarray(tdb, dtype=float)
    _ = np.asarray(rh, dtype=float)
    return SimpleNamespace(pmv=np.array([1.0, 2.0, 3.0], dtype=float))


def outer_margin_invalid_model(tdb, rh):
    tdb_arr = np.asarray(tdb, dtype=float)
    rh_arr = np.asarray(rh, dtype=float)
    pmv = (tdb_arr - 25.0) / 10.0 - (rh_arr - 50.0) / 100.0
    valid = (tdb_arr >= 22.0) & (tdb_arr <= 28.0) & (rh_arr >= 30.0) & (rh_arr <= 70.0)
    return SimpleNamespace(pmv=np.where(valid, pmv, np.nan))


def all_invalid_model(tdb, rh):
    tdb_arr = np.asarray(tdb, dtype=float)
    _ = np.asarray(rh, dtype=float)
    return SimpleNamespace(pmv=np.full_like(tdb_arr, np.nan, dtype=float))


def internal_hole_model(tdb, rh):
    tdb_arr = np.asarray(tdb, dtype=float)
    rh_arr = np.asarray(rh, dtype=float)
    pmv = (tdb_arr - 25.0) / 10.0 - (rh_arr - 50.0) / 100.0
    hole = ((tdb_arr - 25.0) ** 2) / 4.0 + ((rh_arr - 50.0) ** 2) / 225.0 <= 1.0
    return SimpleNamespace(pmv=np.where(hole, np.nan, pmv))


def _new_plot() -> ThresholdPlot:
    return (
        ThresholdPlot(vectorized_attribute_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )


def assert_axis_limits(
    result: ThresholdPlotResult,
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    assert result.ax.get_xlim() == pytest.approx(xlim)
    assert result.ax.get_ylim() == pytest.approx(ylim)


def test_set_params_rejects_invalid_parameter_name() -> None:
    plot = ThresholdPlot(vectorized_attribute_model)

    with pytest.raises(ValueError, match="were not found"):
        plot.set_params(bad_name=1)


def test_set_axis_rejects_parameter_conflicts_with_set_params() -> None:
    plot = ThresholdPlot(vectorized_attribute_model).set_params(tdb=25.0)

    with pytest.raises(ValueError, match="already contains axis parameter"):
        plot.set_x_axis("tdb", 20.0, 30.0, resolution=1.0)

    plot = ThresholdPlot(vectorized_attribute_model).set_x_axis(
        "tdb", 20.0, 30.0, resolution=1.0
    )

    with pytest.raises(ValueError, match="cannot set axis parameter"):
        plot.set_params(tdb=25.0)


@pytest.mark.parametrize("configured_axis", [None, "x", "y"])
@pytest.mark.parametrize("regions_set", [False, True])
def test_plot_rejects_missing_axes_before_regions_and_fill_options(
    configured_axis: str | None, regions_set: bool
) -> None:
    plot = ThresholdPlot(vectorized_attribute_model)
    if configured_axis == "x":
        plot.set_x_axis("tdb", 20.0, 30.0)
    elif configured_axis == "y":
        plot.set_y_axis("rh", 20.0, 80.0)
    if regions_set:
        plot.set_regions(output="pmv", thresholds=[-0.5, 0.5])

    with pytest.raises(ValueError) as exc_info:
        plot.plot(fill_kws={"color": "red"})
    assert str(exc_info.value) == (
        "Axes are not set. Call set_x_axis(...) and set_y_axis(...) first."
    )


def test_plot_rejects_when_regions_not_set() -> None:
    plot = (
        ThresholdPlot(vectorized_attribute_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
    )

    with pytest.raises(ValueError) as exc_info:
        plot.plot(fill_kws={"color": "red"})
    assert str(exc_info.value) == (
        "Regions are not set. Call set_regions(...) before plot(...)."
    )


def test_plot_raises_for_missing_required_model_inputs() -> None:
    plot = (
        ThresholdPlot(required_input_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )

    with pytest.raises(ValueError, match="Missing required parameter"):
        plot.plot()


def test_plot_uses_provided_subplot_axis() -> None:
    fig, ax = plt.subplots()

    result = _new_plot().plot(ax=ax)

    assert isinstance(result, ThresholdPlotResult)
    assert result.ax is ax
    assert result.fig is fig
    assert len(result.fills) > 0


def test_plot_invalid_outer_margins_preserve_requested_limits() -> None:
    plot = (
        ThresholdPlot(outer_margin_invalid_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )

    result = plot.plot(legend=False, show_lines=False)

    assert_axis_limits(result, xlim=(20.0, 30.0), ylim=(20.0, 80.0))


def test_plot_invalid_regions_add_out_of_model_limits_legend_entry() -> None:
    plot = (
        ThresholdPlot(outer_margin_invalid_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )

    result = plot.plot(show_lines=False)

    assert result.legend is not None
    legend_labels = [text.get_text() for text in result.legend.get_texts()]
    assert "Out of model limits" in legend_labels


def test_plot_uses_default_invalid_color_in_legend() -> None:
    plot = (
        ThresholdPlot(outer_margin_invalid_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )

    result = plot.plot(show_lines=False)

    assert result.legend is not None
    legend_patches = result.legend.get_patches()
    invalid_patch = next(
        patch
        for patch, text in zip(legend_patches, result.legend.get_texts(), strict=False)
        if text.get_text() == "Out of model limits"
    )
    assert invalid_patch.get_facecolor()[:3] == pytest.approx(
        to_rgb(OUT_OF_MODEL_LIMITS_COLOR), abs=1e-3
    )


def test_plot_allows_custom_invalid_color() -> None:
    plot = (
        ThresholdPlot(outer_margin_invalid_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )

    result = plot.plot(show_lines=False, invalid_color="#ff00ff")

    assert result.legend is not None
    legend_patches = result.legend.get_patches()
    invalid_patch = next(
        patch
        for patch, text in zip(legend_patches, result.legend.get_texts(), strict=False)
        if text.get_text() == "Out of model limits"
    )
    assert invalid_patch.get_facecolor()[:3] == pytest.approx((1.0, 0.0, 1.0), abs=1e-3)


def test_plot_rejects_invalid_invalid_color() -> None:
    with pytest.raises(ValueError, match="invalid_color"):
        _new_plot().plot(invalid_color="not-a-color")


def test_plot_allows_custom_legend_kwargs() -> None:
    result = _new_plot().plot(legend_kws={"ncol": 1, "loc": "upper right"})

    assert result.legend is not None


def test_plot_all_valid_regions_do_not_add_invalid_legend_entry() -> None:
    result = _new_plot().plot(show_lines=False)

    assert result.legend is not None
    legend_labels = [text.get_text() for text in result.legend.get_texts()]
    assert "Out of model limits" not in legend_labels


def test_plot_supports_default_link_tr_from_tdb() -> None:
    result = (
        ThresholdPlot(linked_input_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[8.0])
        .plot()
    )

    assert isinstance(result, ThresholdPlotResult)
    assert len(result.fills) > 0


def test_plot_supports_default_link_tdb_from_tr() -> None:
    result = (
        ThresholdPlot(linked_input_model)
        .set_x_axis("tr", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[8.0])
        .plot()
    )

    assert isinstance(result, ThresholdPlotResult)
    assert len(result.fills) > 0


def test_plot_smoke_returns_editable_lines_present_fills_title_and_legend() -> None:
    result = _new_plot().plot(
        title="Demo Threshold Plot",
        legend=True,
        show_lines=True,
    )

    assert isinstance(result, ThresholdPlotResult)
    assert len(result.fills) > 0
    assert result.ax.get_title() == "Demo Threshold Plot"
    assert result.legend is not None
    assert result.lines
    assert all(isinstance(line, Line2D) for line in result.lines)

    legend_labels = [text.get_text() for text in result.legend.get_texts()]
    assert legend_labels == ["PMV < -0.5", "-0.5 ≤ PMV < 0.5", "PMV ≥ 0.5"]

    result.lines[0].set_linewidth(2.5)
    assert result.lines[0].get_linewidth() == 2.5


def test_plot_with_show_lines_false_returns_empty_lines_and_nonempty_fills() -> None:
    result = _new_plot().plot(show_lines=False)

    assert result.lines == []
    assert len(result.fills) > 0


def test_plot_uses_custom_labels_in_legend() -> None:
    custom_labels = ["Cold", "Neutral", "Hot"]
    result = (
        ThresholdPlot(vectorized_attribute_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5], labels=custom_labels)
        .plot(legend=True)
    )

    assert result.legend is not None
    legend_labels = [text.get_text() for text in result.legend.get_texts()]
    assert legend_labels == custom_labels


def test_plot_rejects_wrong_color_count() -> None:
    with pytest.raises(ValueError, match="colors must have length 3"):
        (
            ThresholdPlot(vectorized_attribute_model)
            .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
            .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
            .set_regions(
                output="pmv",
                thresholds=[-0.5, 0.5],
                colors=["#4c78a8", "#e15759"],
            )
        )


def test_plot_rejects_wrong_label_count() -> None:
    with pytest.raises(ValueError, match="labels must have length 3"):
        (
            ThresholdPlot(vectorized_attribute_model)
            .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
            .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
            .set_regions(
                output="pmv",
                thresholds=[-0.5, 0.5],
                labels=["Cold", "Hot"],
            )
        )


def test_plot_rejects_contour_payload_size_mismatch() -> None:
    plot = (
        ThresholdPlot(mismatched_payload_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )

    with pytest.raises(
        ValueError, match="Model output shape does not match the contour grid"
    ):
        plot.plot()


@pytest.mark.parametrize(
    "fill_kws",
    [
        {"color": "red"},
        {"facecolor": "red"},
    ],
)
def test_plot_rejects_fill_kws_color_override(fill_kws: dict[str, str]) -> None:
    with pytest.raises(
        ValueError, match="fill_kws cannot include 'color' or 'facecolor'"
    ):
        _new_plot().plot(fill_kws=fill_kws)


def test_plot_extracts_output_from_mapping_result() -> None:
    result = (
        ThresholdPlot(vectorized_mapping_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
        .plot()
    )

    assert isinstance(result, ThresholdPlotResult)
    assert len(result.fills) > 0


def test_plot_coarse_resolution_still_renders() -> None:
    result = (
        ThresholdPlot(vectorized_attribute_model)
        .set_x_axis("tdb", 20.0, 20.5, resolution=1.0)
        .set_y_axis("rh", 20.0, 25.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
        .plot()
    )
    assert isinstance(result, ThresholdPlotResult)
    assert result.ax.get_xlim() == pytest.approx((20.0, 20.5))
    assert result.ax.get_ylim() == pytest.approx((20.0, 25.0))


def test_plot_with_custom_labels_and_colors() -> None:
    result = (
        ThresholdPlot(vectorized_attribute_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(
            output="pmv",
            thresholds=[-0.5, 0.5],
            labels=["Cool", "Comfortable", "Warm"],
            colors=["#A3D1FF", "#A8E6CF", "#FFB7B2"],
        )
        .plot()
    )
    assert isinstance(result, ThresholdPlotResult)
    legend_labels = [t.get_text() for t in result.legend.get_texts()]
    assert legend_labels == ["Cool", "Comfortable", "Warm"]


def test_plot_grid_includes_exact_endpoints() -> None:
    result = (
        ThresholdPlot(vectorized_attribute_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=4.0)
        .set_y_axis("rh", 20.0, 75.0, resolution=8.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
        .plot()
    )
    assert result.ax.get_xlim() == pytest.approx((20.0, 30.0))
    assert result.ax.get_ylim() == pytest.approx((20.0, 75.0))


# ── curve backend ──────────────────────────────────────────────────────────


def x_non_monotone_model(tdb, rh):
    """Cross each threshold twice along x, but only once along rh, so this one
    is solvable in the y direction alone."""
    tdb_arr = np.asarray(tdb, dtype=float)
    rh_arr = np.asarray(rh, dtype=float)
    return SimpleNamespace(
        pmv=((tdb_arr - 25.0) ** 2) / 50.0 + (rh_arr - 50.0) / 40.0,
    )


def analytic_boundary_tdb(threshold: float, rh: np.ndarray) -> np.ndarray:
    """Exact tdb where ``vectorized_attribute_model`` reaches ``threshold``."""
    return 25.0 + 10.0 * threshold + (rh - 50.0) / 10.0


def _curve_plot(resolution: float = 1.0) -> ThresholdPlot:
    return (
        ThresholdPlot(vectorized_attribute_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=resolution)
        .set_y_axis("rh", 20.0, 80.0, resolution=resolution * 6.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )


def test_curve_boundaries_match_the_analytic_solution() -> None:
    result = _curve_plot().plot()

    for curve in result.boundaries:
        drawn = np.isfinite(curve.x)
        assert drawn.any()
        expected = analytic_boundary_tdb(curve.threshold, curve.y[drawn])
        assert curve.x[drawn] == pytest.approx(expected, abs=1e-5)


def test_curve_boundaries_are_nan_where_the_threshold_is_not_crossed() -> None:
    # pmv = -0.5 needs tdb = 20 + (rh - 50) / 10, which only enters the plotted
    # tdb range once rh passes 50.
    lower = _curve_plot().plot().boundaries[0]

    assert np.all(np.isnan(lower.x[lower.y < 50.0]))
    assert np.all(np.isfinite(lower.x[lower.y > 51.0]))


def test_curve_boundaries_do_not_depend_on_grid_resolution() -> None:
    coarse = _curve_plot(resolution=2.5).plot().boundaries[1]
    fine = _curve_plot(resolution=0.05).plot().boundaries[1]

    coarse_x = np.interp(fine.y, coarse.y, coarse.x)
    both_drawn = np.isfinite(coarse_x) & np.isfinite(fine.x)
    assert both_drawn.sum() > 10
    assert coarse_x[both_drawn] == pytest.approx(fine.x[both_drawn], abs=1e-4)


def test_curve_backend_tracks_the_validity_edge_exactly() -> None:
    # outer_margin_invalid_model is valid only on tdb 22-28, rh 30-70.
    plot = (
        ThresholdPlot(outer_margin_invalid_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )
    bands = plot._solve_bands("pmv", [-0.5, 0.5])

    assert bands is not None
    inside = (bands.rows > 30.0) & (bands.rows < 70.0)
    assert bands.has_valid[inside].all()
    assert bands.valid_start[inside] == pytest.approx(22.0, abs=1e-4)
    assert bands.valid_end[inside] == pytest.approx(28.0, abs=1e-4)
    # Rows outside the valid rh band have no valid point at all.
    assert not bands.has_valid[bands.rows < 29.0].any()
    assert not bands.has_valid[bands.rows > 71.0].any()
    assert bands.has_invalid


def test_model_errors_are_raised_not_turned_into_a_backend_fallback() -> None:
    plot = (
        ThresholdPlot(required_input_model)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(ValueError, match="Missing required parameter"):
            plot.plot()


def test_curve_backend_omits_invalid_legend_entry_when_everything_is_valid() -> None:
    result = _curve_plot().plot()

    legend_labels = [text.get_text() for text in result.legend.get_texts()]
    assert "Out of model limits" not in legend_labels
    assert len(result.fills) == 3


def test_plot_styles_an_axis_the_caller_supplied() -> None:
    # _PYTHERMALCOMFORT_RC only reaches axes created inside its rc_context, so
    # the styling has to be applied to the axis itself for a caller-supplied
    # one to match.
    with plt.rc_context(matplotlib.rcParamsDefault):
        _, ax = plt.subplots()
        assert ax.spines["top"].get_visible()

        result = _curve_plot().plot(ax=ax)

    assert not result.ax.spines["top"].get_visible()
    assert not result.ax.spines["right"].get_visible()
    assert not result.ax.xaxis.get_gridlines()[0].get_visible()


def test_plot_turns_off_a_grid_the_callers_rcparams_switched_on() -> None:
    with plt.rc_context({"axes.grid": True}):
        result = _curve_plot().plot()

    assert not result.ax.xaxis.get_gridlines()[0].get_visible()


def test_plot_limits_labeled_ticks_to_six_per_axis() -> None:
    result = _new_plot().plot()

    for axis in (result.ax.xaxis, result.ax.yaxis):
        min_val, max_val = axis.get_view_interval()
        visible_ticks = [
            tick for tick in axis.get_majorticklocs() if min_val <= tick <= max_val
        ]
        assert len(visible_ticks) <= 6


def test_out_of_model_limits_colour_contrasts_with_comfort_neutral() -> None:
    assert OUT_OF_MODEL_LIMITS_COLOR == "#C4C9CC"


def test_title_clears_a_legend_that_wraps_onto_two_rows() -> None:
    # Four regions with legend_ncol_max = 4 fit on one row; five wrap onto two,
    # and the title has to move up to clear the extra row.
    def make(n_thresholds: int) -> ThresholdPlot:
        return (
            ThresholdPlot(vectorized_attribute_model)
            .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
            .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
            .set_regions(
                output="pmv",
                thresholds=[-0.5 + 0.2 * i for i in range(n_thresholds)],
            )
        )

    one_row = make(1).plot(title="t")
    two_rows = make(4).plot(title="t")

    assert two_rows.ax.title.get_position()[1] > one_row.ax.title.get_position()[1]


def test_title_without_a_legend_keeps_the_default_position() -> None:
    result = _curve_plot().plot(title="t", legend=False)

    assert result.ax.title.get_position()[1] == pytest.approx(1.0)


def test_plot_exposes_one_boundary_curve_per_threshold() -> None:
    result = _curve_plot().plot()

    assert [(c.threshold, c.branch) for c in result.boundaries] == [(-0.5, 0), (0.5, 0)]
    assert result.lines == []

    with_lines = _curve_plot().plot(show_lines=True)
    assert len(with_lines.lines) == 2
    assert all(isinstance(line, Line2D) for line in with_lines.lines)


def test_plot_hides_boundary_lines_by_default() -> None:
    assert _curve_plot().plot().lines == []


def u_shaped_model(tdb, rh):
    """Dip to a minimum and rise again, so each threshold is crossed twice."""
    tdb_arr = np.asarray(tdb, dtype=float)
    rh_arr = np.asarray(rh, dtype=float)
    return SimpleNamespace(pmv=((tdb_arr - 25.0) ** 2) / 10.0 + (rh_arr - 50.0) / 200.0)


def _u_shaped_plot() -> ThresholdPlot:
    return (
        ThresholdPlot(u_shaped_model)
        .set_x_axis("tdb", 20.0, 30.0)
        .set_y_axis("rh", 20.0, 80.0)
        .set_regions(output="pmv", thresholds=[0.5, 1.5])
    )


def test_a_threshold_crossed_twice_gives_two_branches_and_five_bands() -> None:
    bands = _u_shaped_plot()._solve_bands("pmv", [0.5, 1.5])

    assert bands is not None
    # High outside, low in the middle: the outer regions each appear twice.
    assert bands.band_regions == [2, 1, 0, 1, 2]
    assert [(c.threshold, c.branch) for c in bands.curves] == [
        (0.5, 0),
        (0.5, 1),
        (1.5, 0),
        (1.5, 1),
    ]


def test_both_branches_of_a_doubled_threshold_are_exact() -> None:
    result = _u_shaped_plot().plot()

    for curve in result.boundaries:
        drawn = np.isfinite(curve.x)
        assert drawn.any()
        # (tdb - 25)^2 / 10 + (rh - 50) / 200 == threshold
        offset = np.sqrt(10.0 * (curve.threshold - (curve.y[drawn] - 50.0) / 200.0))
        expected = 25.0 + (offset if curve.branch == 1 else -offset)
        assert curve.x[drawn] == pytest.approx(expected, abs=1e-5)


def test_bands_of_a_doubled_threshold_are_symmetric_about_the_minimum() -> None:
    bands = _u_shaped_plot()._solve_bands("pmv", [0.5, 1.5])

    assert bands is not None
    # The model is symmetric about tdb = 25, so the band edges must be too.
    left, right = bands.edges[1], bands.edges[-2]
    assert (left + right) / 2 == pytest.approx(25.0, abs=1e-5)


def test_an_internal_hole_cannot_be_laid_out_and_raises() -> None:
    plot = (
        ThresholdPlot(internal_hole_model)
        .set_x_axis("tdb", 20.0, 30.0)
        .set_y_axis("rh", 20.0, 80.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )

    with pytest.raises(ValueError, match="Could not lay out the threshold regions"):
        plot.plot()


def test_an_entirely_invalid_surface_raises() -> None:
    plot = (
        ThresholdPlot(all_invalid_model)
        .set_x_axis("tdb", 20.0, 30.0)
        .set_y_axis("rh", 20.0, 80.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )

    with pytest.raises(ValueError, match="Could not lay out the threshold regions"):
        plot.plot()


def test_resolution_is_optional_and_changes_nothing_visible() -> None:
    def build(resolution: float | None) -> ThresholdPlot:
        return (
            ThresholdPlot(vectorized_attribute_model)
            .set_x_axis("tdb", 20.0, 30.0, resolution=resolution)
            .set_y_axis("rh", 20.0, 80.0, resolution=resolution)
            .set_regions(output="pmv", thresholds=[-0.5, 0.5])
        )

    omitted = build(None).plot().boundaries[1]
    given = build(0.05).plot().boundaries[1]

    # A finer resolution samples more rows, so compare on the common grid.
    resampled = np.interp(given.y, omitted.y, omitted.x)
    both = np.isfinite(resampled) & np.isfinite(given.x)
    assert both.sum() > 10
    assert resampled[both] == pytest.approx(given.x[both], abs=1e-4)


def test_plotting_does_not_emit_the_models_applicability_warnings() -> None:
    # Sweeping across a model's limits is how the chart finds the
    # out-of-model-limits area, so the warning would fire on every evaluation
    # and say nothing the chart is not about to shade.
    plot = (
        ThresholdPlot(pmv_ppd_iso)
        .set_x_axis("tdb", 10.0, 40.0)
        .set_y_axis("rh", 0.0, 100.0)
        .set_params(vr=0.1, met=1.2, clo=0.5, wme=0.0, model="7730-2005")
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = plot.plot()

    limit_warnings = [w for w in caught if "applicability limits" in str(w.message)]
    assert limit_warnings == []
    # The information is still on the chart.
    assert result.legend is not None
    assert "Out of model limits" in [t.get_text() for t in result.legend.get_texts()]


def test_calling_a_model_directly_still_warns() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pmv_ppd_iso(
            tdb=[35.0],
            tr=[35.0],
            vr=0.1,
            rh=[50.0],
            met=1.2,
            clo=0.5,
            model="7730-2005",
        )

    assert any("applicability limits" in str(w.message) for w in caught)


def test_curve_solver_falls_back_to_scanning_y_when_x_cannot_be_laid_out() -> None:
    # A parabola in x crosses each threshold twice in the middle rows and not
    # at all in the outer ones, so the x scan cannot give every row the same
    # layout. rh is monotone, so scanning the other way works.
    plot = (
        ThresholdPlot(x_non_monotone_model)
        .set_x_axis("tdb", 20.0, 30.0)
        .set_y_axis("rh", 20.0, 80.0)
        .set_regions(output="pmv", thresholds=[-0.25, 0.25])
    )
    bands = plot._solve_bands("pmv", [-0.25, 0.25])

    assert bands is not None
    assert bands.scan_axis == "y"
    assert bands.band_regions == [0, 1, 2]

    for curve in plot.plot().boundaries:
        drawn = np.isfinite(curve.y)
        assert drawn.any()
        # pmv = (tdb - 25)^2 / 50 + (rh - 50) / 40 == threshold
        expected = 50.0 + 40.0 * (
            curve.threshold - ((curve.x[drawn] - 25.0) ** 2) / 50.0
        )
        assert curve.y[drawn] == pytest.approx(expected, abs=1e-4)


def test_boundaries_are_solved_on_unrounded_model_output() -> None:
    # pmv_ppd_iso rounds PMV to 0.01, which turns the bisection predicate into
    # a staircase and parks the boundary on the edge of a plateau.
    def build(**params: object) -> ThresholdPlot:
        return (
            ThresholdPlot(pmv_ppd_iso)
            .set_x_axis("tdb", 18.0, 34.0)
            .set_y_axis("rh", 0.0, 100.0)
            .set_params(vr=0.1, met=1.2, clo=0.5, wme=0.0, model="7730-2005", **params)
            .set_regions(output="pmv", thresholds=[-0.5, 0.5])
        )

    def worst_error(curve) -> float:
        drawn = np.isfinite(curve.x)
        pmv = pmv_ppd_iso(
            tdb=curve.x[drawn],
            tr=curve.x[drawn],
            vr=0.1,
            rh=curve.y[drawn],
            met=1.2,
            clo=0.5,
            wme=0.0,
            model="7730-2005",
            round_output=False,
        ).pmv
        return float(np.abs(np.asarray(pmv) - curve.threshold).max())

    assert worst_error(build().plot().boundaries[0]) < 1e-4
    # An explicit round_output is still the caller's to make.
    assert worst_error(build(round_output=True).plot().boundaries[0]) > 1e-4


def test_only_applicability_warnings_are_suppressed_while_plotting() -> None:
    def noisy_model(tdb, rh):
        warnings.warn("solver did not converge", UserWarning, stacklevel=2)
        return SimpleNamespace(pmv=(np.asarray(tdb, dtype=float) - 25.0) / 10.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        (
            ThresholdPlot(noisy_model)
            .set_x_axis("tdb", 20.0, 30.0)
            .set_y_axis("rh", 20.0, 80.0)
            .set_regions(output="pmv", thresholds=[-0.5, 0.5])
            .plot()
        )

    assert any("solver did not converge" in str(w.message) for w in caught)


def test_title_offset_accepts_every_bbox_to_anchor_form() -> None:
    for anchor in (
        (0.5, 1.02),
        (0.5, 1.02, 0.4, 0.1),
        Bbox.from_bounds(0.5, 1.02, 0.4, 0.1),
    ):
        result = _curve_plot().plot(title="t", legend_kws={"bbox_to_anchor": anchor})
        assert result.ax.title.get_position()[1] == pytest.approx(1.14, abs=1e-6)


def test_title_clears_a_legend_anchored_in_figure_coordinates() -> None:
    # legend_kws reaches ax.legend(), so a caller can anchor the legend in
    # figure coordinates. The title is positioned in axes coordinates, so the
    # anchor has to be converted rather than read off raw.
    fig, ax = plt.subplots()
    result = _curve_plot().plot(
        ax=ax,
        title="t",
        legend_kws={"bbox_to_anchor": (0.5, 0.95), "bbox_transform": fig.transFigure},
    )
    fig.canvas.draw()

    to_axes = ax.transAxes.inverted()
    legend_top = to_axes.transform(result.legend.get_window_extent().p1)[1]
    title_bottom = to_axes.transform(ax.title.get_window_extent().p0)[1]

    assert title_bottom >= legend_top


def test_unhashable_callable_model_uses_default_labels() -> None:
    """A callable dataclass is unhashable, so it is treated as an unregistered model."""

    @dataclass
    class OffsetModel:
        offset: float = 0.0

        def __call__(self, tdb, rh):
            return vectorized_attribute_model(tdb + self.offset, rh)

    plot = (
        ThresholdPlot(OffsetModel())
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 20.0, 80.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )
    plot.plot()
