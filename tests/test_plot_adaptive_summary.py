"""Tests for AdaptiveSummary."""

from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend import Legend

from pythermalcomfort.plots.matplotlib.adaptive_summary import AdaptiveSummary
from pythermalcomfort.plots.matplotlib.threshold_summary import SummaryPlotResult


@pytest.fixture(autouse=True)
def close_all_figures():
    yield
    plt.close("all")


# ── fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def ashrae_df() -> pd.DataFrame:
    """Three rows that each fall into a distinct ASHRAE region.

    With t_running_mean=20 → t_cmf=24.0, ce=0 (v=0.1):
      90% band: [21.5, 26.5]
      80% band: [20.5, 27.5]

    - to=25 → in 90% (and 80%, overwritten by narrower)
    - to=21 → in 80% only (21 < 21.5)
    - to=19 → not acceptable (19 < 20.5)
    """
    return pd.DataFrame(
        {
            "tdb": [25.0, 21.0, 19.0],
            "tr": [25.0, 21.0, 19.0],
            "t_running_mean": [20.0, 20.0, 20.0],
            "v": [0.1, 0.1, 0.1],
        }
    )


@pytest.fixture
def en_df() -> pd.DataFrame:
    """Four rows that each fall into a distinct EN category.

    With t_running_mean=20 → t_cmf=25.4, ce=0 (v=0.1):
      Cat I:   [22.4, 27.4]
      Cat II:  [21.4, 28.4]
      Cat III: [20.4, 29.4]

    - to=25 → Cat I
    - to=22 → Cat II only (22 >= 21.4 but 22 < 22.4)
    - to=21 → Cat III only (21 >= 20.4 but 21 < 21.4)
    - to=20 → not acceptable (20 < 20.4)
    """
    return pd.DataFrame(
        {
            "tdb": [25.0, 22.0, 21.0, 20.0],
            "tr": [25.0, 22.0, 21.0, 20.0],
            "t_running_mean": [20.0, 20.0, 20.0, 20.0],
            "v": [0.1, 0.1, 0.1, 0.1],
        }
    )


# ── helpers ────────────────────────────────────────────────────────────────


def _new_ashrae(df: pd.DataFrame) -> AdaptiveSummary:
    return AdaptiveSummary(df).set_regions(output="acceptability", thresholds=[80, 90])


def _new_en(df: pd.DataFrame) -> AdaptiveSummary:
    return AdaptiveSummary(df).set_regions(
        output="acceptability_cat", thresholds=["i", "ii", "iii"]
    )


# ═══════════════════════════════════════════════════════════════════════════
# Initialization
# ═══════════════════════════════════════════════════════════════════════════


def test_init_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        AdaptiveSummary([1, 2, 3])


def test_init_rejects_empty_dataframe() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        AdaptiveSummary(pd.DataFrame())


# ═══════════════════════════════════════════════════════════════════════════
# set_regions validation
# ═══════════════════════════════════════════════════════════════════════════


def test_set_regions_rejects_non_string_output(ashrae_df: pd.DataFrame) -> None:
    with pytest.raises(TypeError, match="output must be a string"):
        AdaptiveSummary(ashrae_df).set_regions(output=123, thresholds=[80])


def test_set_regions_rejects_unknown_output(ashrae_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="output must be one of"):
        AdaptiveSummary(ashrae_df).set_regions(output="unknown", thresholds=[80])


def test_set_regions_rejects_empty_thresholds(ashrae_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="at least one band key"):
        AdaptiveSummary(ashrae_df).set_regions(output="acceptability", thresholds=[])


def test_set_regions_rejects_invalid_ashrae_threshold(
    ashrae_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="Invalid threshold"):
        AdaptiveSummary(ashrae_df).set_regions(
            output="acceptability", thresholds=[80, 99]
        )


def test_set_regions_rejects_invalid_en_threshold(en_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Invalid threshold"):
        AdaptiveSummary(en_df).set_regions(
            output="acceptability_cat", thresholds=["iv"]
        )


def test_set_regions_rejects_wrong_label_count(ashrae_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="labels must have length 3"):
        AdaptiveSummary(ashrae_df).set_regions(
            output="acceptability",
            thresholds=[80, 90],
            labels=["A", "B"],
        )


def test_set_regions_rejects_wrong_color_count(ashrae_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="colors must have length 3"):
        AdaptiveSummary(ashrae_df).set_regions(
            output="acceptability",
            thresholds=[80, 90],
            colors=["#ff0000"],
        )


def test_set_regions_rejects_invalid_color(ashrae_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Invalid color"):
        AdaptiveSummary(ashrae_df).set_regions(
            output="acceptability",
            thresholds=[80, 90],
            colors=["red", "not_a_color", "blue"],
        )


def test_set_regions_rejects_missing_columns() -> None:
    df = pd.DataFrame({"tdb": [25.0], "tr": [25.0]})
    with pytest.raises(ValueError, match="missing required column"):
        AdaptiveSummary(df).set_regions(output="acceptability", thresholds=[80])


def test_plot_requires_set_regions(ashrae_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Call set_regions"):
        AdaptiveSummary(ashrae_df).plot()


# ═══════════════════════════════════════════════════════════════════════════
# ASHRAE 55 — percentages and labels
# ═══════════════════════════════════════════════════════════════════════════


def test_ashrae_default_labels(ashrae_df: pd.DataFrame) -> None:
    result = _new_ashrae(ashrae_df).plot()
    expected = ["Not Acceptable", "80% Acceptability", "90% Acceptability"]
    assert result.percentages.index.tolist() == expected


def test_ashrae_percentages(ashrae_df: pd.DataFrame) -> None:
    result = _new_ashrae(ashrae_df).plot()
    assert result.percentages.tolist() == pytest.approx([33.3, 33.3, 33.3])


def test_ashrae_single_band_80(ashrae_df: pd.DataFrame) -> None:
    result = (
        AdaptiveSummary(ashrae_df)
        .set_regions(output="acceptability", thresholds=[80])
        .plot()
    )
    assert result.percentages.index.tolist() == [
        "Not Acceptable",
        "80% Acceptability",
    ]
    # to=25 and to=21 both fall inside 80% band; to=19 does not
    assert result.percentages.tolist() == pytest.approx([33.3, 66.7])


def test_ashrae_single_band_90(ashrae_df: pd.DataFrame) -> None:
    result = (
        AdaptiveSummary(ashrae_df)
        .set_regions(output="acceptability", thresholds=[90])
        .plot()
    )
    assert result.percentages.index.tolist() == [
        "Not Acceptable",
        "90% Acceptability",
    ]
    # only to=25 falls inside 90% band
    assert result.percentages.tolist() == pytest.approx([66.7, 33.3])


def test_ashrae_custom_labels(ashrae_df: pd.DataFrame) -> None:
    labels = ["Bad", "OK", "Great"]
    result = (
        AdaptiveSummary(ashrae_df)
        .set_regions(output="acceptability", thresholds=[80, 90], labels=labels)
        .plot()
    )
    assert result.percentages.index.tolist() == labels


def test_ashrae_custom_colors(ashrae_df: pd.DataFrame) -> None:
    colors = ["#FF0000", "#00FF00", "#0000FF"]
    result = (
        AdaptiveSummary(ashrae_df)
        .set_regions(output="acceptability", thresholds=[80, 90], colors=colors)
        .plot()
    )
    assert isinstance(result, SummaryPlotResult)


def test_ashrae_threshold_order_independent(ashrae_df: pd.DataFrame) -> None:
    r1 = _new_ashrae(ashrae_df).plot()
    r2 = (
        AdaptiveSummary(ashrae_df)
        .set_regions(output="acceptability", thresholds=[90, 80])
        .plot()
    )
    assert r1.percentages.tolist() == r2.percentages.tolist()
    assert r1.percentages.index.tolist() == r2.percentages.index.tolist()


def test_ashrae_duplicate_thresholds_deduplicated(ashrae_df: pd.DataFrame) -> None:
    result = (
        AdaptiveSummary(ashrae_df)
        .set_regions(output="acceptability", thresholds=[80, 80, 90])
        .plot()
    )
    assert result.percentages.index.tolist() == [
        "Not Acceptable",
        "80% Acceptability",
        "90% Acceptability",
    ]


def test_ashrae_all_acceptable() -> None:
    """Every data point falls within the 90% band."""
    df = pd.DataFrame(
        {
            "tdb": [24.0, 25.0, 23.0],
            "tr": [24.0, 25.0, 23.0],
            "t_running_mean": [20.0, 20.0, 20.0],
            "v": [0.1, 0.1, 0.1],
        }
    )
    result = _new_ashrae(df).plot()
    assert result.percentages.iloc[0] == pytest.approx(0.0)  # Not Acceptable
    assert result.percentages.iloc[1] == pytest.approx(0.0)  # 80% only
    assert result.percentages.iloc[2] == pytest.approx(100.0)  # 90%


def test_ashrae_none_acceptable() -> None:
    """Every data point falls outside all bands (t_running_mean out of range)."""
    df = pd.DataFrame(
        {
            "tdb": [25.0, 26.0],
            "tr": [25.0, 26.0],
            "t_running_mean": [5.0, 5.0],  # below 10 → model returns NaN
            "v": [0.1, 0.1],
        }
    )
    result = _new_ashrae(df).plot()
    assert result.percentages.iloc[0] == pytest.approx(100.0)  # all Not Acceptable


# ═══════════════════════════════════════════════════════════════════════════
# EN 16798 — percentages and labels
# ═══════════════════════════════════════════════════════════════════════════


def test_en_default_labels(en_df: pd.DataFrame) -> None:
    result = _new_en(en_df).plot()
    expected = ["Not Acceptable", "Category III", "Category II", "Category I"]
    assert result.percentages.index.tolist() == expected


def test_en_percentages(en_df: pd.DataFrame) -> None:
    result = _new_en(en_df).plot()
    assert result.percentages.tolist() == pytest.approx([25.0, 25.0, 25.0, 25.0])


def test_en_subset_two_bands(en_df: pd.DataFrame) -> None:
    result = (
        AdaptiveSummary(en_df)
        .set_regions(output="acceptability_cat", thresholds=["i", "ii"])
        .plot()
    )
    assert result.percentages.index.tolist() == [
        "Not Acceptable",
        "Category II",
        "Category I",
    ]
    # to=21 and to=20 are both outside Cat II → Not Acceptable
    assert result.percentages.tolist() == pytest.approx([50.0, 25.0, 25.0])


def test_en_single_band_cat_iii(en_df: pd.DataFrame) -> None:
    result = (
        AdaptiveSummary(en_df)
        .set_regions(output="acceptability_cat", thresholds=["iii"])
        .plot()
    )
    assert result.percentages.index.tolist() == [
        "Not Acceptable",
        "Category III",
    ]
    # to=25,22,21 all in Cat III; to=20 not
    assert result.percentages.tolist() == pytest.approx([25.0, 75.0])


def test_en_threshold_order_independent(en_df: pd.DataFrame) -> None:
    r1 = _new_en(en_df).plot()
    r2 = (
        AdaptiveSummary(en_df)
        .set_regions(output="acceptability_cat", thresholds=["i", "iii", "ii"])
        .plot()
    )
    assert r1.percentages.tolist() == r2.percentages.tolist()
    assert r1.percentages.index.tolist() == r2.percentages.index.tolist()


def test_en_case_insensitive_keys(en_df: pd.DataFrame) -> None:
    result = (
        AdaptiveSummary(en_df)
        .set_regions(output="acceptability_cat", thresholds=["I", "II", "III"])
        .plot()
    )
    expected = ["Not Acceptable", "Category III", "Category II", "Category I"]
    assert result.percentages.index.tolist() == expected


def test_en_custom_labels(en_df: pd.DataFrame) -> None:
    labels = ["Fail", "OK", "Good", "Excellent"]
    result = (
        AdaptiveSummary(en_df)
        .set_regions(
            output="acceptability_cat",
            thresholds=["i", "ii", "iii"],
            labels=labels,
        )
        .plot()
    )
    assert result.percentages.index.tolist() == labels


# ═══════════════════════════════════════════════════════════════════════════
# Plot rendering
# ═══════════════════════════════════════════════════════════════════════════


def test_plot_returns_summary_plot_result(ashrae_df: pd.DataFrame) -> None:
    result = _new_ashrae(ashrae_df).plot()
    assert isinstance(result, SummaryPlotResult)


def test_plot_uses_provided_axis(ashrae_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    result = _new_ashrae(ashrae_df).plot(ax=ax)
    assert result.ax is ax
    assert result.fig is fig


def test_plot_creates_figure_when_no_axis(ashrae_df: pd.DataFrame) -> None:
    result = _new_ashrae(ashrae_df).plot()
    assert result.fig is not None
    assert result.ax is not None


def test_plot_vertical_mode(ashrae_df: pd.DataFrame) -> None:
    result = _new_ashrae(ashrae_df).plot(vertical=True)
    assert isinstance(result, SummaryPlotResult)
    assert len(result.artists) > 0


def test_plot_legend_shown_by_default(ashrae_df: pd.DataFrame) -> None:
    result = _new_ashrae(ashrae_df).plot()
    assert isinstance(result.legend, Legend)


def test_plot_legend_none_when_disabled(ashrae_df: pd.DataFrame) -> None:
    result = _new_ashrae(ashrae_df).plot(legend=False)
    assert result.legend is None


def test_plot_title_rendered(ashrae_df: pd.DataFrame) -> None:
    result = _new_ashrae(ashrae_df).plot(title="Test Title")
    assert result.ax.get_title() == "Test Title"


def test_plot_bar_kws_forwarded(ashrae_df: pd.DataFrame) -> None:
    result = _new_ashrae(ashrae_df).plot(bar_kws={"alpha": 0.5})
    patch = result.artists[0].patches[0]
    assert patch.get_alpha() == pytest.approx(0.5)


def test_plot_has_artists(ashrae_df: pd.DataFrame) -> None:
    result = _new_ashrae(ashrae_df).plot()
    assert len(result.artists) > 0


def test_plot_title_does_not_overlap_horizontal_legend(
    ashrae_df: pd.DataFrame,
) -> None:
    result = _new_ashrae(ashrae_df).plot(title="ASHRAE Summary")
    result.fig.canvas.draw()
    renderer = result.fig.canvas.get_renderer()
    title_bbox = result.ax.title.get_window_extent(renderer)
    legend_bbox = result.legend.get_window_extent(renderer)
    assert title_bbox.y0 > legend_bbox.y1


def test_plot_title_does_not_overlap_vertical_legend(
    en_df: pd.DataFrame,
) -> None:
    result = _new_en(en_df).plot(vertical=True, title="EN Summary (Vertical)")
    result.fig.canvas.draw()
    renderer = result.fig.canvas.get_renderer()
    title_bbox = result.ax.title.get_window_extent(renderer)
    legend_bbox = result.legend.get_window_extent(renderer)
    assert title_bbox.y0 > legend_bbox.y1


# ═══════════════════════════════════════════════════════════════════════════
# Percentages consistency
# ═══════════════════════════════════════════════════════════════════════════


def test_ashrae_percentages_sum_to_100(ashrae_df: pd.DataFrame) -> None:
    result = _new_ashrae(ashrae_df).plot()
    assert result.percentages.sum() == pytest.approx(100.0, abs=0.5)


def test_en_percentages_sum_to_100(en_df: pd.DataFrame) -> None:
    result = _new_en(en_df).plot()
    assert result.percentages.sum() == pytest.approx(100.0, abs=0.5)


# ═══════════════════════════════════════════════════════════════════════════
# set_params
# ═══════════════════════════════════════════════════════════════════════════


def test_set_params_returns_self(ashrae_df: pd.DataFrame) -> None:
    builder = AdaptiveSummary(ashrae_df)
    assert builder.set_params(limit_inputs=False) is builder


def test_set_params_limit_inputs_false(ashrae_df: pd.DataFrame) -> None:
    result = (
        AdaptiveSummary(ashrae_df)
        .set_params(limit_inputs=False)
        .set_regions(output="acceptability", thresholds=[80, 90])
        .plot()
    )
    assert isinstance(result, SummaryPlotResult)
    assert result.percentages.sum() == pytest.approx(100.0, abs=0.5)


# ═══════════════════════════════════════════════════════════════════════════
# Method chaining
# ═══════════════════════════════════════════════════════════════════════════


def test_set_regions_returns_self(ashrae_df: pd.DataFrame) -> None:
    builder = AdaptiveSummary(ashrae_df)
    returned = builder.set_regions(output="acceptability", thresholds=[80, 90])
    assert returned is builder
