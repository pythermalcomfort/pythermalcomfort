from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend import Legend

from pythermalcomfort.plots.matplotlib.threshold_summary import (
    SummaryPlotResult,
    ThresholdSummary,
)


@pytest.fixture(autouse=True)
def close_all_figures():
    yield
    plt.close("all")


@pytest.fixture
def pmv_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tdb": [20.0, 25.0, 30.0],
            "rh": [50.0, 50.0, 50.0],
            "pmv": [-0.6, 0.0, 0.7],
        }
    )


def _new_summary(pmv_df: pd.DataFrame) -> ThresholdSummary:
    return ThresholdSummary(pmv_df).set_regions(output="pmv", thresholds=[-0.5, 0.5])


def test_init_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        ThresholdSummary([1, 2, 3])


def test_init_rejects_empty_dataframe() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        ThresholdSummary(pd.DataFrame())


def test_plot_requires_set_regions(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Call set_regions"):
        ThresholdSummary(pmv_df).plot()


def test_set_regions_rejects_empty_output_name(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        ThresholdSummary(pmv_df).set_regions(output="   ", thresholds=[-0.5, 0.5])


def test_set_regions_rejects_missing_output_column(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(
        ValueError,
        match="output column 'utci' was not found in the DataFrame.",
    ):
        ThresholdSummary(pmv_df).set_regions(output="utci", thresholds=[9, 26])


def test_set_regions_rejects_wrong_label_count(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="labels must have length 3"):
        ThresholdSummary(pmv_df).set_regions(
            output="pmv",
            thresholds=[-0.5, 0.5],
            labels=["Cold", "Hot"],
        )


def test_set_regions_rejects_wrong_color_count(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="colors must have length 3"):
        ThresholdSummary(pmv_df).set_regions(
            output="pmv",
            thresholds=[-0.5, 0.5],
            colors=["#4c78a8", "#e15759"],
        )


def test_plot_uses_provided_axis(pmv_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()

    result = _new_summary(pmv_df).plot(ax=ax)

    assert result.ax is ax
    assert result.fig is fig


def test_plot_vertical_mode_executes(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot(vertical=True)

    assert isinstance(result, SummaryPlotResult)
    assert len(result.artists) > 0


def test_plot_uses_compact_default_figsize(pmv_df: pd.DataFrame) -> None:
    horizontal = _new_summary(pmv_df).plot()
    vertical = _new_summary(pmv_df).plot(vertical=True)

    assert tuple(horizontal.fig.get_size_inches()) == pytest.approx((6.4, 1.8))
    assert tuple(vertical.fig.get_size_inches()) == pytest.approx((2.8, 4.0))


def test_vertical_empty_labels_uses_compact_xlim(pmv_df: pd.DataFrame) -> None:
    result = (
        ThresholdSummary(pmv_df)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5], labels=[])
        .plot(vertical=True, legend=False)
    )

    left, right = result.ax.get_xlim()
    assert right - left == pytest.approx(1.0)


def test_plot_bar_kws_apply_to_horizontal_bars(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot(bar_kws={"alpha": 0.4, "linewidth": 2.5})
    patch = result.artists[0].patches[0]

    assert patch.get_alpha() == pytest.approx(0.4)
    assert patch.get_linewidth() == pytest.approx(2.5)


def test_plot_bar_kws_apply_to_vertical_bars(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot(vertical=True, bar_kws={"hatch": "//"})
    patch = result.artists[0].patches[0]

    assert patch.get_hatch() == "//"


def test_plot_bar_kws_can_override_horizontal_bar_height(
    pmv_df: pd.DataFrame,
) -> None:
    result = _new_summary(pmv_df).plot(bar_kws={"height": 0.42})
    patch = result.artists[0].patches[0]

    assert patch.get_height() == pytest.approx(0.42)


def test_plot_bar_kws_can_override_vertical_bar_width(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot(vertical=True, bar_kws={"width": 0.52})
    patch = result.artists[0].patches[0]

    assert patch.get_width() == pytest.approx(0.52)


def test_plot_returns_percentages(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot()

    expected_labels = ["PMV < -0.5", "-0.5 ≤ PMV < 0.5", "PMV ≥ 0.5"]
    assert result.percentages.index.tolist() == expected_labels
    assert result.percentages.tolist() == pytest.approx([33.3, 33.3, 33.3])


def test_plot_uses_custom_labels_when_provided(pmv_df: pd.DataFrame) -> None:
    custom_labels = ["Cold", "Neutral", "Hot"]
    result = (
        ThresholdSummary(pmv_df)
        .set_regions(
            output="pmv",
            thresholds=[-0.5, 0.5],
            labels=custom_labels,
        )
        .plot()
    )

    assert result.percentages.index.tolist() == custom_labels


def test_plot_supports_utci_like_existing_column() -> None:
    df = pd.DataFrame(
        {
            "tdb": [10.0, 24.0, 32.0],
            "utci": [5.0, 18.0, 28.0],
        }
    )

    result = ThresholdSummary(df).set_regions(output="utci", thresholds=[9, 26]).plot()

    expected_labels = ["UTCI < 9", "9 ≤ UTCI < 26", "UTCI ≥ 26"]
    assert result.percentages.index.tolist() == expected_labels
    assert result.percentages.tolist() == pytest.approx([33.3, 33.3, 33.3])


def test_set_regions_rejects_non_numeric_output_values() -> None:
    df = pd.DataFrame({"pmv": [0.1, "bad", 0.2]})

    with pytest.raises(ValueError, match="non-numeric"):
        ThresholdSummary(df).set_regions(output="pmv", thresholds=[-0.5, 0.5])


def test_set_regions_rejects_non_finite_output_values() -> None:
    df = pd.DataFrame({"pmv": [0.1, float("inf"), 0.2]})

    with pytest.raises(ValueError, match="non-finite"):
        ThresholdSummary(df).set_regions(output="pmv", thresholds=[-0.5, 0.5])


def test_summary_with_custom_labels() -> None:
    df = pd.DataFrame({"pmv": [0.7, -0.3, 0.1, -0.8, 1.2]})
    result = (
        ThresholdSummary(df)
        .set_regions(
            output="pmv",
            thresholds=[-0.5, 0.5],
            labels=["Cool", "Comfortable", "Warm"],
        )
        .plot()
    )
    assert isinstance(result, SummaryPlotResult)
    assert list(result.percentages.index) == ["Cool", "Comfortable", "Warm"]


def test_summary_handles_numeric_string_column() -> None:
    df = pd.DataFrame({"pmv": ["0.7", "-0.3", "0.1", "-0.8", "1.2"]})
    result = (
        ThresholdSummary(df).set_regions(output="pmv", thresholds=[-0.5, 0.5]).plot()
    )
    assert isinstance(result, SummaryPlotResult)
    assert result.percentages.sum() > 99.9


def test_plot_legend_shown_by_default(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot()

    assert isinstance(result.legend, Legend)


def test_plot_title_does_not_overlap_horizontal_legend(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot(title="PMV Distribution")
    result.fig.canvas.draw()
    renderer = result.fig.canvas.get_renderer()

    title_bbox = result.ax.title.get_window_extent(renderer)
    legend_bbox = result.legend.get_window_extent(renderer)

    assert title_bbox.y0 > legend_bbox.y1


def test_plot_title_does_not_overlap_vertical_legend(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot(
        vertical=True,
        title="PMV Distribution (Vertical)",
    )
    result.fig.canvas.draw()
    renderer = result.fig.canvas.get_renderer()

    title_bbox = result.ax.title.get_window_extent(renderer)
    legend_bbox = result.legend.get_window_extent(renderer)

    assert title_bbox.y0 > legend_bbox.y1


def test_plot_legend_none_when_disabled(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot(legend=False)

    assert result.legend is None


def test_plot_result_has_no_data_attribute(pmv_df: pd.DataFrame) -> None:
    result = _new_summary(pmv_df).plot()

    assert not hasattr(result, "data")


def test_set_regions_empty_labels_suppresses_label_text(pmv_df: pd.DataFrame) -> None:
    result = (
        ThresholdSummary(pmv_df)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5], labels=[])
        .plot()
    )

    legend_texts = [t.get_text() for t in result.legend.get_texts()]
    assert legend_texts == ["", "", ""]


def test_empty_labels_suppresses_label_text_via_thresholds(
    pmv_df: pd.DataFrame,
) -> None:
    result = (
        ThresholdSummary(pmv_df)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5], labels=[])
        .plot()
    )

    legend_texts = [t.get_text() for t in result.legend.get_texts()]
    assert legend_texts == ["", "", ""]
