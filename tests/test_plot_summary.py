from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend import Legend

from pythermalcomfort.plots.matplotlib._shared import _default_region_colors
from pythermalcomfort.plots.matplotlib.summary import (
    SummaryPlot,
    SummaryPlotResult,
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


def _new_summary(pmv_df: pd.DataFrame) -> SummaryPlot:
    return SummaryPlot(pmv_df).set_regions(output="pmv", thresholds=[-0.5, 0.5])


@pytest.mark.parametrize(
    ("n_regions", "expected"),
    [
        (2, ["#86AEC8", "#D88B7B"]),
        (3, ["#86AEC8", "#F1F3F2", "#D88B7B"]),
        (5, ["#5F8FA9", "#B7D0DE", "#F1F3F2", "#E7B7AC", "#C66B5E"]),
        (
            7,
            [
                "#527F98",
                "#86AEC8",
                "#C7DCE6",
                "#F1F3F2",
                "#EBCBC3",
                "#D88B7B",
                "#B85F55",
            ],
        ),
    ],
)
def test_default_region_palette_progresses_from_cool_to_warm(
    n_regions: int, expected: list[str]
) -> None:
    assert _default_region_colors(n_regions) == expected


def test_init_rejects_non_dataframe() -> None:
    with pytest.raises(TypeError, match="pandas DataFrame"):
        SummaryPlot([1, 2, 3])


def test_init_rejects_empty_dataframe() -> None:
    with pytest.raises(ValueError, match="at least one row"):
        SummaryPlot(pd.DataFrame())


def test_plot_requires_set_regions(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Call set_regions"):
        SummaryPlot(pmv_df).plot()


def test_set_regions_rejects_empty_output_name(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        SummaryPlot(pmv_df).set_regions(output="   ", thresholds=[-0.5, 0.5])


def test_set_regions_rejects_missing_output_column(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(
        ValueError,
        match="output column 'utci' was not found in the DataFrame.",
    ):
        SummaryPlot(pmv_df).set_regions(output="utci", thresholds=[9, 26])


def test_set_regions_rejects_wrong_label_count(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="labels must have length 3"):
        SummaryPlot(pmv_df).set_regions(
            output="pmv",
            thresholds=[-0.5, 0.5],
            labels=["Cold", "Hot"],
        )


def test_set_regions_rejects_wrong_color_count(pmv_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="colors must have length 3"):
        SummaryPlot(pmv_df).set_regions(
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
        SummaryPlot(pmv_df)
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
        SummaryPlot(pmv_df)
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

    result = SummaryPlot(df).set_regions(output="utci", thresholds=[9, 26]).plot()

    expected_labels = ["UTCI < 9", "9 ≤ UTCI < 26", "UTCI ≥ 26"]
    assert result.percentages.index.tolist() == expected_labels
    assert result.percentages.tolist() == pytest.approx([33.3, 33.3, 33.3])


def test_set_regions_rejects_non_numeric_output_values() -> None:
    df = pd.DataFrame({"pmv": [0.1, "bad", 0.2]})

    with pytest.raises(ValueError, match="non-numeric"):
        SummaryPlot(df).set_regions(output="pmv", thresholds=[-0.5, 0.5])


def test_set_regions_rejects_non_finite_output_values() -> None:
    df = pd.DataFrame({"pmv": [0.1, float("inf"), 0.2]})

    with pytest.raises(ValueError, match="non-finite"):
        SummaryPlot(df).set_regions(output="pmv", thresholds=[-0.5, 0.5])


def test_summary_with_custom_labels() -> None:
    df = pd.DataFrame({"pmv": [0.7, -0.3, 0.1, -0.8, 1.2]})
    result = (
        SummaryPlot(df)
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
    result = SummaryPlot(df).set_regions(output="pmv", thresholds=[-0.5, 0.5]).plot()
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
        SummaryPlot(pmv_df)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5], labels=[])
        .plot()
    )

    legend_texts = [t.get_text() for t in result.legend.get_texts()]
    assert legend_texts == ["", "", ""]


def test_empty_labels_suppresses_label_text_via_thresholds(
    pmv_df: pd.DataFrame,
) -> None:
    result = (
        SummaryPlot(pmv_df)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5], labels=[])
        .plot()
    )

    legend_texts = [t.get_text() for t in result.legend.get_texts()]
    assert legend_texts == ["", "", ""]


# ── set_categories ──────────────────────────────────────────────────────────


@pytest.fixture
def categories_df() -> pd.DataFrame:
    return pd.DataFrame({"row": range(10)})


def test_set_categories_returns_expected_percentages(
    categories_df: pd.DataFrame,
) -> None:
    categories = ["A"] * 3 + ["B"] * 7
    result = (
        SummaryPlot(categories_df)
        .set_categories(
            categories, labels=["A", "B", "C"], colors=["#111", "#222", "#333"]
        )
        .plot()
    )

    assert result.percentages.tolist() == [30.0, 70.0, 0.0]
    assert list(result.percentages.index) == ["A", "B", "C"]


def test_set_categories_zero_count_category_renders_at_zero(
    categories_df: pd.DataFrame,
) -> None:
    categories = ["A"] * 10
    result = (
        SummaryPlot(categories_df)
        .set_categories(categories, labels=["A", "B"], colors=["#111", "#222"])
        .plot(legend=False)
    )

    assert result.percentages["B"] == 0.0


def test_set_categories_rejects_length_mismatch(categories_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="one value per row"):
        SummaryPlot(categories_df).set_categories(
            ["A"] * 9, labels=["A"], colors=["#111"]
        )


def test_set_categories_rejects_unknown_value(categories_df: pd.DataFrame) -> None:
    categories = ["A"] * 9 + ["Z"]
    with pytest.raises(ValueError, match="not present in labels"):
        SummaryPlot(categories_df).set_categories(
            categories, labels=["A"], colors=["#111"]
        )


def test_set_categories_rejects_labels_colors_length_mismatch(
    categories_df: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="same length"):
        SummaryPlot(categories_df).set_categories(
            ["A"] * 10, labels=["A", "B"], colors=["#111"]
        )


def test_set_categories_rejects_duplicate_labels(categories_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        SummaryPlot(categories_df).set_categories(
            ["A"] * 10, labels=["A", "A"], colors=["#111", "#222"]
        )


def test_set_categories_rejects_invalid_color(categories_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Invalid color"):
        SummaryPlot(categories_df).set_categories(
            ["A"] * 10, labels=["A"], colors=["not-a-color"]
        )


def test_set_categories_clears_region_config(pmv_df: pd.DataFrame) -> None:
    sp = _new_summary(pmv_df)
    assert sp._region_config is not None

    sp.set_categories(["A", "B", "A"], labels=["A", "B"], colors=["#111", "#222"])
    assert sp._region_config is None
    assert sp._categories is not None


def test_set_regions_clears_categories(pmv_df: pd.DataFrame) -> None:
    sp = SummaryPlot(pmv_df).set_categories(
        ["A", "B", "A"], labels=["A", "B"], colors=["#111", "#222"]
    )
    assert sp._categories is not None

    sp.set_regions(output="pmv", thresholds=[-0.5, 0.5])
    assert sp._categories is None
    assert sp._category_labels is None
    assert sp._category_colors is None


def test_set_categories_adaptive_np_select_recipe_end_to_end() -> None:
    """Reproduces the np.select recipe documented on set_categories()."""
    np = pytest.importorskip("numpy")
    from pythermalcomfort.models import adaptive_ashrae

    df = pd.DataFrame(
        {
            "tdb": [22.0, 25.0, 30.0, 19.0],
            "tr": [22.0, 25.0, 30.0, 19.0],
            "t_rm": [20.0, 20.0, 20.0, 20.0],
            "v": [0.1, 0.1, 0.1, 0.1],
        }
    )
    result = adaptive_ashrae(
        tdb=df["tdb"], tr=df["tr"], t_running_mean=df["t_rm"], v=df["v"]
    )
    categories = np.select(
        [result.acceptability_90 == 1, result.acceptability_80 == 1],
        ["90% Acceptability", "80% Acceptability"],
        default="Outside",
    )

    plot_result = (
        SummaryPlot(df)
        .set_categories(
            categories,
            labels=["90% Acceptability", "80% Acceptability", "Outside"],
            colors=["#6BB3FF", "#B3D9FF", "#D9D9D9"],
        )
        .plot(title="Adaptive comfort distribution")
    )

    # Rows 0-1 (tdb=tr=22/25) satisfy acceptability_90; rows 2-3 (tdb=tr=30/19)
    # satisfy neither band at t_rm=20.
    assert plot_result.percentages.to_dict() == {
        "90% Acceptability": 50.0,
        "80% Acceptability": 0.0,
        "Outside": 50.0,
    }
