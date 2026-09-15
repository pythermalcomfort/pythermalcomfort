"""Tests for AdaptivePlot."""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from pythermalcomfort.models import adaptive_ashrae, adaptive_en
from pythermalcomfort.plots.matplotlib.adaptive import (
    AdaptivePlot,
    AdaptivePlotResult,
    RegionsConfig,
)


@pytest.fixture(autouse=True)
def close_all_figures():
    yield
    plt.close("all")


# ── constructor ────────────────────────────────────────────────────────────


def test_constructor_accepts_adaptive_ashrae() -> None:
    plot = AdaptivePlot(adaptive_ashrae)
    assert plot._standard == "ashrae"


def test_constructor_accepts_adaptive_en() -> None:
    plot = AdaptivePlot(adaptive_en)
    assert plot._standard == "en"


def test_constructor_rejects_non_adaptive_function() -> None:
    def some_other_model():
        pass

    with pytest.raises(ValueError, match="model_func must be one of"):
        AdaptivePlot(some_other_model)


def test_constructor_rejects_non_callable() -> None:
    with pytest.raises(ValueError, match="model_func must be one of"):
        AdaptivePlot("not_a_function")


def test_constructor_sets_default_x_range() -> None:
    plot = AdaptivePlot(adaptive_ashrae)
    assert plot._t_rm_range == (10.0, 33.5)


def test_constructor_sets_no_y_range_by_default() -> None:
    plot = AdaptivePlot(adaptive_ashrae)
    assert plot._y_range is None


# ── set_x_axis ─────────────────────────────────────────────────────────────


def test_set_x_axis_stores_range() -> None:
    plot = AdaptivePlot(adaptive_ashrae).set_x_axis(15.0, 30.0)
    assert plot._t_rm_range == (15.0, 30.0)


def test_set_x_axis_rejects_min_ge_max() -> None:
    with pytest.raises(ValueError, match="min < max"):
        AdaptivePlot(adaptive_ashrae).set_x_axis(30.0, 10.0)


def test_set_x_axis_rejects_equal_values() -> None:
    with pytest.raises(ValueError, match="min < max"):
        AdaptivePlot(adaptive_ashrae).set_x_axis(20.0, 20.0)


def test_set_x_axis_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="numeric"):
        AdaptivePlot(adaptive_ashrae).set_x_axis("a", 30.0)


def test_set_x_axis_returns_self_for_chaining() -> None:
    plot = AdaptivePlot(adaptive_ashrae)
    assert plot.set_x_axis(15.0, 30.0) is plot


# ── set_y_axis ─────────────────────────────────────────────────────────────


def test_set_y_axis_stores_range() -> None:
    plot = AdaptivePlot(adaptive_ashrae).set_y_axis(16.0, 34.0)
    assert plot._y_range == (16.0, 34.0)


def test_set_y_axis_rejects_min_ge_max() -> None:
    with pytest.raises(ValueError, match="min < max"):
        AdaptivePlot(adaptive_ashrae).set_y_axis(34.0, 16.0)


def test_set_y_axis_rejects_non_numeric() -> None:
    with pytest.raises(ValueError, match="numeric"):
        AdaptivePlot(adaptive_ashrae).set_y_axis("a", 34.0)


def test_set_y_axis_returns_self_for_chaining() -> None:
    plot = AdaptivePlot(adaptive_ashrae)
    assert plot.set_y_axis(16.0, 34.0) is plot


# ── set_params ─────────────────────────────────────────────────────────────


def test_set_params_stores_v() -> None:
    plot = AdaptivePlot(adaptive_ashrae).set_params(v=0.5)
    assert plot._v == 0.5


def test_set_params_returns_self_for_chaining() -> None:
    plot = AdaptivePlot(adaptive_ashrae)
    assert plot.set_params(v=0.5) is plot


# ── RegionsConfig ──────────────────────────────────────────────────────────


def test_regions_config_defaults() -> None:
    cfg = RegionsConfig()
    assert cfg.show is None
    assert cfg.labels is None
    assert cfg.colors is None


def test_regions_config_validates_invalid_keys() -> None:
    cfg = RegionsConfig(show=["invalid_key"])
    with pytest.raises(ValueError, match="Invalid band key"):
        cfg._validate("ashrae")


def test_regions_config_rejects_ashrae_keys_on_en() -> None:
    cfg = RegionsConfig(show=["80"])
    with pytest.raises(ValueError, match="Invalid band key"):
        cfg._validate("en")


def test_regions_config_rejects_en_keys_on_ashrae() -> None:
    cfg = RegionsConfig(show=["cat_i"])
    with pytest.raises(ValueError, match="Invalid band key"):
        cfg._validate("ashrae")


def test_regions_config_validates_label_length() -> None:
    cfg = RegionsConfig(show=["90"], labels=["A", "B"])
    with pytest.raises(ValueError, match="labels must have length 1"):
        cfg._validate("ashrae")


def test_regions_config_validates_color_length() -> None:
    cfg = RegionsConfig(show=["90"], colors=["#ff0000", "#00ff00"])
    with pytest.raises(ValueError, match="colors must have length 1"):
        cfg._validate("ashrae")


def test_regions_config_validates_color_values() -> None:
    cfg = RegionsConfig(colors=["not-a-color", "#ff0000"])
    with pytest.raises(ValueError, match="Invalid color"):
        cfg._validate("ashrae")


def test_regions_config_validates_all_bands_label_length() -> None:
    cfg = RegionsConfig(labels=["A", "B", "C"])
    with pytest.raises(ValueError, match="labels must have length 2"):
        cfg._validate("ashrae")


def test_regions_config_valid_en() -> None:
    cfg = RegionsConfig(
        show=["cat_i", "cat_ii"],
        labels=["Best", "OK"],
        colors=["#00ff00", "#ffff00"],
    )
    cfg._validate("en")  # must not raise


# ── set_regions ────────────────────────────────────────────────────────────


def test_set_regions_raw_params() -> None:
    plot = AdaptivePlot(adaptive_ashrae).set_regions(
        show=["90"], labels=["90% Zone"], colors=["#FF6B6B"]
    )
    assert plot._regions_config is not None
    assert list(plot._regions_config.show) == ["90"]


def test_set_regions_with_regions_config() -> None:
    config = RegionsConfig(show=["90"], labels=["90% Zone"], colors=["#FF6B6B"])
    plot = AdaptivePlot(adaptive_ashrae).set_regions(show=config)
    assert plot._regions_config is config


def test_set_regions_rejects_separate_labels_with_config() -> None:
    config = RegionsConfig(show=["90"])
    with pytest.raises(ValueError, match="must not be provided separately"):
        AdaptivePlot(adaptive_ashrae).set_regions(show=config, labels=["X"])


def test_set_regions_rejects_separate_colors_with_config() -> None:
    config = RegionsConfig(show=["90"])
    with pytest.raises(ValueError, match="must not be provided separately"):
        AdaptivePlot(adaptive_ashrae).set_regions(show=config, colors=["#ff0000"])


def test_set_regions_rejects_invalid_keys() -> None:
    with pytest.raises(ValueError, match="Invalid band key"):
        AdaptivePlot(adaptive_ashrae).set_regions(show=["cat_i"])


def test_set_regions_rejects_en_keys_on_ashrae_plot() -> None:
    with pytest.raises(ValueError, match="Invalid band key"):
        AdaptivePlot(adaptive_ashrae).set_regions(show=["cat_i", "cat_ii"])


def test_set_regions_returns_self_for_chaining() -> None:
    plot = AdaptivePlot(adaptive_ashrae)
    assert plot.set_regions(show=["90"]) is plot


def test_set_regions_labels_follow_show_order() -> None:
    config = RegionsConfig(
        show=["90", "80"],
        labels=["Narrow band", "Wide band"],
        colors=["#ff0000", "#00ff00"],
    )
    plot = AdaptivePlot(adaptive_ashrae).set_regions(show=config)
    resolved = {b.spec.key: (b.label, b.color) for b in plot._resolve_bands()}
    assert resolved["90"] == ("Narrow band", "#ff0000")
    assert resolved["80"] == ("Wide band", "#00ff00")


def test_set_regions_labels_follow_show_order_en() -> None:
    plot = AdaptivePlot(adaptive_en).set_regions(
        show=["cat_i", "cat_iii"], labels=["First", "Third"]
    )
    resolved = {b.spec.key: b.label for b in plot._resolve_bands()}
    assert resolved["cat_i"] == "First"
    assert resolved["cat_iii"] == "Third"


# ── plot — ASHRAE ──────────────────────────────────────────────────────────


def test_ashrae_plot_returns_result() -> None:
    result = AdaptivePlot(adaptive_ashrae).plot()
    assert isinstance(result, AdaptivePlotResult)
    assert isinstance(result.fig, Figure)


def test_ashrae_plot_default_fills() -> None:
    result = AdaptivePlot(adaptive_ashrae).plot()
    assert len(result.fills) == 2  # 80% and 90%


def test_ashrae_plot_title() -> None:
    result = AdaptivePlot(adaptive_ashrae).plot(title="ASHRAE Test")
    assert result.ax.get_title() == "ASHRAE Test"


def test_ashrae_plot_default_center_line() -> None:
    result = AdaptivePlot(adaptive_ashrae).plot()
    assert result.center_line is not None


def test_ashrae_plot_no_center_line() -> None:
    result = AdaptivePlot(adaptive_ashrae).plot(show_center_line=False)
    assert result.center_line is None


def test_ashrae_plot_default_legend() -> None:
    result = AdaptivePlot(adaptive_ashrae).plot()
    assert result.legend is not None


def test_ashrae_plot_no_legend() -> None:
    result = AdaptivePlot(adaptive_ashrae).plot(legend=False)
    assert result.legend is None


def test_ashrae_plot_default_legend_labels() -> None:
    result = AdaptivePlot(adaptive_ashrae).plot()
    labels = [t.get_text() for t in result.legend.get_texts()]
    assert "80% Acceptability" in labels
    assert "90% Acceptability" in labels
    assert "Comfort Temperature" in labels


def test_ashrae_plot_show_only_90() -> None:
    result = AdaptivePlot(adaptive_ashrae).set_regions(show=["90"]).plot()
    assert len(result.fills) == 1
    labels = [t.get_text() for t in result.legend.get_texts()]
    assert "90% Acceptability" in labels
    assert "80% Acceptability" not in labels


def test_ashrae_plot_custom_labels() -> None:
    result = AdaptivePlot(adaptive_ashrae).set_regions(labels=["Wide", "Narrow"]).plot()
    labels = [t.get_text() for t in result.legend.get_texts()]
    assert "Wide" in labels
    assert "Narrow" in labels


def test_ashrae_plot_custom_colors() -> None:
    result = (
        AdaptivePlot(adaptive_ashrae).set_regions(colors=["#FF0000", "#00FF00"]).plot()
    )
    assert len(result.fills) == 2


def test_ashrae_plot_center_line_kws() -> None:
    result = AdaptivePlot(adaptive_ashrae).plot(
        center_line_kws={"color": "red", "linewidth": 3.0}
    )
    assert result.center_line.get_color() == "red"
    assert result.center_line.get_linewidth() == 3.0


def test_ashrae_plot_fill_kws() -> None:
    result = AdaptivePlot(adaptive_ashrae).plot(fill_kws={"alpha": 0.3})
    assert len(result.fills) > 0


def test_ashrae_plot_legend_kws() -> None:
    result = AdaptivePlot(adaptive_ashrae).plot(legend_kws={"loc": "upper left"})
    assert result.legend is not None


def test_ashrae_plot_xlabel_ylabel() -> None:
    result = AdaptivePlot(adaptive_ashrae).plot(xlabel="X", ylabel="Y")
    assert result.ax.get_xlabel() == "X"
    assert result.ax.get_ylabel() == "Y"


def test_ashrae_plot_no_xlabel_ylabel() -> None:
    result = AdaptivePlot(adaptive_ashrae).plot(xlabel=None, ylabel=None)
    assert result.ax.get_xlabel() == ""
    assert result.ax.get_ylabel() == ""


def test_ashrae_plot_uses_provided_ax() -> None:
    fig, ax = plt.subplots()
    result = AdaptivePlot(adaptive_ashrae).plot(ax=ax)
    assert result.ax is ax
    assert result.fig is fig


def test_ashrae_plot_set_x_axis_xlim() -> None:
    result = AdaptivePlot(adaptive_ashrae).set_x_axis(15.0, 30.0).plot()
    assert result.ax.get_xlim() == pytest.approx((15.0, 30.0))


def test_ashrae_plot_set_y_axis_ylim() -> None:
    result = AdaptivePlot(adaptive_ashrae).set_y_axis(16.0, 34.0).plot()
    assert result.ax.get_ylim() == pytest.approx((16.0, 34.0))


def test_ashrae_plot_cooling_effect_with_high_v() -> None:
    result_low_v = AdaptivePlot(adaptive_ashrae).set_params(v=0.1).plot()
    result_high_v = AdaptivePlot(adaptive_ashrae).set_params(v=1.5).plot()
    assert len(result_low_v.fills) == len(result_high_v.fills)


# ── plot — EN ──────────────────────────────────────────────────────────────


def test_en_plot_returns_result() -> None:
    result = AdaptivePlot(adaptive_en).plot()
    assert isinstance(result, AdaptivePlotResult)


def test_en_plot_default_fills() -> None:
    result = AdaptivePlot(adaptive_en).plot()
    assert len(result.fills) == 3  # Cat I, II, III


def test_en_plot_default_legend_labels() -> None:
    result = AdaptivePlot(adaptive_en).plot()
    labels = [t.get_text() for t in result.legend.get_texts()]
    assert "Category I" in labels
    assert "Category II" in labels
    assert "Category III" in labels


def test_en_plot_show_cat_i_only() -> None:
    result = AdaptivePlot(adaptive_en).set_regions(show=["cat_i"]).plot()
    assert len(result.fills) == 1
    labels = [t.get_text() for t in result.legend.get_texts()]
    assert "Category I" in labels
    assert "Category II" not in labels


def test_en_plot_custom_labels() -> None:
    result = AdaptivePlot(adaptive_en).set_regions(labels=["Best", "OK", "Min"]).plot()
    labels = [t.get_text() for t in result.legend.get_texts()]
    assert "Best" in labels
    assert "OK" in labels
    assert "Min" in labels


def test_en_plot_show_two_bands() -> None:
    result = (
        AdaptivePlot(adaptive_en)
        .set_regions(
            show=["cat_i", "cat_ii"],
            labels=["Strict", "Normal"],
            colors=["#00FF00", "#FFFF00"],
        )
        .plot()
    )
    assert len(result.fills) == 2
    labels = [t.get_text() for t in result.legend.get_texts()]
    assert "Strict" in labels
    assert "Normal" in labels


# ── RegionsConfig reuse ────────────────────────────────────────────────────


def test_regions_config_reuse_across_plots() -> None:
    config = RegionsConfig(show=["90"], labels=["Comfort"], colors=["#FF6B6B"])

    result1 = AdaptivePlot(adaptive_ashrae).set_regions(show=config).plot()
    result2 = AdaptivePlot(adaptive_ashrae).set_regions(show=config).plot()

    labels1 = [t.get_text() for t in result1.legend.get_texts()]
    labels2 = [t.get_text() for t in result2.legend.get_texts()]
    assert labels1 == labels2
    assert "Comfort" in labels1


# ── fluent chaining ────────────────────────────────────────────────────────


def test_full_chain_ashrae() -> None:
    result = (
        AdaptivePlot(adaptive_ashrae)
        .set_x_axis(12.0, 32.0)
        .set_y_axis(16.0, 34.0)
        .set_params(v=0.8)
        .set_regions(show=["90"], labels=["90% Zone"], colors=["#6BB3FF"])
        .plot(title="Full Chain Test")
    )
    assert isinstance(result, AdaptivePlotResult)
    assert result.ax.get_title() == "Full Chain Test"
    assert len(result.fills) == 1
