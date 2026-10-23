"""Tests for shared Matplotlib font-family configuration."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.text import Text

from pythermalcomfort.models import adaptive_ashrae, pmv_ppd_iso
from pythermalcomfort.plots.matplotlib import (
    AdaptivePlot,
    PsychrometricPlot,
    SummaryPlot,
    ThresholdPlot,
)


@pytest.fixture(autouse=True)
def close_all_figures():
    """Close Matplotlib figures created by each test."""
    yield
    plt.close("all")


def _adaptive_plot() -> AdaptivePlot:
    """Return a minimally configured adaptive plot."""
    return AdaptivePlot(adaptive_ashrae)


def _threshold_plot() -> ThresholdPlot:
    """Return a minimally configured threshold plot."""
    return (
        ThresholdPlot(pmv_ppd_iso)
        .set_params(vr=0.1, met=1.2, clo=0.5, tr=25.0)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("rh", 30.0, 70.0, resolution=10.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )


def _psychrometric_plot() -> PsychrometricPlot:
    """Return a minimally configured psychrometric plot."""
    return (
        PsychrometricPlot(pmv_ppd_iso)
        .set_params(vr=0.1, met=1.2, clo=0.5, tr=25.0)
        .set_x_axis("tdb", 20.0, 30.0, resolution=1.0)
        .set_y_axis("hr", 0.0, 20.0, resolution=2.0)
        .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    )


def _summary_plot() -> SummaryPlot:
    """Return a minimally configured summary plot."""
    data = pd.DataFrame({"pmv": [-0.8, 0.0, 0.7]})
    return SummaryPlot(data).set_regions(output="pmv", thresholds=[-0.5, 0.5])


@pytest.mark.parametrize(
    "plot_factory",
    [_adaptive_plot, _threshold_plot, _psychrometric_plot, _summary_plot],
)
@pytest.mark.parametrize(
    ("font_family", "expected_family"),
    [
        ("DejaVu Serif", ["DejaVu Serif"]),
        (["monospace", "sans-serif"], ["monospace", "sans-serif"]),
    ],
)
def test_per_plot_font_family(
    plot_factory: Callable,
    font_family: str | list[str],
    expected_family: list[str],
) -> None:
    """Apply a per-plot font without leaking Matplotlib configuration."""
    _, ax = plt.subplots()
    annotation = ax.text(
        0.5,
        0.5,
        "Existing annotation",
        fontfamily="serif",
    )
    original_family = list(matplotlib.rcParams["font.family"])

    result = plot_factory().plot(
        ax=ax,
        title="Font test",
        font_family=font_family,
    )
    result.fig.canvas.draw()

    visible_text = [
        text
        for text in result.ax.findobj(match=Text)
        if text is not annotation and text.get_text().strip()
    ]
    assert visible_text
    assert all(text.get_fontfamily() == expected_family for text in visible_text)
    if plot_factory is _summary_plot:
        assert annotation not in result.ax.texts
    else:
        assert annotation in result.ax.texts
        assert annotation.get_fontfamily() == ["serif"]
    assert list(matplotlib.rcParams["font.family"]) == original_family


def test_global_matplotlib_font_family_is_preserved() -> None:
    """Inherit the global font family without changing it after plotting."""
    with plt.rc_context({"font.family": "serif"}):
        result = _adaptive_plot().plot(title="Global font")
        assert result.ax.title.get_fontfamily() == ["serif"]
        assert result.ax.xaxis.label.get_fontfamily() == ["serif"]

    assert result.ax.title.get_fontfamily() == ["serif"]
