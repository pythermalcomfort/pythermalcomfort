"""Regression tests for standard SummaryPlot region presets."""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pythermalcomfort.models import (
    heat_index_rothfusz,
    heat_index_schoen,
    pmv_ppd_ashrae,
    pmv_ppd_iso,
    utci,
)
from pythermalcomfort.plots.matplotlib import SummaryPlot
from pythermalcomfort.plots.matplotlib.summary import _compute_region_percentages


@pytest.fixture(autouse=True)
def close_figures():
    """Close figures after each preset test."""
    yield
    plt.close("all")


@pytest.mark.parametrize(
    ("preset", "output", "edges", "labels", "right"),
    [
        (
            "pmv_iso",
            "pmv",
            [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
            [
                "Cold",
                "Cool",
                "Slightly Cool",
                "Neutral",
                "Slightly Warm",
                "Warm",
                "Hot",
            ],
            False,
        ),
        (
            "pmv_ashrae",
            "pmv",
            [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5],
            [
                "Cold",
                "Cool",
                "Slightly Cool",
                "Neutral",
                "Slightly Warm",
                "Warm",
                "Hot",
            ],
            True,
        ),
        (
            "utci",
            "utci",
            [-40, -27, -13, 0, 9, 26, 32, 38, 46],
            [
                "extreme cold stress",
                "very strong cold stress",
                "strong cold stress",
                "moderate cold stress",
                "slight cold stress",
                "no thermal stress",
                "moderate heat stress",
                "strong heat stress",
                "very strong heat stress",
                "extreme heat stress",
            ],
            True,
        ),
        (
            "heat_index",
            "hi",
            [27, 32, 41, 54],
            ["no risk", "caution", "extreme caution", "danger", "extreme danger"],
            True,
        ),
    ],
)
def test_preset_classifies_every_boundary(preset, output, edges, labels, right):
    """Check each exact boundary and its nearest floating-point neighbors."""
    values = [
        value
        for edge in edges
        for value in [np.nextafter(edge, -np.inf), edge, np.nextafter(edge, np.inf)]
    ]
    frame = pd.DataFrame({output: values})
    chart = SummaryPlot(frame).set_regions(preset=preset)
    config = chart._region_config
    assert list(config.thresholds) == edges
    assert list(config.labels) == labels
    assert len(config.colors) == len(labels)
    for i in range(len(edges)):
        for offset, expected in enumerate([i, i if right else i + 1, i + 1]):
            percentages = _compute_region_percentages(
                frame.iloc[[3 * i + offset]],
                output_column=config.output_name,
                levels=config.thresholds,
                region_labels=config.labels,
                right=chart._regions_right,
            )
            assert percentages.iloc[expected] == 100
            assert percentages.sum() == 100
    result = chart.plot()
    result.fig.canvas.draw()
    assert result.percentages.index.tolist() == labels
    assert [text.get_text() for text in result.legend.get_texts()] == labels


@pytest.mark.parametrize("preset", ["pmv_iso", "pmv_ashrae", "utci", "heat_index"])
def test_preset_agrees_with_model_categories(preset):
    """Compare summary shares with categories returned by actual models."""
    temperatures = np.linspace(-30, 50, 100)
    if preset in ("pmv_iso", "pmv_ashrae"):
        model = pmv_ppd_iso if preset == "pmv_iso" else pmv_ppd_ashrae
        result = model(
            tdb=temperatures,
            tr=temperatures,
            vr=0.1,
            rh=50,
            met=1.2,
            clo=0.5,
            limit_inputs=False,
            round_output=False,
        )
        output, values, categories = "pmv", result.pmv, result.tsv
    elif preset == "utci":
        result = utci(tdb=temperatures, tr=temperatures, v=1, rh=50, round_output=False)
        output, values, categories = "utci", result.utci, result.stress_category
    else:
        result = heat_index_rothfusz(
            tdb=np.linspace(27, 45, 100), rh=70, round_output=False
        )
        output, values, categories = "hi", result.hi, result.stress_category
    frame = pd.DataFrame({output: values, "category": categories}).dropna()
    plotted = SummaryPlot(frame).set_regions(preset=preset).plot()
    expected = frame["category"].value_counts(normalize=True).mul(100).round(1)
    expected = expected.reindex(plotted.percentages.index, fill_value=0)
    pd.testing.assert_series_equal(plotted.percentages, expected, check_names=False)


@pytest.mark.parametrize(
    ("model", "tdb"),
    [(heat_index_rothfusz, 28.68), (heat_index_schoen, 28.794)],
)
def test_heat_index_rounding_near_boundary(model, tdb):
    """Unrounded Heat Index preserves model categories near a rounding boundary."""
    raw = model(tdb=tdb, rh=70, round_output=False)
    rounded = model(tdb=tdb, rh=70)
    assert 32 < raw.hi < 32.05
    assert rounded.hi == 32
    assert raw.stress_category == rounded.stress_category == "extreme caution"

    raw_summary = (
        SummaryPlot(pd.DataFrame({"hi": [float(raw.hi)]}))
        .set_regions(preset="heat_index")
        .plot()
    )
    rounded_summary = (
        SummaryPlot(pd.DataFrame({"hi": [float(rounded.hi)]}))
        .set_regions(preset="heat_index")
        .plot()
    )
    assert raw_summary.percentages.loc["extreme caution"] == 100
    assert rounded_summary.percentages.loc["caution"] == 100


def test_preset_overrides_and_switching():
    """Preserve custom styling and reset preset boundaries on reconfiguration."""
    frame = pd.DataFrame({"measured": [27, 32, 41, 54]})
    labels = ["A", "B", "C", "D", "E"]
    colors = ["red", "green", "blue", "black", "white"]
    chart = SummaryPlot(frame).set_regions(
        preset="heat_index",
        output="measured",
        labels=labels,
        colors=colors,
    )
    result = chart.plot()
    assert result.percentages.tolist() == [25, 25, 25, 25, 0]
    assert result.percentages.index.tolist() == labels
    assert chart._region_config.colors == colors
    chart.set_regions(output="measured", thresholds=[32])
    assert chart.plot().percentages.tolist() == [25, 75]
    chart.set_regions(preset="heat_index", output="measured", labels=[])
    assert chart._region_config.labels == [""] * 5
    chart.set_categories(["A"] * 4, labels=["A"], colors=["red"])
    assert chart.plot().percentages.tolist() == [100]
    chart.set_regions(preset="heat_index", output="measured")
    assert chart.plot().percentages.tolist() == [25, 25, 25, 25, 0]


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"preset": "unknown"}, ValueError, "Unknown preset"),
        ({"preset": []}, TypeError, "preset must be a string"),
        ({"preset": "pmv_iso", "thresholds": [0]}, ValueError, "cannot be combined"),
        ({}, ValueError, "required without a preset"),
        ({"output": "pmv"}, ValueError, "required without a preset"),
        ({"preset": "utci"}, ValueError, "was not found"),
        ({"preset": "pmv_iso", "output": ""}, ValueError, "non-empty"),
        ({"preset": "pmv"}, ValueError, "Choose from pmv_iso, pmv_ashrae"),
        ({"preset": "pmv_iso", "labels": ["A"]}, ValueError, "labels must have length"),
        (
            {"preset": "pmv_iso", "colors": ["red"]},
            ValueError,
            "colors must have length",
        ),
        ({"preset": "pmv_iso", "colors": ["bad"] * 7}, ValueError, "Invalid color"),
    ],
)
def test_invalid_preset_configuration(kwargs, error, message):
    """Reject ambiguous or invalid preset configurations before rendering."""
    with pytest.raises(error, match=message):
        SummaryPlot(pd.DataFrame({"pmv": [0.0]})).set_regions(**kwargs)


@pytest.mark.parametrize("value", [np.nan, np.inf, "bad"])
def test_presets_reject_invalid_data(value):
    """Keep existing finite numeric input validation for preset users."""
    with pytest.raises(ValueError, match="non-numeric, non-finite, or missing"):
        SummaryPlot(pd.DataFrame({"pmv": [value]})).set_regions(preset="pmv_iso")


def test_pmv_presets_differ_only_on_a_boundary():
    """A PMV of exactly 0.5 is Slightly Warm under ISO 7730 but Neutral under ASHRAE 55."""
    frame = pd.DataFrame({"pmv": [0.5]})
    iso = SummaryPlot(frame).set_regions(preset="pmv_iso").plot()
    ashrae = SummaryPlot(frame).set_regions(preset="pmv_ashrae").plot()
    assert iso.percentages.loc["Slightly Warm"] == 100
    assert ashrae.percentages.loc["Neutral"] == 100
