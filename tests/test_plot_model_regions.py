"""Tests for the model category registry used by the plotting classes."""

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from pythermalcomfort._internal.validation import _CategoryBins
from pythermalcomfort.models import (
    discomfort_index,
    heat_index_rothfusz,
    heat_index_schoen,
    pmv_ppd_ashrae,
    pmv_ppd_iso,
    utci,
)
from pythermalcomfort.plots.matplotlib._model_regions import (
    _MODEL_REGIONS,
    _get_model_regions,
)

_rng = np.random.default_rng(0)
_N = 500
_TDB = _rng.uniform(10, 45, _N)
_RH = _rng.uniform(5, 95, _N)

# Unrounded model calls, so the numeric output is exactly the value the model classified.
_MODEL_CALLS: dict[Callable[..., Any], dict[str, Any]] = {
    pmv_ppd_iso: dict(
        tdb=_TDB,
        tr=_TDB,
        vr=0.1,
        rh=_RH,
        met=1.2,
        clo=0.5,
        limit_inputs=False,
        round_output=False,
    ),
    pmv_ppd_ashrae: dict(
        tdb=_TDB,
        tr=_TDB,
        vr=0.1,
        rh=_RH,
        met=1.2,
        clo=0.5,
        limit_inputs=False,
        round_output=False,
    ),
    utci: dict(
        tdb=_TDB - 20,
        tr=_TDB,
        v=1.0,
        rh=_RH,
        limit_inputs=False,
        round_output=False,
    ),
    heat_index_rothfusz: dict(tdb=_TDB, rh=_RH, limit_inputs=False, round_output=False),
    heat_index_schoen: dict(tdb=_TDB, rh=_RH, round_output=False),
    discomfort_index: dict(tdb=_TDB, rh=_RH),
}


# discomfort_index has no round_output and always reports di rounded to 0.1, while it
# classifies the unrounded value, so points within 0.05 of an edge can differ.
_ALWAYS_ROUNDED = {discomfort_index}


def test_every_registered_model_has_a_test_call() -> None:
    assert set(_MODEL_CALLS) == set(_MODEL_REGIONS)


@pytest.mark.parametrize("model", list(_MODEL_REGIONS), ids=lambda f: f.__name__)
def test_registry_reproduces_the_model_categories(model: Callable[..., Any]) -> None:
    """The registry classifies the model output exactly as the model does."""
    regions = _get_model_regions(model)
    result = model(**_MODEL_CALLS[model])
    value = np.asarray(getattr(result, regions.output))
    expected = np.asarray(getattr(result, regions.category_output)).astype(str)
    actual = regions.bins.map(value).astype(str)
    if model in _ALWAYS_ROUNDED:
        edges = np.array(regions.bins.thresholds)
        distance = np.min(np.abs(value[:, None] - edges), axis=1)
        far_from_edges = distance > 0.05
        actual, expected = actual[far_from_edges], expected[far_from_edges]
    np.testing.assert_array_equal(actual, expected)
    # the grid must cover several categories, otherwise the test proves little
    assert len(set(expected)) >= 3


@pytest.mark.parametrize(
    ("model", "right"),
    [
        (pmv_ppd_iso, False),
        (pmv_ppd_ashrae, True),
        (utci, True),
        (heat_index_rothfusz, True),
        (heat_index_schoen, True),
        (discomfort_index, False),
    ],
    ids=lambda x: getattr(x, "__name__", str(x)),
)
def test_values_on_an_edge_follow_the_model_closure(
    model: Callable[..., Any], right: bool
) -> None:
    bins = _get_model_regions(model).bins
    assert bins.right is right
    for i, edge in enumerate(bins.thresholds):
        expected = bins.labels[i] if right else bins.labels[i + 1]
        assert bins.map(edge) == expected


def test_category_bins_properties() -> None:
    bins = _CategoryBins(edges={0.0: "low", 10.0: "mid", 100.0: "high"}, right=True)
    assert bins.thresholds == (0.0, 10.0)
    assert bins.labels == ("low", "mid", "high")
    assert bins.map([-5, 0, 5, 10, 50]).tolist() == ["low", "low", "mid", "mid", "high"]
    assert np.isnan(bins.map(200))
    with pytest.raises(TypeError):
        bins.edges[5.0] = "new"  # type: ignore[index]
    hash(bins)  # usable as a dict key or in a set


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="No category regions are defined for max"):
        _get_model_regions(max)
