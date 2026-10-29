"""Category regions of model outputs, shared by the plotting classes.

Each entry links a model function to the numeric output it classifies and to the
:class:`_CategoryBins` object the model itself uses, so the thresholds, labels and
interval closure of a plot always match the model's own categories.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pythermalcomfort._internal.validation import (
    HEAT_INDEX_RISK_CATEGORIES,
    _CategoryBins,
)
from pythermalcomfort.models.discomfort_index import (
    DISCOMFORT_CATEGORIES,
    discomfort_index,
)
from pythermalcomfort.models.heat_index_rothfusz import heat_index_rothfusz
from pythermalcomfort.models.heat_index_schoen import heat_index_schoen
from pythermalcomfort.models.pmv_ppd_ashrae import TSV_CATEGORIES as _PMV_ASHRAE_TSV
from pythermalcomfort.models.pmv_ppd_ashrae import pmv_ppd_ashrae
from pythermalcomfort.models.pmv_ppd_iso import TSV_CATEGORIES as _PMV_ISO_TSV
from pythermalcomfort.models.pmv_ppd_iso import pmv_ppd_iso
from pythermalcomfort.models.utci import STRESS_CATEGORIES as _UTCI_STRESS
from pythermalcomfort.models.utci import utci


@dataclass(frozen=True)
class _ModelRegions:
    """Numeric output of a model and the category bins the model classifies it with.

    Attributes
    ----------
    output : str
        Attribute of the model's return dataclass holding the numeric value,
        e.g. ``"utci"``.
    category_output : str
        Attribute holding the model's own category label, e.g. ``"stress_category"``.
    bins : _CategoryBins
        The bins the model uses. Their edges are in SI units.
    """

    output: str
    category_output: str
    bins: _CategoryBins


_MODEL_REGIONS: Mapping[Callable[..., Any], _ModelRegions] = MappingProxyType(
    {
        pmv_ppd_iso: _ModelRegions(
            output="pmv",
            category_output="tsv",
            bins=_PMV_ISO_TSV,
        ),
        pmv_ppd_ashrae: _ModelRegions(
            output="pmv",
            category_output="tsv",
            bins=_PMV_ASHRAE_TSV,
        ),
        utci: _ModelRegions(
            output="utci",
            category_output="stress_category",
            bins=_UTCI_STRESS,
        ),
        heat_index_rothfusz: _ModelRegions(
            output="hi",
            category_output="stress_category",
            bins=HEAT_INDEX_RISK_CATEGORIES,
        ),
        heat_index_schoen: _ModelRegions(
            output="hi",
            category_output="stress_category",
            bins=HEAT_INDEX_RISK_CATEGORIES,
        ),
        discomfort_index: _ModelRegions(
            output="di",
            category_output="discomfort_condition",
            bins=DISCOMFORT_CATEGORIES,
        ),
    }
)


def _get_model_regions(model: Callable[..., Any]) -> _ModelRegions:
    """Return the category regions of ``model``.

    Raises
    ------
    ValueError
        If no category regions are defined for ``model``.
    """
    try:
        return _MODEL_REGIONS[model]
    except (KeyError, TypeError):
        supported = ", ".join(func.__name__ for func in _MODEL_REGIONS)
        name = getattr(model, "__name__", repr(model))
        msg = f"No category regions are defined for {name}. Supported models: {supported}."
        raise ValueError(msg) from None
