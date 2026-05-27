"""Class-based summary plotting for adaptive comfort regions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.legend import Legend
from matplotlib.patches import Patch

from pythermalcomfort.models.adaptive_ashrae import adaptive_ashrae
from pythermalcomfort.models.adaptive_en import adaptive_en
from pythermalcomfort.plots.matplotlib._base import BasePlot
from pythermalcomfort.plots.matplotlib._shared import (
    _PYTHERMALCOMFORT_RC,
    _PlotDefaults,
    _resolve_region_colors,
)
from pythermalcomfort.plots.matplotlib.threshold_summary import (
    SummaryPlotResult,
    _apply_compact_layout,
    _default_legend_ncol,
    _ensure_title_legend_spacing,
    _plot_summary,
    _prepare_axis,
    _should_show_region_labels,
    _summary_figsize,
    _validate_dataframe,
)

# ── standard configuration ─────────────────────────────────────────────────

_ADAPTIVE_STANDARDS: dict[str, dict[str, Any]] = {
    "acceptability": {
        "model_func": adaptive_ashrae,
        # ordered widest → narrowest (90 % band ⊂ 80 % band)
        "band_order": [80, 90],
        "default_labels": {
            80: "80% Acceptability",
            90: "90% Acceptability",
        },
    },
    "acceptability_cat": {
        "model_func": adaptive_en,
        # ordered widest → narrowest (Cat I ⊂ Cat II ⊂ Cat III)
        "band_order": ["iii", "ii", "i"],
        "default_labels": {
            "iii": "Category III",
            "ii": "Category II",
            "i": "Category I",
        },
    },
}

_REQUIRED_INPUT_COLUMNS: list[str] = ["tdb", "tr", "t_running_mean", "v"]

# ── internal resolved container ────────────────────────────────────────────


@dataclass
class _AdaptiveRegionConfig:
    """Fully-resolved adaptive region configuration (internal use only)."""

    output_prefix: str
    standard_key: str
    band_keys: list  # sorted widest → narrowest
    labels: list[str]  # length = len(band_keys) + 1
    colors: list[str]  # length = len(band_keys) + 1


# ── public API ─────────────────────────────────────────────────────────────


class AdaptiveSummary(BasePlot):
    """Build and render an adaptive comfort summary from input measurements.

    Runs the appropriate adaptive comfort model (ASHRAE 55 or EN 16798) on
    every row of the input DataFrame, assigns each row to the narrowest
    comfort band it satisfies, and displays the result as a stacked bar
    chart identical in style to :class:`ThresholdSummary`.

    The DataFrame must contain columns ``tdb``, ``tr``,
    ``t_running_mean``, and ``v``.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.plots.matplotlib import AdaptiveSummary

        # ASHRAE 55
        result = (
            AdaptiveSummary(df)
            .set_regions(output="acceptability", thresholds=[80, 90])
            .plot(title="Adaptive Comfort Summary (ASHRAE 55)")
        )

        # EN 16798
        result = (
            AdaptiveSummary(df)
            .set_regions(
                output="acceptability_cat",
                thresholds=["i", "ii", "iii"],
            )
            .plot(title="Adaptive Comfort Summary (EN 16798)")
        )
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize an adaptive summary builder from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame containing measurement columns (``tdb``, ``tr``,
            ``t_running_mean``, ``v``).

        Raises
        ------
        TypeError
            If *df* is not a pandas DataFrame.
        ValueError
            If *df* is empty.
        """
        super().__init__()
        _validate_dataframe(df)
        self._df = df
        self._adaptive_config: _AdaptiveRegionConfig | None = None
        self._model_params: dict[str, Any] = {}

    # ── configuration ──────────────────────────────────────────────────────

    def set_regions(
        self,
        *,
        output: str,
        thresholds: Sequence,
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
    ) -> AdaptiveSummary:
        """Configure which adaptive comfort bands to display.

        Parameters
        ----------
        output : str
            Selects the adaptive model via naming convention:

            - ``"acceptability"`` → ASHRAE 55
              (valid thresholds: ``80``, ``90``)
            - ``"acceptability_cat"`` → EN 16798
              (valid thresholds: ``"i"``, ``"ii"``, ``"iii"``)

        thresholds : sequence
            Band keys to display.  Order does not matter — bands are always
            arranged from widest to narrowest internally.  The resulting
            number of regions is ``len(thresholds) + 1`` (one extra region
            for data points outside all selected bands).
        labels : sequence of str, optional
            Custom region labels.  Must have length ``len(thresholds) + 1``.
        colors : sequence of str, optional
            Custom region colors.  Same length rule as *labels*.

        Returns
        -------
        AdaptiveSummary
            Self, to support method chaining.

        Raises
        ------
        TypeError
            If *output* is not a string.
        ValueError
            If *output* is unrecognised, thresholds are invalid or empty,
            required DataFrame columns are missing, or labels/colors have
            the wrong length.
        """
        if not isinstance(output, str):
            raise TypeError("output must be a string.")

        output_name = output.strip()
        if output_name not in _ADAPTIVE_STANDARDS:
            valid = ", ".join(f"'{k}'" for k in sorted(_ADAPTIVE_STANDARDS))
            msg = f"output must be one of {valid}, got '{output_name}'."
            raise ValueError(msg)

        std_cfg = _ADAPTIVE_STANDARDS[output_name]
        band_order: list = std_cfg["band_order"]
        valid_keys = set(band_order)

        # ── validate thresholds ────────────────────────────────────────────
        if not thresholds:
            raise ValueError("thresholds must contain at least one band key.")

        normalized: list = []
        for t in thresholds:
            if isinstance(band_order[0], int):
                try:
                    normalized.append(int(t))
                except (TypeError, ValueError) as exc:
                    msg = f"ASHRAE thresholds must be integers (80 or 90), got {t!r}."
                    raise ValueError(msg) from exc
            else:
                normalized.append(str(t).strip().lower())

        invalid = [t for t in normalized if t not in valid_keys]
        if invalid:
            msg = f"Invalid threshold(s): {invalid}. Valid keys: {band_order}"
            raise ValueError(msg)

        # deduplicate, then sort widest → narrowest
        seen: set = set()
        unique: list = []
        for t in normalized:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        band_keys = sorted(unique, key=lambda k: band_order.index(k))

        # ── validate DataFrame columns ─────────────────────────────────────
        missing = sorted(
            col for col in _REQUIRED_INPUT_COLUMNS if col not in self._df.columns
        )
        if missing:
            msg = (
                f"DataFrame is missing required column(s): "
                f"{', '.join(missing)}. "
                f"Required columns: {', '.join(_REQUIRED_INPUT_COLUMNS)}."
            )
            raise ValueError(msg)

        # ── build labels ───────────────────────────────────────────────────
        n_regions = len(band_keys) + 1
        if labels is not None:
            if len(labels) != n_regions:
                msg = f"labels must have length {n_regions} (got {len(labels)})."
                raise ValueError(msg)
            region_labels = [str(lbl) for lbl in labels]
        else:
            default_labels = std_cfg["default_labels"]
            region_labels = ["Not Acceptable"]
            for key in band_keys:
                region_labels.append(default_labels[key])

        # ── build colors ───────────────────────────────────────────────────
        region_colors = _resolve_region_colors(n_regions=n_regions, colors=colors)

        self._adaptive_config = _AdaptiveRegionConfig(
            output_prefix=output_name,
            standard_key=output_name,
            band_keys=band_keys,
            labels=region_labels,
            colors=region_colors,
        )
        return self

    def set_params(self, **kwargs: Any) -> AdaptiveSummary:
        """Set additional parameters forwarded to the adaptive model call.

        Common parameters include ``units`` (``"SI"`` or ``"IP"``) and
        ``limit_inputs`` (bool).  Any keyword accepted by the underlying
        model function (``adaptive_ashrae`` or ``adaptive_en``) can be
        passed here.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded unchanged to the model function.

        Returns
        -------
        AdaptiveSummary
            Self, to support method chaining.
        """
        self._model_params.update(kwargs)
        return self

    # ── internal evaluation ────────────────────────────────────────────────

    def _evaluate_and_categorize(self) -> pd.Series:
        """Run the model and return region percentages.

        Each row is assigned to the **narrowest** comfort band it satisfies.
        Rows outside all selected bands are labelled *Not Acceptable*.

        Returns
        -------
        Series
            Percentage share per region, indexed by region label.
        """
        cfg = self._adaptive_config
        std_cfg = _ADAPTIVE_STANDARDS[cfg.standard_key]
        model_func = std_cfg["model_func"]

        try:
            result = model_func(
                tdb=self._df["tdb"].values,
                tr=self._df["tr"].values,
                t_running_mean=self._df["t_running_mean"].values,
                v=self._df["v"].values,
                **self._model_params,
            )
        except Exception as exc:
            msg = f"Adaptive model evaluation failed: {exc}"
            raise ValueError(msg) from exc

        n = len(self._df)
        categories = np.zeros(n, dtype=int)

        # iterate widest → narrowest; narrower bands overwrite wider ones
        for i, key in enumerate(cfg.band_keys):
            attr_name = f"{cfg.output_prefix}_{key}"
            try:
                mask = np.asarray(getattr(result, attr_name), dtype=bool)
            except AttributeError:
                msg = f"Model result has no attribute '{attr_name}'."
                raise ValueError(msg) from None
            categories[mask] = i + 1

        n_regions = len(cfg.band_keys) + 1
        counts = np.bincount(categories, minlength=n_regions)[:n_regions]
        pcts = np.round(counts / n * 100, 1)
        return pd.Series(pcts, index=pd.Index(cfg.labels))

    # ── plotting ───────────────────────────────────────────────────────────

    def plot(
        self,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        vertical: bool = False,
        legend: bool = True,
        bar_kws: Mapping[str, Any] | None = None,
        legend_kws: Mapping[str, Any] | None = None,
    ) -> SummaryPlotResult:
        """Render an adaptive comfort summary plot.

        Parameters
        ----------
        ax : Axes, optional
            Existing axis to draw on.  If ``None``, a new figure/axis is
            created with a compact default size.
        title : str, optional
            Optional chart title.
        vertical : bool
            If ``True``, render a vertical stacked bar; otherwise horizontal.
        legend : bool
            Whether to draw a colour-coded legend above the bar.
        bar_kws : dict, optional
            Keyword overrides forwarded to ``ax.bar`` / ``ax.barh``.
        legend_kws : dict, optional
            Keyword overrides forwarded to ``ax.legend``.

        Returns
        -------
        SummaryPlotResult
            Result with figure, axis, percentages, artists, and legend.

        Raises
        ------
        ValueError
            If regions are not configured or model evaluation fails.
        """
        with mpl.rc_context(_PYTHERMALCOMFORT_RC):
            if self._adaptive_config is None:
                raise ValueError(
                    "Regions are not set. Call set_regions(...) before plot(...)."
                )

            cfg = self._adaptive_config
            show_region_labels = _should_show_region_labels(
                legend=legend,
                region_labels=cfg.labels,
            )
            created_figure = ax is None

            if created_figure:
                fig, ax = plt.subplots(
                    figsize=_summary_figsize(
                        vertical=vertical,
                        legend=legend,
                        show_region_labels=show_region_labels,
                    )
                )
            else:
                fig = ax.figure

            percentages = self._evaluate_and_categorize()

            _prepare_axis(ax)
            artists = _plot_summary(
                ax,
                vertical=vertical,
                region_percentages=percentages,
                region_labels=cfg.labels,
                region_colors=cfg.colors,
                show_region_labels=show_region_labels,
                bar_kws=bar_kws or {},
            )

            legend_artist: Legend | None = None
            if legend:
                lg_opts = dict(legend_kws or {})
                lg_opts.setdefault("loc", "lower center")
                lg_opts.setdefault(
                    "bbox_to_anchor",
                    _PlotDefaults.legend_bbox_to_anchor_with_title
                    if title is not None
                    else _PlotDefaults.Threshold.legend_bbox_to_anchor,
                )
                lg_opts.setdefault(
                    "ncol",
                    _default_legend_ncol(vertical=vertical, n_labels=len(cfg.labels)),
                )
                handles = [
                    Patch(facecolor=color, label=label)
                    for label, color in zip(cfg.labels, cfg.colors, strict=False)
                ]
                legend_artist = ax.legend(handles=handles, **lg_opts)

            if title is not None:
                ax.set_title(
                    title,
                    y=_PlotDefaults.title_y_with_legend if legend else None,
                )

            if created_figure:
                _apply_compact_layout(fig)

            _ensure_title_legend_spacing(
                fig,
                ax,
                legend_artist,
                adjust_layout=created_figure,
            )

            return SummaryPlotResult(
                fig=fig,
                ax=ax,
                percentages=percentages,
                artists=artists,
                legend=legend_artist,
            )
