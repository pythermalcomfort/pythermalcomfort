"""Class-based summary plotting for threshold regions."""

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

from pythermalcomfort.plots.matplotlib._base import BasePlot
from pythermalcomfort.plots.matplotlib._shared import (
    _PYTHERMALCOMFORT_RC,
    BasePlotResult,
    _is_light_color,
    _PlotDefaults,
)
from pythermalcomfort.plots.matplotlib.adaptive import (
    _ACCEPTABILITY_FIELDS,
    _MODEL_TO_STANDARD,
    _STANDARD_CONFIGS,
    RegionsConfig,
    _BandSpec,
)

# ── result container ───────────────────────────────────────────────────────


@dataclass
class SummaryPlotResult(BasePlotResult):
    """Container with handles returned by :meth:`SummaryPlot.plot`.

    Attributes
    ----------
    fig : Figure
        Matplotlib figure containing the summary plot.
    ax : Axes
        Matplotlib axis containing the summary plot.
    percentages : Series
        Percentage share per region, indexed by region label.
    artists : list
        List of rendered bar and text artists.
    legend : Legend or None
        Legend artist if ``legend=True``, otherwise ``None``.
    """

    percentages: pd.Series
    artists: list[Any]
    legend: Legend | None


# ── validation helpers ─────────────────────────────────────────────────────


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate input DataFrame for summary plotting."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("df must contain at least one row.")


def _validate_output_column(df: pd.DataFrame, output: str) -> str:
    """Validate output column name and ensure it exists."""
    if not isinstance(output, str):
        raise TypeError("output must be a string.")

    output_name = output.strip()
    if not output_name:
        raise ValueError("output must be a non-empty string.")

    if output_name not in df.columns:
        msg = f"output column '{output_name}' was not found in the DataFrame."
        raise ValueError(msg)

    return output_name


def _validate_output_values(df: pd.DataFrame, output_column: str) -> None:
    """Ensure output column contains numeric finite values only.

    Raises rather than silently dropping rows so callers are aware of missing
    data and can decide how to handle it before plotting.
    """
    numeric_values = pd.to_numeric(df[output_column], errors="coerce")
    invalid_mask = ~numeric_values.notna() | ~np.isfinite(numeric_values.to_numpy())
    if invalid_mask.any():
        invalid_count = int(invalid_mask.sum())
        msg = (
            f"output column '{output_column}' contains {invalid_count} non-numeric, "
            "non-finite, or missing value(s)."
        )
        raise ValueError(msg)


# ── categorization ─────────────────────────────────────────────────────────


def _compute_region_percentages(
    df: pd.DataFrame,
    *,
    output_column: str,
    levels: Sequence[float],
    region_labels: Sequence[str],
) -> pd.Series:
    """Assign each row to a threshold region and return percentage per region.

    Uses integer indices internally for pd.cut so that duplicate or empty
    display labels (e.g. ``["", "", ""]``) are handled correctly.  The
    returned Series carries the display labels as its index.
    """
    bins = [-np.inf, *levels, np.inf]
    values = pd.to_numeric(df[output_column], errors="raise")
    n_regions = len(levels) + 1
    int_labels = list(range(n_regions))
    categorized = pd.cut(values, bins=bins, labels=int_labels, right=False)
    result = (
        categorized.value_counts(normalize=True)
        .reindex(int_labels, fill_value=0.0)
        .mul(100)
        .round(1)
    )
    result.index = pd.Index(region_labels)
    return result


def _compute_adaptive_percentages(
    df: pd.DataFrame,
    *,
    model_func: Any,
    col_tdb: str,
    col_tr: str,
    col_t_rm: str,
    col_v: str,
    visible_specs: list[_BandSpec],
    visible_fields: list[str],
    labels: list[str],
    colors: list[str],
) -> tuple[list[str], list[str], pd.Series]:
    result = model_func(
        tdb=df[col_tdb].values,
        tr=df[col_tr].values,
        t_running_mean=df[col_t_rm].values,
        v=df[col_v].values,
    )

    n = len(df)
    categories = np.full(n, len(visible_specs), dtype=int)

    for i, field_name in enumerate(visible_fields):
        accepted = np.asarray(getattr(result, field_name), dtype=bool)
        categories[accepted] = i

    outside_label = _PlotDefaults.Summary.outside_label
    outside_color = _PlotDefaults.Summary.outside_color
    all_labels = labels + [outside_label]
    all_colors = colors + [outside_color]
    counts = np.bincount(categories, minlength=len(all_labels))
    percentages = pd.Series(
        np.round(counts / n * 100, 1),
        index=all_labels,
    )

    return all_labels, all_colors, percentages


# ── axis preparation ───────────────────────────────────────────────────────


def _prepare_axis(ax: Axes) -> None:
    """Prepare a clean, spine-free axis for summary bar rendering."""
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── annotation helper ─────────────────────────────────────────────────────


def _add_center_text(
    ax: Axes,
    *,
    x: float,
    y: float,
    text: str,
    color: str,
) -> Any:
    """Add a bold, centred text annotation."""
    return ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=_PlotDefaults.Summary.percentage_fontsize,
        fontweight="bold",
        color=color,
    )


# ── unified summary renderer ──────────────────────────────────────────────


def _plot_summary(
    ax: Axes,
    *,
    vertical: bool,
    region_percentages: pd.Series,
    region_labels: Sequence[str],
    region_colors: Sequence[str],
    show_region_labels: bool,
) -> list[Any]:
    """Render a stacked summary bar (horizontal or vertical) with annotations."""
    D = _PlotDefaults.Summary
    artists: list[Any] = []

    if vertical:
        ax.set_xlim(*(D.v_xlim if show_region_labels else D.v_xlim_legend))
        ax.set_ylim(*D.v_ylim)
    else:
        ax.set_xlim(*D.h_xlim)
        ax.set_ylim(*(D.h_ylim if show_region_labels else D.h_ylim_legend))

    cumulative = 0.0

    for i, (label, color) in enumerate(zip(region_labels, region_colors, strict=False)):
        value = float(region_percentages.iloc[i])

        if vertical:
            bar = ax.bar(
                x=D.v_bar_x,
                height=value,
                width=D.v_bar_width,
                bottom=cumulative,
                color=color,
                edgecolor=D.bar_edgecolor,
                linewidth=D.bar_linewidth,
            )
        else:
            bar = ax.barh(
                y=D.h_bar_y,
                width=value,
                left=cumulative,
                height=D.h_bar_height,
                color=color,
                edgecolor=D.bar_edgecolor,
                linewidth=D.bar_linewidth,
            )
        artists.append(bar)

        if value >= D.pct_min_to_show:
            is_light = _is_light_color(color)
            pct_color = "black" if is_light else "white"
            label_color = "dimgray" if is_light else color

            if vertical:
                center_y = cumulative + value / 2
                pct_x, pct_y = D.v_bar_x, center_y
                lbl_x, lbl_y = D.v_bar_x + D.v_label_x_offset, center_y
                lbl_ha, lbl_va = "left", "center"
            else:
                pct_x, pct_y = cumulative + value / 2, D.h_bar_y
                lbl_x, lbl_y = cumulative + value / 2, D.h_label_y
                lbl_ha, lbl_va = "center", "bottom"

            artists.append(
                _add_center_text(
                    ax, x=pct_x, y=pct_y, text=f"{value:.1f}%", color=pct_color
                )
            )

            if show_region_labels:
                artists.append(
                    ax.text(
                        lbl_x,
                        lbl_y,
                        label,
                        ha=lbl_ha,
                        va=lbl_va,
                        fontsize=D.label_fontsize,
                        color=label_color,
                    )
                )

        cumulative += value

    return artists


# ── public API ─────────────────────────────────────────────────────────────


class SummaryPlot(BasePlot):
    """Build and render a threshold summary plot from tabular model outputs.

    The class works with an existing DataFrame that already contains the target
    model output column (e.g., ``pmv`` or ``utci``).
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize a summary plot builder from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            Input DataFrame containing at least one output column to summarize.

        Raises
        ------
        TypeError
            If ``df`` is not a pandas DataFrame.
        ValueError
            If ``df`` is empty.
        """
        super().__init__()
        _validate_dataframe(df)
        self._df = df

        # adaptive mode
        self._adaptive_model: Any | None = None
        self._adaptive_standard: str | None = None
        self._adaptive_col_tdb: str | None = None
        self._adaptive_col_tr: str | None = None
        self._adaptive_col_t_rm: str | None = None
        self._adaptive_col_v: str | None = None
        self._adaptive_labels: list[str] | None = None
        self._adaptive_colors: list[str] | None = None
        self._adaptive_visible_specs: list[_BandSpec] | None = None
        self._adaptive_visible_fields: list[str] | None = None

    def set_regions(
        self,
        *,
        output: str,
        thresholds: Sequence[float],
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
    ) -> SummaryPlot:
        """Set output variable and threshold region configuration.

        Parameters
        ----------
        output : str
            Name of the DataFrame column to categorize.
        thresholds : sequence of float
            Numeric boundary values that divide the output range into regions.
        labels : sequence of str, optional
            Region labels.  Must have length ``len(thresholds) + 1`` when
            provided.
        colors : sequence of str, optional
            Region colors.  Must have length ``len(thresholds) + 1`` when
            provided.

        Returns
        -------
        SummaryPlot
            Self, to support method chaining.

        Raises
        ------
        TypeError
            If ``output`` is not a string.
        ValueError
            If the output column is missing or has invalid values, or if
            thresholds/labels/colors are invalid.
        """
        self._adaptive_model = None

        output_name = _validate_output_column(self._df, output)
        _validate_output_values(self._df, output_name)
        super().set_regions(
            output=output_name,
            thresholds=thresholds,
            labels=labels,
            colors=colors,
        )
        return self

    def set_adaptive_regions(
        self,
        *,
        model_func: Any,
        tdb: str,
        tr: str,
        t_running_mean: str,
        v: str,
        show: RegionsConfig | Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
    ) -> SummaryPlot:
        """Configure adaptive comfort band classification.

        Uses the adaptive model's ``acceptability_*`` outputs to classify
        each row, instead of fixed thresholds.  Cannot be combined with
        :meth:`set_regions` — calling one clears the other.

        Parameters
        ----------
        model_func : callable
            ``adaptive_ashrae`` or ``adaptive_en``.
        tdb : str
            Column name for dry-bulb air temperature.
        tr : str
            Column name for mean radiant temperature.
        t_running_mean : str
            Column name for running mean outdoor temperature.
        v : str
            Column name for air speed.
        show : RegionsConfig, sequence of str, or None
            Band keys to display (same as :meth:`AdaptivePlot.set_regions`).
        labels : sequence of str, optional
            Custom labels for the visible bands.
        colors : sequence of str, optional
            Custom colors for the visible bands.

        Returns
        -------
        SummaryPlot
            Self, to support method chaining.

        Raises
        ------
        ValueError
            If model is not recognized, columns are missing, or band
            keys are invalid.

        Examples
        --------
        .. code-block:: python

            from pythermalcomfort.models import adaptive_ashrae

            SummaryPlot(df).set_adaptive_regions(
                model_func=adaptive_ashrae,
                tdb="tdb",
                tr="tr",
                t_running_mean="t_rm",
                v="v",
            ).plot(title="Adaptive Comfort Distribution")
        """
        # Clear threshold mode
        self._region_config = None

        # Validate model
        name = getattr(model_func, "__name__", "")
        if name not in _MODEL_TO_STANDARD:
            valid = ", ".join(sorted(_MODEL_TO_STANDARD))
            msg = (
                f"model_func must be one of the adaptive model functions "
                f"({valid}), got '{name}'."
            )
            raise ValueError(msg)

        standard = _MODEL_TO_STANDARD[name]

        # Validate columns
        missing = [c for c in [tdb, tr, t_running_mean, v] if c not in self._df.columns]
        if missing:
            msg = f"Column(s) not found in DataFrame: {', '.join(missing)}"
            raise ValueError(msg)

        # Resolve RegionsConfig
        if isinstance(show, RegionsConfig):
            if labels is not None or colors is not None:
                raise ValueError(
                    "labels and colors must not be provided separately when "
                    "show is a RegionsConfig instance."
                )
            config = show
        else:
            config = RegionsConfig(show=show, labels=labels, colors=colors)
        config._validate(standard)

        # Resolve visible bands
        cfg = _STANDARD_CONFIGS[standard]
        all_specs: list[_BandSpec] = cfg["bands"]
        all_fields = _ACCEPTABILITY_FIELDS[standard]

        if config.show is not None:
            show_set = set(config.show)
            visible_specs = [s for s in all_specs if s.key in show_set]
            visible_fields = [
                f
                for s, f in zip(all_specs, all_fields, strict=False)
                if s.key in show_set
            ]
        else:
            visible_specs = list(all_specs)
            visible_fields = list(all_fields)

        resolved_labels: list[str] = []
        resolved_colors: list[str] = []
        for i, spec in enumerate(visible_specs):
            lbl = spec.default_label
            clr = spec.default_color
            if config.labels is not None:
                lbl = str(config.labels[i])
            if config.colors is not None:
                clr = str(config.colors[i])
            resolved_labels.append(lbl)
            resolved_colors.append(clr)

        self._adaptive_model = model_func
        self._adaptive_standard = standard
        self._adaptive_col_tdb = tdb
        self._adaptive_col_tr = tr
        self._adaptive_col_t_rm = t_running_mean
        self._adaptive_col_v = v
        self._adaptive_visible_specs = visible_specs
        self._adaptive_visible_fields = visible_fields
        self._adaptive_labels = resolved_labels
        self._adaptive_colors = resolved_colors

        return self

    def plot(
        self,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        vertical: bool = False,
        legend: bool = True,
        legend_kws: Mapping[str, Any] | None = None,
    ) -> SummaryPlotResult:
        """Render a summary plot.

        Works in two modes depending on configuration:

        - **Threshold mode** (via :meth:`set_regions`): fixed-threshold
          classification using ``pd.cut``.
        - **Adaptive mode** (via :meth:`set_adaptive_regions`): per-row
          classification using the adaptive model's acceptability outputs.

        Parameters
        ----------
        ax : Axes, optional
            Existing axis.  If ``None``, a new figure/axis is created.
        title : str, optional
            Optional chart title.
        vertical : bool
            If ``True``, render a vertical stacked bar.
        legend : bool
            Whether to draw a legend.
        legend_kws : dict, optional
            Overrides for the legend.

        Returns
        -------
        SummaryPlotResult
            Result with figure, axis, percentages, artists, and legend.

        Raises
        ------
        ValueError
            If neither :meth:`set_regions` nor :meth:`set_adaptive_regions`
            has been called.
        """
        with mpl.rc_context(_PYTHERMALCOMFORT_RC):
            # Determine mode and compute percentages
            if self._adaptive_model is not None:
                all_labels, all_colors, percentages = _compute_adaptive_percentages(
                    self._df,
                    model_func=self._adaptive_model,
                    col_tdb=self._adaptive_col_tdb,
                    col_tr=self._adaptive_col_tr,
                    col_t_rm=self._adaptive_col_t_rm,
                    col_v=self._adaptive_col_v,
                    visible_specs=self._adaptive_visible_specs,
                    visible_fields=self._adaptive_visible_fields,
                    labels=self._adaptive_labels,
                    colors=self._adaptive_colors,
                )
            elif self._region_config is not None:
                rc = self._region_config
                percentages = _compute_region_percentages(
                    self._df,
                    output_column=rc.output_name,
                    levels=rc.thresholds,
                    region_labels=rc.labels,
                )
                all_labels = rc.labels
                all_colors = rc.colors
            else:
                raise ValueError(
                    "Regions are not set. Call set_regions(...) or "
                    "set_adaptive_regions(...) before plot(...)."
                )

            if ax is None:
                fig, ax = plt.subplots(figsize=_PlotDefaults.figsize)
            else:
                fig = ax.figure

            _prepare_axis(ax)
            artists = _plot_summary(
                ax,
                vertical=vertical,
                region_percentages=percentages,
                region_labels=all_labels,
                region_colors=all_colors,
                show_region_labels=not legend,
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
                    min(len(all_labels), _PlotDefaults.Threshold.legend_ncol_max),
                )
                handles = [
                    Patch(facecolor=color, label=label)
                    for label, color in zip(all_labels, all_colors, strict=False)
                ]
                legend_artist = ax.legend(handles=handles, **lg_opts)

            if title is not None:
                ax.set_title(
                    title, y=_PlotDefaults.title_y_with_legend if legend else None
                )

            return SummaryPlotResult(
                fig=fig,
                ax=ax,
                percentages=percentages,
                artists=artists,
                legend=legend_artist,
            )
