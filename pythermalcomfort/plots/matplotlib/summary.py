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
    _resolve_layout,
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
    bar_kws: Mapping[str, Any] | None = None,
) -> list[Any]:
    D = _PlotDefaults.Summary
    artists: list[Any] = []

    bar_opts = dict(bar_kws or {})
    bar_opts.setdefault("edgecolor", D.bar_edgecolor)
    bar_opts.setdefault("linewidth", D.bar_linewidth)

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
                **bar_opts,  
            )
        else:
            bar = ax.barh(
                y=D.h_bar_y,
                width=value,
                left=cumulative,
                height=D.h_bar_height,
                color=color,
                **bar_opts, 
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
        output_name = _validate_output_column(self._df, output)
        _validate_output_values(self._df, output_name)
        super().set_regions(
            output=output_name,
            thresholds=thresholds,
            labels=labels,
            colors=colors,
        )
        return self

    def plot(
        self,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        vertical: bool = False,
        legend: bool = True,
        legend_kws: Mapping[str, Any] | None = None,
        bar_kws: Mapping[str, Any] | None = None,
        # fixme missing bar kwargs
    ) -> SummaryPlotResult:
        """Render a threshold summary plot for the configured output column.

        Parameters
        ----------
        ax : Axes, optional
            Existing axis to draw on.  If ``None``, a new figure/axis is created
            with a default size of ``(7, 4)`` inches.
        title : str, optional
            Optional axis title.  When both *title* and *legend* are shown the
            legend sits just above the chart and the title floats above it,
            matching the spacing used by :class:`ThresholdPlot`.
        vertical : bool
            If ``True``, render a vertical stacked bar; otherwise horizontal.
        legend : bool
            Whether to draw a colour-coded legend above the bar.  When
            ``True``, region labels are omitted from the bar itself.
        legend_kws : dict, optional
            Keyword overrides forwarded to ``ax.legend``.

        Returns
        -------
        SummaryPlotResult
            Result with figure, axis, percentages, artists, and legend handle.

        Raises
        ------
        ValueError
            If regions are not configured first via :meth:`set_regions`.
        """
        with mpl.rc_context(_PYTHERMALCOMFORT_RC):
            if self._region_config is None:
                raise ValueError(
                    "Regions are not set. Call set_regions(...) before plot(...)."
                )
            rc = self._region_config

            if ax is None:
                fsize = (7, 2.0) if not vertical else _PlotDefaults.figsize
                fig, ax = plt.subplots(figsize=fsize)
                if not vertical:
                    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.75)
            else:
                fig = ax.figure

            percentages = _compute_region_percentages(
                self._df,
                output_column=rc.output_name,
                levels=rc.thresholds,
                region_labels=rc.labels,
            )

            _prepare_axis(ax)
            artists = _plot_summary(
                ax,
                vertical=vertical,
                region_percentages=percentages,
                region_labels=rc.labels,
                region_colors=rc.colors,
                show_region_labels=not legend,
                bar_kws=bar_kws,
            )

            legend_artist: Legend | None = None
            if legend:
                dynamic_ncol, dynamic_anchor, dynamic_title_pad = _resolve_layout(
                    title=title, 
                    labels=rc.labels, 
                    default_ncol=_PlotDefaults.Threshold.legend_ncol_max,
                    is_summary=True
                )
                
                lg_opts = dict(legend_kws or {})
                lg_opts.setdefault("loc", "lower center")
                lg_opts.setdefault(
                    "bbox_to_anchor",
                    (0.5, dynamic_anchor) if title is not None else _PlotDefaults.Threshold.legend_bbox_to_anchor,
                )
                lg_opts.setdefault("ncol", dynamic_ncol)

                handles = [
                    Patch(facecolor=color, label=label)
                    for label, color in zip(rc.labels, rc.colors, strict=False)
                ]
                legend_artist = ax.legend(handles=handles, **lg_opts)
            else:
                _, _, dynamic_title_pad = _resolve_layout(title, [], 1, is_summary=True)

            if title is not None:
                ax.set_title(title, pad=dynamic_title_pad)

            return SummaryPlotResult(
                fig=fig,
                ax=ax,
                percentages=percentages,
                artists=artists,
                legend=legend_artist,
            )