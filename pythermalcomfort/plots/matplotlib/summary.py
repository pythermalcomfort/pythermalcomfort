"""Class-based summary plotting for threshold regions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.patches import Patch

from pythermalcomfort.plots.matplotlib._base import BasePlot
from pythermalcomfort.plots.matplotlib._shared import (
    _PYTHERMALCOMFORT_RC,
    BasePlotResult,
    _configure_regions,
    _is_light_color,
    _PlotDefaults,
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


def _compute_category_percentages(
    categories: Sequence[str] | pd.Series | np.ndarray,
    labels: Sequence[str],
) -> pd.Series:
    """Return percentage per label for a pre-computed array of category values.

    Unlike ``pandas.Series.value_counts(normalize=True)``, this never silently
    drops or renormalizes around unexpected values (including NaN) — any value
    not present in ``labels`` raises instead of being dropped.
    """
    series = pd.Series(np.asarray(categories, dtype=object))
    label_list = list(labels)

    valid = series.isin(label_list)
    if not valid.all():
        unknown = sorted(
            {"NaN" if pd.isna(v) else str(v) for v in series[~valid].unique()}
        )
        msg = (
            f"categories contains value(s) not present in labels: {unknown}. "
            f"Every value must match one of {label_list}."
        )
        raise ValueError(msg)

    counts = series.value_counts().reindex(label_list, fill_value=0)
    percentages = (counts / len(series) * 100).round(1)
    return pd.Series(percentages.to_numpy(), index=pd.Index(label_list))


# ── axis preparation ───────────────────────────────────────────────────────


def _prepare_axis(ax: Axes) -> None:
    """Prepare a clean, spine-free axis for summary bar rendering."""
    ax.clear()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _should_show_region_labels(*, legend: bool, region_labels: Sequence[str]) -> bool:
    """Return whether region labels should be drawn next to the bar."""
    return not legend and any(label.strip() for label in region_labels)


def _summary_figsize(
    *,
    vertical: bool,
    legend: bool,
    show_region_labels: bool,
) -> tuple[float, float]:
    """Return compact default figure size for standalone summary plots."""
    D = _PlotDefaults.Summary
    if vertical:
        if show_region_labels:
            return D.figsize_vertical_labels
        return D.figsize_vertical_legend if legend else D.figsize_vertical_plain

    if legend:
        return D.figsize_horizontal_legend
    return (
        D.figsize_horizontal_labels
        if show_region_labels
        else D.figsize_horizontal_plain
    )


def _apply_compact_layout(fig: Figure) -> None:
    """Trim default Matplotlib margins for standalone summary figures."""
    fig.tight_layout(pad=_PlotDefaults.Summary.layout_pad)


def _ensure_title_legend_spacing(
    fig: Figure,
    ax: Axes,
    legend: Legend | None,
    *,
    adjust_layout: bool,
) -> None:
    """Keep the title above the legend with only the measured gap needed."""
    if legend is None or not ax.get_title():
        return

    D = _PlotDefaults.Summary
    min_gap_px = D.title_legend_gap * fig.dpi / 72
    for _ in range(D.title_spacing_iterations):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        title_bbox = ax.title.get_window_extent(renderer)
        legend_bbox = legend.get_window_extent(renderer)
        current_gap_px = title_bbox.y0 - legend_bbox.y1
        if current_gap_px >= min_gap_px:
            return

        axes_height_px = ax.get_window_extent(renderer).height
        if axes_height_px <= 0:
            return

        _, title_y = ax.title.get_position()
        y_adjustment = (min_gap_px - current_gap_px) / axes_height_px
        ax.title.set_y(title_y + y_adjustment)
        if adjust_layout:
            _apply_compact_layout(fig)


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
    bar_kws: Mapping[str, Any],
) -> list[Any]:
    """Render a stacked summary bar (horizontal or vertical) with annotations."""
    D = _PlotDefaults.Summary
    artists: list[Any] = []
    bar_opts = {
        "edgecolor": D.bar_edgecolor,
        "linewidth": D.bar_linewidth,
        **dict(bar_kws),
    }

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
            bar_args = {
                "x": D.v_bar_x,
                "height": value,
                "width": D.v_bar_width,
                "bottom": cumulative,
                "color": color,
                **bar_opts,
            }
            bar = ax.bar(**bar_args)
        else:
            bar_args = {
                "y": D.h_bar_y,
                "width": value,
                "left": cumulative,
                "height": D.h_bar_height,
                "color": color,
                **bar_opts,
            }
            bar = ax.barh(**bar_args)
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


def _default_legend_ncol(*, vertical: bool, n_labels: int) -> int:
    """Return a compact default legend column count."""
    D = _PlotDefaults.Summary
    if vertical:
        return D.vertical_legend_ncol
    return min(n_labels, D.legend_ncol_max)


# ── public API ─────────────────────────────────────────────────────────────


class SummaryPlot(BasePlot):
    """Build and render a summary plot from tabular model outputs.

    Two mutually exclusive ways to configure regions are available — calling
    one clears any configuration set by the other:

    - :meth:`set_regions` — bin one continuous output column against fixed
      thresholds (e.g. PMV thresholds at -0.5/0.5).
    - :meth:`set_categories` — summarize an already-classified per-row
      category array, computed however makes sense for the model at hand
      (see :meth:`set_categories` for a worked example with adaptive comfort).
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
        self._categories: np.ndarray | None = None
        self._category_labels: list[str] | None = None
        self._category_colors: list[str] | None = None

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
        self._region_config = _configure_regions(
            output=output_name,
            thresholds=thresholds,
            labels=labels,
            colors=colors,
        )
        self._categories = None
        self._category_labels = None
        self._category_colors = None
        return self

    def set_categories(
        self,
        categories: Sequence[str] | pd.Series | np.ndarray,
        *,
        labels: Sequence[str],
        colors: Sequence[str],
    ) -> SummaryPlot:
        """Summarize an already-classified, per-row category array.

        Use this when a model's output can't be reduced to one continuous
        column plus fixed thresholds — e.g. adaptive comfort, where
        acceptability depends on each row's own running mean temperature.
        Compute the categories however makes sense for the model at hand,
        then hand the result to this method.  ``SummaryPlot`` never needs to
        know how a category was derived.

        Parameters
        ----------
        categories : sequence of str, Series, or ndarray
            One category label per row of the DataFrame passed to the
            constructor.  Every value must be present in ``labels``.
        labels : sequence of str
            Every category to display, in display order.  Categories with
            zero matching rows still render at 0%.
        colors : sequence of str
            Color for each entry in ``labels``.  Must be the same length.

        Returns
        -------
        SummaryPlot
            Self, to support method chaining.

        Raises
        ------
        ValueError
            If ``categories`` has the wrong length, contains a value not in
            ``labels``, or if ``labels``/``colors`` have mismatched lengths.

        Examples
        --------
        Adaptive comfort has no single continuous column to threshold — each
        row's acceptability depends on its own running mean temperature, so
        the model itself returns per-row boolean acceptability fields.
        ``np.select`` picks the narrowest (best) band each row satisfies,
        falling back to "Outside" for rows that satisfy none:

        .. code-block:: python

            import numpy as np
            from pythermalcomfort.models import adaptive_ashrae

            result = adaptive_ashrae(
                tdb=df["tdb"], tr=df["tr"], t_running_mean=df["t_rm"], v=df["v"]
            )
            categories = np.select(
                [result.acceptability_90, result.acceptability_80],
                ["90% Acceptability", "80% Acceptability"],
                default="Outside",
            )
            SummaryPlot(df).set_categories(
                categories,
                labels=["90% Acceptability", "80% Acceptability", "Outside"],
                colors=["#6BB3FF", "#B3D9FF", "#D9D9D9"],
            ).plot(title="Adaptive comfort distribution (ASHRAE 55)")

        The same pattern applies to any other model with its own
        classification logic — only the ``np.select`` conditions change.
        """
        label_list = list(labels)
        color_list = list(colors)

        if len(set(label_list)) != len(label_list):
            raise ValueError("labels must not contain duplicate values.")

        if len(label_list) != len(color_list):
            msg = (
                f"labels and colors must have the same length "
                f"(got {len(label_list)} and {len(color_list)})."
            )
            raise ValueError(msg)

        invalid_colors = [c for c in color_list if not mcolors.is_color_like(c)]
        if invalid_colors:
            msg = f"Invalid color value(s): {', '.join(map(str, invalid_colors))}."
            raise ValueError(msg)

        if len(categories) != len(self._df):
            msg = (
                f"categories must have one value per row of the DataFrame "
                f"(got {len(categories)}, expected {len(self._df)})."
            )
            raise ValueError(msg)

        # Validate up front so errors surface at configuration time, not
        # inside plot() — mirrors set_regions()'s _validate_output_values.
        _compute_category_percentages(categories, label_list)

        self._categories = np.asarray(categories, dtype=object)
        self._category_labels = label_list
        self._category_colors = color_list
        self._region_config = None
        return self

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
        """Render a summary plot for the configured regions or categories.

        Parameters
        ----------
        ax : Axes, optional
            Existing axis to draw on.  If ``None``, a new figure/axis is created
            with a compact default size for the selected orientation.
        title : str, optional
            Optional axis title.  When both *title* and *legend* are shown the
            legend sits just above the chart and the title floats above it,
            matching the spacing used by :class:`ThresholdPlot`.
        vertical : bool
            If ``True``, render a vertical stacked bar; otherwise horizontal.
        legend : bool
            Whether to draw a colour-coded legend above the bar.  When
            ``True``, region labels are omitted from the bar itself.
        bar_kws : dict, optional
            Keyword overrides forwarded to ``ax.bar`` or ``ax.barh`` for the
            stacked bar segments after SummaryPlot defaults are applied.
        legend_kws : dict, optional
            Keyword overrides forwarded to ``ax.legend``.

        Returns
        -------
        SummaryPlotResult
            Result with figure, axis, percentages, artists, and legend handle.

        Raises
        ------
        ValueError
            If neither :meth:`set_regions` nor :meth:`set_categories` has
            been called first.
        """
        with mpl.rc_context(_PYTHERMALCOMFORT_RC):
            if self._categories is not None:
                percentages = _compute_category_percentages(
                    self._categories, self._category_labels
                )
                labels, colors = self._category_labels, self._category_colors
            elif self._region_config is not None:
                rc = self._region_config
                percentages = _compute_region_percentages(
                    self._df,
                    output_column=rc.output_name,
                    levels=rc.thresholds,
                    region_labels=rc.labels,
                )
                labels, colors = rc.labels, rc.colors
            else:
                raise ValueError(
                    "Regions are not set. Call set_regions(...) or "
                    "set_categories(...) before plot(...)."
                )

            show_region_labels = _should_show_region_labels(
                legend=legend,
                region_labels=labels,
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

            _prepare_axis(ax)
            artists = _plot_summary(
                ax,
                vertical=vertical,
                region_percentages=percentages,
                region_labels=labels,
                region_colors=colors,
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
                    _default_legend_ncol(vertical=vertical, n_labels=len(labels)),
                )
                handles = [
                    Patch(facecolor=color, label=label)
                    for label, color in zip(labels, colors, strict=False)
                ]
                legend_artist = ax.legend(handles=handles, **lg_opts)

            if title is not None:
                ax.set_title(
                    title, y=_PlotDefaults.title_y_with_legend if legend else None
                )

            if created_figure:
                _apply_compact_layout(fig)

            _ensure_title_legend_spacing(
                fig=fig,
                ax=ax,
                legend=legend_artist,
                adjust_layout=created_figure,
            )

            return SummaryPlotResult(
                fig=fig,
                ax=ax,
                percentages=percentages,
                artists=artists,
                legend=legend_artist,
            )
