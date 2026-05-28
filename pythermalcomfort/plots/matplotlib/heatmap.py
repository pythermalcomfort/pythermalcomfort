"""Class-based heatmap plotting for hourly temporal comfort data."""

from __future__ import annotations

import calendar
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.patches import Patch

from pythermalcomfort.plots.matplotlib._base import BasePlot
from pythermalcomfort.plots.matplotlib._shared import (
    _PYTHERMALCOMFORT_RC,
    BasePlotResult,
    _configure_regions,
    _PlotDefaults,
)

# ── result container ───────────────────────────────────────────────────────


@dataclass
class HeatmapPlotResult(BasePlotResult):
    """Container with handles returned by :meth:`HeatmapPlot.plot`.

    Attributes
    ----------
    fig : Figure
        Matplotlib figure containing the heatmap.
    ax : Axes
        Matplotlib axis containing the heatmap.
    mesh : QuadMesh
        The ``pcolormesh`` artist for further customisation.
    legend : Legend or None
        Legend artist if ``legend=True``, otherwise ``None``.
    """

    fig: Figure
    ax: Axes
    mesh: Any
    legend: Legend | None


# ── validation helpers ─────────────────────────────────────────────────────


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Validate that *df* is a non-empty DataFrame with a DatetimeIndex."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("df must contain at least one row.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "df must have a DatetimeIndex. "
            "Set your datetime column as the index with df.set_index('<col>')."
        )


def _validate_single_year(df: pd.DataFrame) -> None:
    """Raise if the DatetimeIndex spans more than one calendar year."""
    years = df.index.year.unique()
    if len(years) > 1:
        year_list = sorted(years.tolist())
        msg = (
            f"HeatmapPlot requires data from a single calendar year, "
            f"but found years: {year_list}. "
            "Filter your DataFrame to one year before plotting."
        )
        raise ValueError(msg)


def _validate_output_column(df: pd.DataFrame, output: str) -> str:
    """Validate output column name and ensure it exists in *df*."""
    if not isinstance(output, str):
        raise TypeError("output must be a string.")
    output_name = output.strip()
    if not output_name:
        raise ValueError("output must be a non-empty string.")
    if output_name not in df.columns:
        msg = f"output column '{output_name}' was not found in the DataFrame."
        raise ValueError(msg)
    return output_name


# ── grid helpers ───────────────────────────────────────────────────────────


def _build_heatmap_grid(
    df: pd.DataFrame,
    *,
    output_col: str,
    thresholds: list[float],
) -> tuple[np.ma.MaskedArray, int, int]:
    """Pivot data into a (24, n_days) masked integer grid of region indices.

    Parameters
    ----------
    df : DataFrame
        Source data with DatetimeIndex.
    output_col : str
        Name of the numeric output column.
    thresholds : list of float
        Sorted threshold values that define region boundaries.

    Returns
    -------
    Z_masked : MaskedArray, shape (24, n_days)
        Integer region indices (0 … n_regions-1); NaN cells are masked.
    day_min : int
        Minimum day-of-year present in the data.
    day_max : int
        Maximum day-of-year present in the data.
    """
    work = pd.DataFrame(
        {
            "_hour": df.index.hour,
            "_doy": df.index.day_of_year,
            "_val": pd.to_numeric(df[output_col], errors="coerce"),
        }
    )

    day_min = int(work["_doy"].min())
    day_max = int(work["_doy"].max())

    pivot = work.pivot_table(
        values="_val", index="_hour", columns="_doy", aggfunc="mean"
    )
    pivot = pivot.reindex(index=range(24), columns=range(day_min, day_max + 1))

    Z_numeric = pivot.to_numpy(dtype=float)
    nan_mask = np.isnan(Z_numeric)

    Z_int = np.digitize(Z_numeric, thresholds)
    Z_masked = np.ma.masked_where(nan_mask, Z_int)

    return Z_masked, day_min, day_max


# ── axis helpers ───────────────────────────────────────────────────────────


def _set_hour_axis(ax: Axes) -> None:
    """Configure y-axis for hours (0 at bottom, 24 at top)."""
    D = _PlotDefaults.Heatmap
    ax.set_ylim(0, 24)
    ax.set_yticks(D.hour_ticks)
    ax.set_ylabel("Hour of Day")


def _set_day_axis_month(ax: Axes, *, day_min: int, day_max: int, year: int) -> None:
    """Set x-axis ticks at month midpoints with abbreviated month labels."""
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    for month in range(1, 13):
        month_start = pd.Timestamp(year=year, month=month, day=1).day_of_year
        days_in_month = calendar.monthrange(year, month)[1]
        month_mid = month_start + days_in_month / 2

        if month_mid < day_min - 1 or month_mid > day_max + 1:
            continue

        tick_positions.append(month_mid)
        tick_labels.append(calendar.month_abbr[month])

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("")


def _set_day_axis_day(ax: Axes) -> None:
    """Configure x-axis with integer day-of-year labels."""
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_xlabel("Day of Year")


# ── public API ─────────────────────────────────────────────────────────────


class HeatmapPlot(BasePlot):
    """Build and render a temporal comfort heatmap from hourly model outputs.

    The plot shows a grid with hours of the day on the y-axis (0 at the
    bottom, 24 at the top) and days on the x-axis.  Each cell is coloured
    by the threshold region it falls in.  NaN values appear as blank cells.

    The class works with a DataFrame that has a :class:`~pandas.DatetimeIndex`
    and a numeric output column (e.g. ``pmv``, ``utci``).

    Examples
    --------
    .. code-block:: python

        import pandas as pd
        from pythermalcomfort.plots.matplotlib import HeatmapPlot

        result = (
            HeatmapPlot(df)
            .set_regions(
                output="pmv",
                thresholds=[-2, -1, 0, 1, 2],
                labels=["Cold", "Cool", "Sl. Cool", "Neutral", "Sl. Warm", "Warm"],
            )
            .plot(title="Annual PMV Heatmap")
        )
        result.fig.savefig("heatmap.png")
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize a heatmap plot builder.

        Parameters
        ----------
        df : DataFrame
            Hourly data with a :class:`~pandas.DatetimeIndex` and at least one
            numeric output column.  Must contain data from a single calendar
            year.

        Raises
        ------
        TypeError
            If ``df`` is not a DataFrame or does not have a DatetimeIndex.
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
    ) -> HeatmapPlot:
        """Set output variable and threshold region configuration.

        Parameters
        ----------
        output : str
            Name of the DataFrame column to visualise.
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
        HeatmapPlot
            Self, to support method chaining.

        Raises
        ------
        TypeError
            If ``output`` is not a string.
        ValueError
            If the output column is missing or thresholds/labels/colors are
            invalid.
        """
        output_name = _validate_output_column(self._df, output)
        self._region_config = _configure_regions(
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
        x_label_format: Literal["month", "day"] = "month",
        legend: bool = True,
        legend_kws: Mapping[str, Any] | None = None,
        mesh_kws: Mapping[str, Any] | None = None,
    ) -> HeatmapPlotResult:
        """Render the temporal comfort heatmap.

        Parameters
        ----------
        ax : Axes, optional
            Existing axis to draw on.  If ``None``, a new figure/axis is
            created with a wide default size.
        title : str, optional
            Optional axis title.
        x_label_format : {'month', 'day'}
            Controls x-axis tick labelling.  ``'month'`` (default) places
            abbreviated month names at each month's midpoint.  ``'day'``
            shows numeric day-of-year ticks.
        legend : bool
            Whether to draw a colour-coded legend above the chart.
        legend_kws : dict, optional
            Keyword overrides forwarded to ``ax.legend``.
        mesh_kws : dict, optional
            Keyword overrides forwarded to ``ax.pcolormesh``.

        Returns
        -------
        HeatmapPlotResult
            Result with figure, axis, mesh artist, and legend handle.

        Raises
        ------
        ValueError
            If regions are not configured first via :meth:`set_regions`, if
            ``x_label_format`` is not ``'month'`` or ``'day'``, or if the
            DataFrame spans more than one calendar year.
        """
        if self._region_config is None:
            raise ValueError(
                "Regions are not set. Call set_regions(...) before plot(...)."
            )
        if x_label_format not in ("month", "day"):
            msg = f"x_label_format must be 'month' or 'day', got '{x_label_format}'."
            raise ValueError(msg)

        _validate_single_year(self._df)

        rc = self._region_config
        n_regions = len(rc.thresholds) + 1

        with mpl.rc_context(_PYTHERMALCOMFORT_RC):
            created_figure = ax is None
            if created_figure:
                fig, ax = plt.subplots(figsize=_PlotDefaults.Heatmap.figsize)
            else:
                fig = ax.figure

            Z_masked, day_min, day_max = _build_heatmap_grid(
                self._df,
                output_col=rc.output_name,
                thresholds=rc.thresholds,
            )

            hour_edges = np.arange(25, dtype=float)
            day_edges = np.arange(day_min - 0.5, day_max + 1.5)

            cmap = ListedColormap(rc.colors)
            extra_kws = dict(mesh_kws or {})
            mesh = ax.pcolormesh(
                day_edges,
                hour_edges,
                Z_masked,
                cmap=cmap,
                vmin=-0.5,
                vmax=n_regions - 0.5,
                **extra_kws,
            )
            ax.set_facecolor("white")

            _set_hour_axis(ax)

            year = int(self._df.index.year[0])
            if x_label_format == "month":
                _set_day_axis_month(ax, day_min=day_min, day_max=day_max, year=year)
            else:
                _set_day_axis_day(ax)

            legend_artist: Legend | None = None
            if legend:
                handles = [
                    Patch(facecolor=color, label=label)
                    for label, color in zip(rc.labels, rc.colors, strict=False)
                ]
                D = _PlotDefaults.Heatmap
                lg_opts = dict(legend_kws or {})
                lg_opts.setdefault("loc", D.legend_loc)
                lg_opts.setdefault("bbox_to_anchor", D.legend_bbox_to_anchor)
                lg_opts.setdefault("ncol", min(n_regions, D.legend_ncol_max))
                legend_artist = ax.legend(handles=handles, **lg_opts)

            if title is not None:
                title_y = _PlotDefaults.title_y_with_legend if legend else None
                ax.set_title(
                    title,
                    fontsize=_PlotDefaults.title_fontsize,
                    y=title_y,
                )

            if created_figure:
                fig.tight_layout()

            return HeatmapPlotResult(
                fig=fig,
                ax=ax,
                mesh=mesh,
                legend=legend_artist,
            )
