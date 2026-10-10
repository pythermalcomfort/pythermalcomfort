"""Class-based threshold plotting, with boundaries solved by root-finding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from typing_extensions import Self
else:
    Self = Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import is_color_like
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pythermalcomfort.plots.matplotlib._base import GridBasePlot
from pythermalcomfort.plots.matplotlib._boundaries import (
    BoundaryCurve,
    RegionBands,
    solve_region_bands,
)
from pythermalcomfort.plots.matplotlib._shared import (
    _EMPTY_REGION_PRESETS,
    _PYTHERMALCOMFORT_RC,
    BasePlotResult,
    _apply_axes_style,
    _AxisConfig,
    _configure_regions,
    _legend_anchor_y,
    _PlotDefaults,
    _title_y_above_legend,
)

#: Default color for grid cells that fall outside the model's applicability limits.
OUT_OF_MODEL_LIMITS_COLOR: str = _PlotDefaults.color_out_of_model


@dataclass
class ThresholdPlotResult(BasePlotResult):
    """Container with handles returned by :meth:`ThresholdPlot.plot`.

    Attributes
    ----------
    fig : Figure
        Matplotlib figure containing the rendered threshold plot.
    ax : Axes
        Matplotlib axis containing the rendered threshold plot.
    lines : list of Line2D
        Threshold boundary lines as editable artists.
    fills : list of PolyCollection
        Filled threshold regions as artists.
    legend : Legend or None
        Legend artist if ``legend=True``, otherwise ``None``.
    boundaries : list of BoundaryCurve
        Threshold boundaries as ``(x, y)`` coordinate arrays, ordered by
        threshold then branch.  A threshold crossed twice per row -- as
        ``ppd`` is, falling to a minimum at neutrality and rising again --
        contributes two curves, distinguished by their ``branch``.
    """

    lines: list[Line2D]
    fills: list[PolyCollection]
    legend: Legend | None
    boundaries: list[BoundaryCurve] = field(default_factory=list)


class ThresholdPlot(GridBasePlot):
    """Configure and render threshold regions for a selected model function.

    The API is staged and explicit:

    1. configure x and y axes,
    2. set fixed model parameters,
    3. define output thresholds and optional labels/colors,
    4. render with :meth:`plot`.

    The returned result contains editable Matplotlib artists, so users can apply
    additional styling with standard Matplotlib code.

    Boundaries are solved by root-finding rather than traced off a raster
    grid, so they follow smooth curves and do not move when ``resolution``
    changes.  See the Notes on :meth:`plot`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import pmv_ppd_iso
        from pythermalcomfort.plots.matplotlib import ThresholdPlot

        result = (
            ThresholdPlot(pmv_ppd_iso)
            .set_x_axis("tdb", 18.0, 34.0, resolution=0.2)
            .set_y_axis("rh", 20.0, 100.0, resolution=0.5)
            .set_params(vr=0.10, met=1.2, clo=0.5, wme=0.0)
            .set_regions(output="pmv", thresholds=[-0.5, 0.5])
            .plot(title="PMV Threshold Regions")
        )
        result.ax.set_xlabel("Air temperature [°C]")

        # The boundaries are available as plain coordinate arrays.
        lower = result.boundaries[0]
        print(lower.threshold, lower.x[:3], lower.y[:3])
    """

    _region_presets = _EMPTY_REGION_PRESETS

    def set_regions(
        self,
        *,
        output: str,
        thresholds: Sequence[float],
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
    ) -> Self:
        """Configure output regions.

        Parameters
        ----------
        output : str
            Output field or column name.
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
        ThresholdPlot
            Self, to support method chaining.

        Raises
        ------
        TypeError
            If ``output`` is not a string.
        ValueError
            If output name is empty, or thresholds/labels/colors are invalid.
        """
        self._region_config = _configure_regions(
            output=output,
            thresholds=thresholds,
            labels=labels,
            colors=colors,
            presets=self._region_presets,
        )
        return self

    def _validate_invalid_color(self, invalid_color: str) -> None:
        """Validate color used to render out-of-model areas."""
        if not isinstance(invalid_color, str) or not is_color_like(invalid_color):
            raise ValueError("invalid_color must be a valid Matplotlib color string.")

    # ── boundary solving ───────────────────────────────────────────────────

    @staticmethod
    def _axis_samples(axis: _AxisConfig, *, minimum: int) -> np.ndarray:
        """Sample one axis, honoring its resolution but never going coarser than
        ``minimum`` points."""
        steps = (
            minimum
            if axis.resolution is None
            else int(np.ceil((axis.max_val - axis.min_val) / axis.resolution)) + 1
        )
        return np.linspace(axis.min_val, axis.max_val, max(steps, minimum))

    def _solve_bands(
        self, output_name: str, thresholds: list[float]
    ) -> RegionBands | None:
        """Solve region boundaries by root-finding, scanning x then y.

        The output usually varies monotonically along one axis and not the
        other — PMV against dry-bulb temperature, say — so a scan that fails
        one way round often succeeds the other.

        Returns
        -------
        RegionBands or None
            ``None`` when neither scan direction yields one band per region
            per row.  Errors raised while evaluating the model propagate
            untouched: they mean the plot is misconfigured, not that the
            geometry is awkward.
        """
        for scan_axis in ("x", "y"):
            scan_config, row_config = (
                (self._x_axis, self._y_axis)
                if scan_axis == "x"
                else (self._y_axis, self._x_axis)
            )
            scan = self._axis_samples(
                scan_config, minimum=_PlotDefaults.Threshold.curve_min_scan_samples
            )
            rows = self._axis_samples(
                row_config, minimum=_PlotDefaults.Threshold.curve_min_rows
            )

            def evaluate(
                scan_values: np.ndarray,
                row_values: np.ndarray,
                _axis: str = scan_axis,
            ) -> np.ndarray:
                x, y = (
                    (scan_values, row_values)
                    if _axis == "x"
                    else (row_values, scan_values)
                )
                return self._evaluate_grid_output(
                    x=np.asarray(x, dtype=float),
                    y=np.asarray(y, dtype=float),
                    output_name=output_name,
                )

            bands = solve_region_bands(
                evaluate=evaluate,
                scan=scan,
                rows=rows,
                thresholds=thresholds,
                scan_axis=scan_axis,
            )
            if bands is not None:
                return bands

        return None

    def _draw_bands(
        self,
        ax: Axes,
        *,
        bands: RegionBands,
        colors: Sequence[str],
        fill_opts: Mapping[str, Any],
        line_opts: Mapping[str, Any],
        show_lines: bool,
        invalid_color: str,
    ) -> tuple[list[PolyCollection], list[Line2D]]:
        """Fill the solved bands and draw their boundary curves."""
        scan_config = self._x_axis if bands.scan_axis == "x" else self._y_axis
        scan_min, scan_max = scan_config.min_val, scan_config.max_val
        # fill_betweenx spans along x for each y; fill_between the other way.
        fill = ax.fill_betweenx if bands.scan_axis == "x" else ax.fill_between

        # One polygon per band rather than per region: a region split in two
        # by a non-monotone output -- "PPD above 10" sits on both sides of the
        # comfort dip -- needs a polygon on each side.
        fills = [
            cast(
                PolyCollection,
                fill(
                    bands.rows,
                    bands.edges[band],
                    bands.edges[band + 1],
                    color=colors[region],
                    **fill_opts,
                ),
            )
            for band, region in enumerate(bands.band_regions)
            if 0 <= region < len(colors)
        ]

        if bands.has_invalid:
            # The two out-of-limits bands are the complement of the valid
            # interval. They are drawn above the region fills so that the
            # wedge a region sweeps out on its way to a fully-invalid row is
            # covered rather than left showing.
            edge = np.full_like(bands.rows, scan_min)
            fills.extend(
                cast(
                    PolyCollection,
                    fill(
                        bands.rows,
                        low,
                        high,
                        color=invalid_color,
                        zorder=_PlotDefaults.Threshold.zorder_invalid,
                        antialiased=False,
                    ),
                )
                for low, high in (
                    (edge, bands.valid_start),
                    (bands.valid_end, np.full_like(bands.rows, scan_max)),
                )
            )

        lines: list[Line2D] = []
        if show_lines:
            for curve in bands.curves:
                (line,) = ax.plot(curve.x, curve.y, **line_opts)
                lines.append(line)

        return fills, lines

    # ── rendering ──────────────────────────────────────────────────────────

    def plot(
        self,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        legend: bool = True,
        show_lines: bool = False,
        line_kws: Mapping[str, Any] | None = None,
        fill_kws: Mapping[str, Any] | None = None,
        legend_kws: Mapping[str, Any] | None = None,
        invalid_color: str = _PlotDefaults.color_out_of_model,
    ) -> ThresholdPlotResult:
        """Render threshold regions and their boundaries on a Matplotlib axis.

        Parameters
        ----------
        ax : Axes, optional
            Existing axis to draw on.  If ``None``, a new figure/axis is
            created with a default size of ``(7, 4)`` inches.
        title : str, optional
            Optional axis title.
        legend : bool
            Whether to draw a legend.
        show_lines : bool
            Whether to draw a line along each threshold boundary.  Defaults to
            ``False``: the region fills meet exactly on the boundary, so the
            colour change already marks it, and an extra line mostly adds
            visual weight -- noticeably so on charts with several bands.  Pass
            ``True`` to draw them.
        line_kws : dict, optional
            Keyword overrides forwarded to ``ax.plot`` for boundary lines.
        fill_kws : dict, optional
            Keyword overrides forwarded to ``ax.fill_between`` for the region
            fills.  Keys ``color`` and ``facecolor`` are reserved and rejected.
        legend_kws : dict, optional
            Keyword overrides forwarded to ``ax.legend``.
        invalid_color : str
            Color used for areas outside the model's applicability limits.
        Returns
        -------
        ThresholdPlotResult
            Result with axis and artist handles.

        Raises
        ------
        ValueError
            If required configuration is missing, plotting inputs are invalid,
            model evaluation fails, or the region layout cannot be solved.

        Notes
        -----
        Boundaries are found by bisection, so ``resolution`` does not set their
        precision.  On the scanned axis it only has to be fine enough to
        separate one threshold crossing from the next; on the other axis it
        sets how finely the boundary curves are sampled.  Both have floors, so
        omitting ``resolution`` gives a chart that is already smooth.
        """
        with mpl.rc_context(_PYTHERMALCOMFORT_RC):
            self._validate_plot_inputs(fill_kws=fill_kws)
            self._validate_invalid_color(invalid_color)

            rc = self._region_config

            line_opts = dict(line_kws or {})
            line_opts.setdefault("color", _PlotDefaults.Threshold.line_color)
            line_opts.setdefault("linewidth", _PlotDefaults.Threshold.line_linewidth)

            legend_opts = dict(legend_kws or {})
            legend_opts.setdefault("loc", _PlotDefaults.Threshold.legend_loc)
            legend_opts.setdefault(
                "bbox_to_anchor",
                _PlotDefaults.legend_bbox_to_anchor_with_title
                if title is not None
                else _PlotDefaults.Threshold.legend_bbox_to_anchor,
            )

            if ax is None:
                fig, ax = plt.subplots(figsize=_PlotDefaults.figsize)
            else:
                fig = ax.figure
            _apply_axes_style(ax)

            fill_opts = dict(fill_kws or {})
            # Degenerate bands occur where a threshold reaches the model's
            # applicability boundary. Without anti-aliasing disabled,
            # Matplotlib can rasterize their zero-width polygon as a thin
            # coloured seam over the adjoining region.
            fill_opts.setdefault("antialiased", False)

            bands = self._solve_bands(rc.output_name, rc.thresholds)
            if bands is None:
                msg = (
                    "Could not lay out the threshold regions: every row of the "
                    "chart has to cut the same way, so the valid area must be "
                    "contiguous along one axis and each threshold crossed the "
                    "same number of times in every row. Narrowing the axis "
                    "ranges to where the model is well behaved usually fixes "
                    "it -- utci(), for instance, needs tdb capped near 42 degC "
                    "before its polynomial stops diverging."
                )
                raise ValueError(msg)

            fills, lines = self._draw_bands(
                ax,
                bands=bands,
                colors=rc.colors,
                fill_opts=fill_opts,
                line_opts=line_opts,
                show_lines=show_lines,
                invalid_color=invalid_color,
            )

            legend_artist: Legend | None = None
            title_y: float | None = None
            if legend:
                handles = [
                    Patch(
                        facecolor=color,
                        alpha=fill_opts.get("alpha", _PlotDefaults.fill_alpha),
                        label=label,
                    )
                    for label, color in zip(rc.labels, rc.colors, strict=False)
                ]
                if bands.has_invalid:
                    handles.append(
                        Patch(
                            facecolor=invalid_color,
                            alpha=1.0,
                            label="Out of model limits",
                        )
                    )
                legend_opts.setdefault(
                    "ncol", min(len(handles), _PlotDefaults.Threshold.legend_ncol_max)
                )
                legend_artist = ax.legend(
                    handles=handles,
                    **legend_opts,
                )
                title_y = _title_y_above_legend(
                    n_handles=len(handles),
                    ncol=int(legend_opts["ncol"]),
                    anchor_y=_legend_anchor_y(
                        legend_opts["bbox_to_anchor"],
                        ax=ax,
                        bbox_transform=legend_opts.get("bbox_transform"),
                    ),
                )

            ax.set_xlim(self._x_axis.min_val, self._x_axis.max_val)
            ax.set_ylim(self._y_axis.min_val, self._y_axis.max_val)
            ax.set_xlabel(self._x_axis.name)
            ax.set_ylabel(self._y_axis.name)
            if title is not None:
                ax.set_title(title, y=title_y)

            return ThresholdPlotResult(
                fig=fig,
                ax=ax,
                lines=lines,
                fills=fills,
                legend=legend_artist,
                boundaries=bands.curves,
            )
