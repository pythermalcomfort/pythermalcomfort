"""Class-based threshold plotting with a contour backend."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, is_color_like
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pythermalcomfort.plots.matplotlib._base import GridBasePlot
from pythermalcomfort.plots.matplotlib._shared import (
    _PYTHERMALCOMFORT_RC,
    BasePlotResult,
    _PlotDefaults,
    _resolve_layout,
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
        Contour boundary lines as editable artists.
    fills : list of PolyCollection
        Filled threshold regions as artists.
    legend : Legend or None
        Legend artist if ``legend=True``, otherwise ``None``.
    """

    lines: list[Line2D]
    fills: list[PolyCollection]
    legend: Legend | None


def _contour_paths_to_lines(
    ax: Axes,
    *,
    contour_set: Any,
    line_opts: Mapping[str, Any],
) -> list[Line2D]:
    """Convert contour paths into editable line artists."""
    lines: list[Line2D] = []

    for path in contour_set.get_paths():
        segments = path.to_polygons(closed_only=False)
        x_combined: list[float] = []
        y_combined: list[float] = []

        for segment in segments:
            if len(segment) == 0:
                continue
            x_combined.extend(segment[:, 0].tolist())
            x_combined.append(np.nan)
            y_combined.extend(segment[:, 1].tolist())
            y_combined.append(np.nan)

        if x_combined:
            (line,) = ax.plot(x_combined, y_combined, **line_opts)
            lines.append(line)

    contour_set.remove()
    return lines


class ThresholdPlot(GridBasePlot):
    """Configure and render threshold regions for a selected model function.

    The API is staged and explicit:

    1. configure x and y axes,
    2. set fixed model parameters,
    3. define output thresholds and optional labels/colors,
    4. render with :meth:`plot`.

    The returned result contains editable Matplotlib artists, so users can apply
    additional styling with standard Matplotlib code.

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
    """

    def _validate_invalid_color(self, invalid_color: str) -> None:
        """Validate color used to render out-of-model areas."""
        if not isinstance(invalid_color, str) or not is_color_like(invalid_color):
            raise ValueError("invalid_color must be a valid Matplotlib color string.")

    def plot(
        self,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        legend: bool = True,
        show_lines: bool = True,
        line_kws: Mapping[str, Any] | None = None,
        fill_kws: Mapping[str, Any] | None = None,
        legend_kws: Mapping[str, Any] | None = None,
        invalid_color: str = _PlotDefaults.color_out_of_model,
    ) -> ThresholdPlotResult:
        """Render threshold regions and contours on a Matplotlib axis.

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
            Whether to draw threshold contour boundaries.
        line_kws : dict, optional
            Keyword overrides forwarded to ``ax.plot`` for contour lines.
        fill_kws : dict, optional
            Keyword overrides forwarded to ``ax.contourf`` for region fills.
            Keys ``color`` and ``facecolor`` are reserved and rejected.
        legend_kws : dict, optional
            Keyword overrides forwarded to ``ax.legend``.
        invalid_color : str
            Color used for out-of-model/invalid grid areas.

        Returns
        -------
        ThresholdPlotResult
            Result with axis and artist handles.

        Raises
        ------
        ValueError
            If required configuration is missing, plotting inputs are invalid,
            or model evaluation/output extraction fails.
        """
        with mpl.rc_context(_PYTHERMALCOMFORT_RC):
            self._validate_plot_inputs(fill_kws=fill_kws)
            self._validate_invalid_color(invalid_color)

            rc = self._region_config

            line_opts = dict(line_kws or {})
            line_opts.setdefault("color", _PlotDefaults.Threshold.line_color)
            line_opts.setdefault("linewidth", _PlotDefaults.Threshold.line_linewidth)

            fill_opts = dict(fill_kws or {})
            fill_opts.setdefault(
                "corner_mask", _PlotDefaults.Threshold.fill_corner_mask
            )

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

            x_min, x_max, y_min, y_max, x, y = self._build_grid()
            z = self._evaluate_grid_output(x=x, y=y, output_name=rc.output_name)

            finite = np.isfinite(z)
            invalid = ~finite
            z_masked = np.ma.masked_invalid(z)

            # Layer 1: Set the canvas background color to an invalid color
            if invalid.any():
                ax.set_facecolor(invalid_color)

            extended_levels = [-np.inf, *rc.thresholds, np.inf]
            fills: list[PolyCollection] = []
            
            if finite.any():
                # Layer 2: Limited interval pure white opacity (zorder=2)
                z_min, z_max = np.nanmin(z), np.nanmax(z)
                underlay_levels = [z_min - 1.0, z_max + 1.0] if z_min != z_max else [z_min - 1.0, z_min + 1.0]
                
                underlay_opts = {k: v for k, v in fill_opts.items() if k != "alpha"}
                ax.contourf(
                    x,
                    y,
                    z_masked,
                    levels=underlay_levels,
                    colors=["white"],
                    extend="neither",
                    zorder=2,  
                    **underlay_opts,
                )

                # Layer 3: HD Color Blocks (zorder=3)
                filled_contours = ax.contourf(
                    x,
                    y,
                    z_masked,
                    levels=extended_levels,
                    colors=rc.colors,
                    extend="neither",
                    zorder=3,
                    **fill_opts,
                )

                if hasattr(filled_contours, "collections"):
                    for c in filled_contours.collections:
                        c.set_edgecolor("face")
                        c.set_linewidth(0.5)
                    fills.extend(
                        cast(list[PolyCollection], list(filled_contours.collections))
                    )
                else:
                    if hasattr(filled_contours, "set_edgecolor"):
                        filled_contours.set_edgecolor("face")
                    if hasattr(filled_contours, "set_linewidth"):
                        filled_contours.set_linewidth(0.5)
                    fills.append(cast(PolyCollection, filled_contours))

            # Layer 4: Border black line (zorder=4) 
            lines: list[Line2D] = []
            if show_lines and finite.any():
                contour_lines = ax.contour(
                    x, y, z_masked, levels=rc.thresholds, antialiased=True, linewidths=0
                )
                
                line_opts = dict(line_opts)
                line_opts.setdefault("zorder", 4)
                lines = _contour_paths_to_lines(
                    ax,
                    contour_set=contour_lines,
                    line_opts=line_opts,
                )

            legend_artist: Legend | None = None
            if legend:
                handles = [
                    Patch(
                        facecolor=color,
                        alpha=fill_opts.get("alpha", _PlotDefaults.fill_alpha),
                        label=label,
                    )
                    for label, color in zip(rc.labels, rc.colors, strict=False)
                ]
                if invalid.any():
                    handles.append(
                        Patch(
                            facecolor=invalid_color,
                            alpha=1.0,
                            label="Out of model limits",
                        )
                    )
                
                labels_text = [h.get_label() for h in handles]
                dynamic_ncol, dynamic_anchor, dynamic_title_pad = _resolve_layout(
                    title=title, 
                    labels=labels_text, 
                    default_ncol=_PlotDefaults.Threshold.legend_ncol_max
                )

                legend_opts = dict(legend_kws or {})
                legend_opts.setdefault("loc", _PlotDefaults.Threshold.legend_loc)
                legend_opts.setdefault(
                    "bbox_to_anchor",
                    (0.5, dynamic_anchor) if title is not None else _PlotDefaults.Threshold.legend_bbox_to_anchor,
                )
                legend_opts.setdefault("ncol", dynamic_ncol)
                
                legend_artist = ax.legend(
                    handles=handles,
                    **legend_opts,
                )
            else:
                _, _, dynamic_title_pad = _resolve_layout(title, [], 1)

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel(self._x_axis.name)
            ax.set_ylabel(self._y_axis.name)
            
            if title is not None:
                ax.set_title(title, pad=dynamic_title_pad)

            return ThresholdPlotResult(
                fig=fig,
                ax=ax,
                lines=lines,
                fills=fills,
                legend=legend_artist,
            )