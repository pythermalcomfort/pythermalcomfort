"""Class-based psychrometric charting with contour threshold regions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.path import Path as MplPath

from pythermalcomfort.plots.matplotlib._shared import (
    _apply_default_links_to_kwargs,
    _AxisConfig,
    _extract_output_by_name,
    _parse_axis_range,
    _PlotDefaults,
    _validate_model_kwargs,
    _validate_resolution,
)
from pythermalcomfort.plots.matplotlib.threshold import (
    ThresholdPlot,
    ThresholdPlotResult,
)
from pythermalcomfort.utilities import hr_to_rh, psy_ta_rh


class PsychrometricPlot(ThresholdPlot):
    """Configure and render a psychrometric chart with threshold regions.

    Inherits from :class:`ThresholdPlot` and strictly enforces ``hr``
    (humidity ratio) on the y-axis.  Any model temperature parameter
    (``tdb``, ``tr``, etc.) may be used on the x-axis.  Grid evaluation
    converts humidity ratio back to relative humidity before calling the
    underlying model.  Constant-RH background curves are drawn on top of
    the threshold regions.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import pmv_ppd_iso
        from pythermalcomfort.plots.matplotlib import PsychrometricPlot

        result = (
            PsychrometricPlot(pmv_ppd_iso)
            .set_x_axis("tdb", 10.0, 36.0, resolution=0.2)
            .set_y_axis("hr", 0.0, 0.030, resolution=0.0005)
            .set_params(vr=0.10, met=1.2, clo=0.5, wme=0.0)
            .set_regions(output="pmv", thresholds=[-0.5, 0.5])
            .plot(title="PMV — Psychrometric Chart")
        )
    """

    def set_x_axis(
        self,
        name: str,
        min_val: float,
        max_val: float,
        *,
        resolution: float,
    ) -> PsychrometricPlot:
        """Set x-axis; any model temperature parameter is accepted.

        Common choices are ``'tdb'`` (dry-bulb temperature) and ``'tr'``
        (mean radiant temperature).  The RH curves and saturation boundary
        overlaid on the chart are computed using the x-axis values as the
        reference temperature, so accuracy is highest when ``'tdb'`` is used.

        Parameters
        ----------
        name : str
            Model argument name mapped to the x-axis (e.g. ``'tdb'``, ``'tr'``).
        min_val : float
            Minimum value.
        max_val : float
            Maximum value.
        resolution : float
            Grid step along the x-axis.

        Returns
        -------
        PsychrometricPlot
            Self, to support method chaining.

        Raises
        ------
        ValueError
            If ``name`` is not a valid model argument, or range/resolution
            are invalid.
        """
        return super().set_x_axis(name, min_val, max_val, resolution=resolution)  # type: ignore[return-value]

    def set_y_axis(
        self,
        name: str,
        min_val: float,
        max_val: float,
        *,
        resolution: float,
    ) -> PsychrometricPlot:
        """Set y-axis; must be ``'hr'`` (humidity ratio).

        The standard model-argument check is bypassed because thermal comfort
        models accept ``rh`` (relative humidity), not ``hr`` directly.  Grid
        evaluation handles the conversion internally.

        Parameters
        ----------
        name : str
            Must be ``'hr'``.
        min_val : float
            Minimum humidity ratio (kg/kg).
        max_val : float
            Maximum humidity ratio (kg/kg).
        resolution : float
            Grid step along the y-axis.

        Returns
        -------
        PsychrometricPlot
            Self, to support method chaining.

        Raises
        ------
        ValueError
            If ``name`` is not ``'hr'``, conflicts with a fixed parameter set
            via :meth:`set_params`, or if range/resolution are invalid.
        """
        if name != "hr":
            raise ValueError(
                "PsychrometricPlot requires the y-axis to be 'hr' (humidity ratio)."
            )
        if name in self._fixed_values:
            msg = (
                f"set_params() already contains axis parameter '{name}'. "
                "Remove it before calling set_y_axis()."
            )
            raise ValueError(msg)
        if self._x_axis is not None and name == self._x_axis.name:
            raise ValueError("x and y axis parameters must be different. ")

        min_float, max_float = _parse_axis_range(min_val, max_val)
        resolution_float = _validate_resolution(resolution)
        self._y_axis = _AxisConfig(
            name=name,
            min_val=min_float,
            max_val=max_float,
            resolution=resolution_float,
        )
        return self

    def _evaluate_grid_output(
        self,
        *,
        x: np.ndarray,
        y: np.ndarray,
        output_name: str,
    ) -> np.ndarray:
        """Evaluate the model on the psychrometric grid and return shaped output.

        Converts the humidity-ratio grid (*y*) to relative humidity before
        calling the model.  Cells above the saturation curve (RH > 100 %) are
        clamped to RH = 100 % rather than set to NaN, so the contourf fills the
        entire grid rectangle without holes or a jagged upper edge.  The
        ``plot()`` method then overlays a smooth white fill that hides the
        above-saturation region.  Cells where the model itself returns NaN due
        to applicability limits are still propagated as NaN.

        When the x-axis is ``'tdb'``, dry-bulb temperature is used directly for
        the saturation-pressure calculation.  When a fixed ``'tdb'`` is provided
        via :meth:`set_params`, that value is used instead.  Otherwise the
        x-axis values serve as an approximation (accurate when ``tr ≈ tdb``).
        """
        x_flat = np.asarray(x).ravel()
        y_flat = np.asarray(y).ravel()  # hr (kg/kg)

        # Determine which temperature to use for the hr → rh conversion.
        if self._x_axis.name == "tdb":
            tdb_for_psat = x_flat
        elif "tdb" in self._fixed_values:
            tdb_for_psat = np.full_like(x_flat, float(self._fixed_values["tdb"]))
        else:
            # tr auto-links to tdb; using x-axis values is a reasonable approximation.
            tdb_for_psat = x_flat

        p_atm = _PlotDefaults.Psychrometric.p_atm
        rh_flat = hr_to_rh(y_flat, tdb_for_psat, p_atm)

        # Clamp RH to [0, 100] — super-saturated cells are evaluated at rh=100%
        # rather than being excluded.  This keeps the contourf gap-free; the
        # white overlay in plot() hides the above-saturation region.
        rh_safe = np.clip(rh_flat, 0.0, 100.0)

        grid_kwargs: dict[str, Any] = dict(self._fixed_values)
        grid_kwargs[self._x_axis.name] = x_flat
        grid_kwargs["rh"] = rh_safe
        grid_kwargs = _apply_default_links_to_kwargs(
            grid_kwargs,
            allowed_args=self._allowed_args,
            default_links=self._default_links,
        )
        _validate_model_kwargs(
            grid_kwargs,
            allowed_args=self._allowed_args,
            required_args=self._required_args,
            accepts_var_kwargs=self._accepts_var_kwargs,
        )

        try:
            result = self._model_func(**grid_kwargs)
        except Exception as exc:
            msg = f"Failed to evaluate model on psychrometric grid: {exc}"
            raise ValueError(msg) from exc

        try:
            payload = _extract_output_by_name(result, output_name)
        except Exception as exc:
            msg = f"Failed to extract output '{output_name}' from psychrometric result: {exc}"
            raise ValueError(msg) from exc

        z_flat = np.asarray(payload, dtype=float)
        if z_flat.size != x.size:
            msg = (
                "Model output shape does not match the contour grid. "
                f"Expected {x.size} values for the flattened grid, got {z_flat.size}."
            )
            raise ValueError(msg)

        return z_flat.reshape(x.shape)

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
        """Render the psychrometric chart with threshold regions and RH curves.

        Delegates to :meth:`ThresholdPlot.plot` for contour rendering, then
        overlays:

        - A white fill masking the physically impossible RH > 100 % area,
          starting exactly at the smooth saturation curve.
        - Dotted constant-RH background curves at 10 % intervals.

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
        """
        result = super().plot(
            ax=ax,
            title=title,
            legend=legend,
            show_lines=show_lines,
            line_kws=line_kws,
            fill_kws=fill_kws,
            legend_kws=legend_kws,
            invalid_color=invalid_color,
        )
        ax = result.ax

        t_dense = np.linspace(
            self._x_axis.min_val,
            self._x_axis.max_val,
            _PlotDefaults.Psychrometric.n_tdb_points,
        )
        label_offset = (
            self._y_axis.max_val - self._y_axis.min_val
        ) * _PlotDefaults.Psychrometric.rh_label_offset_fraction

        # White fill masks the physically impossible RH > 100% region.
        # Because the contourf fills the entire grid (super-saturated cells are
        # evaluated at rh=100% rather than NaN), there are no jagged pcolormesh
        # edges to cover.  The mask starts exactly at the smooth saturation curve.
        hr_100 = psy_ta_rh(t_dense, np.full_like(t_dense, 100.0)).hr
        ax.fill_between(
            t_dense,
            hr_100,
            self._y_axis.max_val,
            color="white",
            zorder=_PlotDefaults.Psychrometric.zorder_rh_mask,
            edgecolor="none",
        )


        step = _PlotDefaults.Psychrometric.rh_curve_step
        for rh_target in range(step, 110, step):
            hr_line = psy_ta_rh(t_dense, np.full_like(t_dense, float(rh_target))).hr
            in_range = hr_line <= self._y_axis.max_val
            if not in_range.any():
                continue
            ax.plot(
                t_dense[in_range],
                hr_line[in_range],
                color=_PlotDefaults.Psychrometric.rh_line_color,
                linestyle=":",
                linewidth=_PlotDefaults.Psychrometric.rh_line_linewidth,
                zorder=_PlotDefaults.Psychrometric.zorder_rh_lines,
            )
            last_idx = int(np.where(in_range)[0][-1])
            ax.text(
                t_dense[last_idx],
                hr_line[last_idx] + label_offset,
                f"{rh_target}%",
                color=_PlotDefaults.Psychrometric.rh_line_color,
                fontsize=_PlotDefaults.Psychrometric.rh_label_fontsize,
                zorder=_PlotDefaults.Psychrometric.zorder_rh_lines,
            )

        ax.set_xlim(self._x_axis.min_val, self._x_axis.max_val)
        ax.set_ylim(self._y_axis.min_val, self._y_axis.max_val)

        return result
