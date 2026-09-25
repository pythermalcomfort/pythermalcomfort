"""Class-based psychrometric charting with contour threshold regions."""

from __future__ import annotations

import warnings
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
    _suppress_applicability_warnings,
    _validate_model_kwargs,
    _validate_resolution,
)
from pythermalcomfort.plots.matplotlib.threshold import (
    ThresholdPlot,
    ThresholdPlotResult,
)
from pythermalcomfort.psychrometrics import hr_to_rh, psy_ta_rh

#: Grams of water per kilogram of dry air, used to convert between the
#: chart's display units (g/kg dry air) and the kg/kg dry air that
#: :func:`~pythermalcomfort.psychrometrics.psy_ta_rh` and
#: :func:`~pythermalcomfort.psychrometrics.hr_to_rh` work in.
_G_PER_KG = 1000.0

#: A y-axis upper bound below this is *usually* kg/kg dry air left over from
#: before 4.5.0, since typical indoor humidity ratios are 5-20 g/kg.  It is not
#: conclusive, though: sub-freezing air genuinely holds well under 1 g/kg (at
#: -20 degC, 0.5 g/kg is roughly 80 % RH), and this package covers cold stress.
#: So this triggers a warning rather than an error, and the message covers both
#: readings.
_MIN_PLAUSIBLE_HR_MAX_G_KG = 1.0

#: Default y-axis label.  Spelled out because the denominator being *dry* air
#: is the part readers get wrong.
_HR_AXIS_LABEL = r"Humidity ratio [g$_\mathrm{water}$/kg$_\mathrm{dry\,air}$]"


def _label_split_index(x: np.ndarray) -> int:
    """Index on a curve where its label sits."""
    return min(int(x.size * _PlotDefaults.Psychrometric.rh_label_position), x.size - 2)


def _label_along_curve(ax: Axes, *, x: np.ndarray, y: np.ndarray, text: str) -> None:
    """Write *text* in a gap in a curve, rotated to follow it.

    Psychrometric charts conventionally carry the RH value on the curve rather
    than beside it: the curves fan out, and a label parked at the end of one is
    easy to read against the wrong line.  The caller leaves a gap in the curve
    for the label, so nothing has to be painted over -- a filled background
    would sit as a pale rectangle on whatever comfort region it lands on.

    The rotation is computed in display coordinates, so it tracks the curve as
    drawn rather than its slope in data units.

    Parameters
    ----------
    ax : Axes
        Axis holding the curve.
    x, y : numpy.ndarray
        The curve's visible points, in data coordinates.
    text : str
        Label to draw.
    """
    if x.size < 2:
        return

    index = _label_split_index(x)
    (x0, y0), (x1, y1) = ax.transData.transform(
        [(x[index], y[index]), (x[index + 1], y[index + 1])]
    )
    angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))

    ax.text(
        x[index],
        y[index],
        text,
        color=_PlotDefaults.Psychrometric.rh_label_color,
        zorder=_PlotDefaults.Psychrometric.zorder_rh_lines,
        rotation=angle,
        rotation_mode="anchor",
        ha="center",
        va="center",
    )


class PsychrometricPlot(ThresholdPlot):
    """Configure and render a psychrometric chart with threshold regions.

    Inherits from :class:`ThresholdPlot` and strictly enforces ``hr``
    (humidity ratio) on the y-axis.  Any model temperature parameter
    (``tdb``, ``tr``, etc.) may be used on the x-axis.  Grid evaluation
    converts humidity ratio back to relative humidity before calling the
    underlying model.  Constant-RH background curves are drawn on top of
    the threshold regions.

    .. versionchanged:: 4.5.0
        The y-axis is expressed in **g/kg dry air** rather than kg/kg dry air,
        because typical indoor values (roughly 5-20 g/kg) are far easier to
        read than their 0.005-0.020 kg/kg equivalents.  Pass ``0.0, 30.0``
        where you previously passed ``0.0, 0.030``.  This affects only the
        chart; the psychrometric utilities are unchanged.
        :func:`~pythermalcomfort.utilities.psy_ta_rh` still *returns* humidity
        ratio in kg/kg dry air, and
        :func:`~pythermalcomfort.utilities.hr_to_rh` and
        :func:`~pythermalcomfort.utilities.enthalpy_air` still *accept* it in
        those units.  So multiply by 1000 when plotting ``psy_ta_rh(...).hr``
        here, and divide by 1000 when passing a value read off this chart back
        to those functions.
        A y-axis whose upper bound is below 1 g/kg warns, so an un-migrated
        range does not silently render a blank chart.  It warns rather than
        raises because sub-1 g/kg is physically real in cold or very dry air.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import pmv_ppd_iso
        from pythermalcomfort.plots.matplotlib import PsychrometricPlot

        result = (
            PsychrometricPlot(pmv_ppd_iso)
            .set_x_axis("tdb", 10.0, 36.0)
            .set_y_axis("hr", 0.0, 30.0)
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
        resolution: float | None = None,
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
        resolution : float, optional
            Sampling step along the x-axis.  Optional; see
            :meth:`ThresholdPlot.set_x_axis`.

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
        resolution: float | None = None,
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
            Minimum humidity ratio, [g water / kg dry air].
        max_val : float
            Maximum humidity ratio, [g water / kg dry air].
        resolution : float, optional
            Sampling step along the y-axis, [g water / kg dry air].  Optional;
            see :meth:`ThresholdPlot.set_y_axis`.

        Returns
        -------
        PsychrometricPlot
            Self, to support method chaining.

        Raises
        ------
        ValueError
            If ``name`` is not ``'hr'``, conflicts with a fixed parameter set
            via :meth:`set_params`, or if range/resolution are invalid.

        Warns
        -----
        UserWarning
            If ``max_val`` is below 1 g/kg, which usually means a pre-4.5.0
            kg/kg range, but is legitimate for cold or very dry air.
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
            raise ValueError("x and y axis parameters must be different.")

        min_float, max_float = _parse_axis_range(min_val, max_val)
        if max_float < _MIN_PLAUSIBLE_HR_MAX_G_KG:
            msg = (
                f"The y-axis upper bound is {max_float:g} g/kg dry air. Since "
                "4.5.0 this axis is in g/kg dry air rather than kg/kg, so if "
                f"this range was written for an older version, pass "
                f"{max_float * _G_PER_KG:g} instead of {max_float:g}, and "
                f"scale min_val and resolution by the same factor of "
                f"{_G_PER_KG:g} (an unscaled resolution such as 0.001 can "
                "produce tens of thousands of y-axis grid points). If you "
                "are deliberately charting very cold or very dry air, where "
                "humidity ratios below 1 g/kg are real, this warning can be "
                "ignored."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
        resolution_float = (
            None if resolution is None else _validate_resolution(resolution)
        )
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
        # The axis, and therefore the grid, is in g/kg dry air; hr_to_rh below
        # expects kg/kg dry air.
        y_flat = np.asarray(y).ravel() / _G_PER_KG

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
        grid_kwargs = self._prefer_unrounded_output(grid_kwargs)
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
            with _suppress_applicability_warnings():
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
        show_lines: bool = False,
        line_kws: Mapping[str, Any] | None = None,
        fill_kws: Mapping[str, Any] | None = None,
        legend_kws: Mapping[str, Any] | None = None,
        invalid_color: str = _PlotDefaults.color_out_of_model,
    ) -> ThresholdPlotResult:
        """Render the psychrometric chart with threshold regions and RH curves.

        Delegates to :meth:`ThresholdPlot.plot` for region rendering, then
        overlays:

        - A white fill masking the physically impossible RH > 100 % area,
          starting exactly at the smooth saturation curve.
        - Dotted constant-RH background curves at 25 % intervals.
        - A y-axis label naming the humidity ratio and its units, replacing
          the bare parameter name the base class would otherwise use.  Call
          ``result.ax.set_ylabel(...)`` afterwards to override it.

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
            ``False``; see :meth:`ThresholdPlot.plot`.
        line_kws : dict, optional
            Keyword overrides forwarded to ``ax.plot`` for boundary lines.
        fill_kws : dict, optional
            Keyword overrides forwarded to the region fill.
            Keys ``color`` and ``facecolor`` are reserved and rejected.
        legend_kws : dict, optional
            Keyword overrides forwarded to ``ax.legend``.
        invalid_color : str
            Color used for out-of-model/invalid areas.

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
        # White fill masks the physically impossible RH > 100% region.
        # Super-saturated cells are evaluated at rh=100% rather than NaN, so the
        # regions cover the entire grid and there is no out-of-limits shading
        # underneath for the mask to have to hide.  The mask starts exactly at
        # the smooth saturation curve.
        # psy_ta_rh returns kg/kg dry air; the axis is g/kg dry air.
        hr_100 = psy_ta_rh(t_dense, np.full_like(t_dense, 100.0)).hr * _G_PER_KG
        ax.fill_between(
            t_dense,
            hr_100,
            self._y_axis.max_val,
            color="white",
            zorder=_PlotDefaults.Psychrometric.zorder_rh_mask,
            edgecolor="none",
        )

        # Clip threshold boundary lines to the valid region so they do not
        # extend above the saturation curve.  The clip path is a closed polygon
        # tracing the bottom of the plot → saturation curve (right-to-left) → close.
        if result.lines:
            x_min = self._x_axis.min_val
            x_max = self._x_axis.max_val
            y_min = self._y_axis.min_val
            clip_x = np.concatenate([[x_min, x_max], t_dense[::-1]])
            clip_y = np.concatenate([[y_min, y_min], hr_100[::-1]])
            n = len(clip_x)
            verts = np.column_stack(
                [np.append(clip_x, clip_x[0]), np.append(clip_y, clip_y[0])]
            )
            codes = np.array(
                [MplPath.MOVETO] + [MplPath.LINETO] * (n - 1) + [MplPath.CLOSEPOLY],
                dtype=np.uint8,
            )
            valid_clip = MplPath(verts, codes)
            for line in result.lines:
                line.set_clip_path(valid_clip, ax.transData)

        step = _PlotDefaults.Psychrometric.rh_curve_step
        for rh_target in range(step, 110, step):
            hr_line = (
                psy_ta_rh(t_dense, np.full_like(t_dense, float(rh_target))).hr
                * _G_PER_KG
            )
            # Both bounds, not just the top: on an elevated y window the
            # samples below it still steered where the label went and how wide
            # a gap it cut, which blanked up to a third of the visible curve.
            in_range = (hr_line >= self._y_axis.min_val) & (
                hr_line <= self._y_axis.max_val
            )
            if not in_range.any():
                continue
            curve_t = t_dense[in_range]
            curve_hr = hr_line[in_range]
            # Blank out the stretch the label covers rather than drawing the
            # label on a filled patch: a gap reads as part of the chart, a
            # pale rectangle over a comfort region does not.
            broken = curve_hr.copy()
            if curve_t.size >= 2:
                gap = max(
                    1,
                    round(
                        curve_t.size * _PlotDefaults.Psychrometric.rh_label_gap_fraction
                    ),
                )
                split = _label_split_index(curve_t)
                broken[max(split - gap, 0) : split + gap + 1] = np.nan
            ax.plot(
                curve_t,
                broken,
                color=_PlotDefaults.Psychrometric.rh_line_color,
                linestyle=":",
                linewidth=_PlotDefaults.Psychrometric.rh_line_linewidth,
                zorder=_PlotDefaults.Psychrometric.zorder_rh_lines,
            )
            _label_along_curve(ax, x=curve_t, y=curve_hr, text=f"{rh_target}%")

        # ThresholdPlot.plot() labels the y-axis with the raw parameter name,
        # which here would be the bare string "hr".  Every caller was therefore
        # writing its own label, and they disagreed with each other about the
        # units.  Give the chart a correct default instead; callers who want
        # something else can still override it via result.ax.set_ylabel().
        ax.set_ylabel(_HR_AXIS_LABEL)

        # ThresholdPlot.plot() already clamped these back from whatever the
        # region fills autoscaled to, which is why the RH labels above compute
        # their rotation against the right transform.  Repeating it guards the
        # overlays drawn since, none of which currently autoscale.
        ax.set_xlim(self._x_axis.min_val, self._x_axis.max_val)
        ax.set_ylim(self._y_axis.min_val, self._y_axis.max_val)

        return result
