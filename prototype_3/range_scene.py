"""Prototype 3: class-based threshold-region scene."""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


class Output(str, Enum):
    """Common model outputs we want to support in prototype_3."""

    PMV = "pmv"
    PPD = "ppd"
    UTCI = "utci"


# This is a small container for one axis configuration
@dataclass
class AxisConfig:
    name: str
    min_val: float
    max_val: float


def _parse_axis_range(param_name: str, value: Any) -> tuple[float, float]:
    if not isinstance(value, tuple | list) or len(value) != 2:
        msg = f"Axis '{param_name}' must be a tuple/list of length 2: (min, max)."
        raise ValueError(msg)
    try:
        min_val = float(value[0])
        max_val = float(value[1])
    except (TypeError, ValueError) as exc:
        msg = f"Axis '{param_name}' range values must be numeric."
        raise ValueError(msg) from exc
    if min_val >= max_val:
        msg = f"Axis '{param_name}' requires min < max (got {min_val} >= {max_val})."
        raise ValueError(msg)
    return min_val, max_val


def _extract_output_value(result: Any, output: Output) -> float:
    """Extract scalar value from model result for a selected output."""
    if output is Output.PMV:
        if hasattr(result, "pmv"):
            return float(result.pmv)
        if isinstance(result, dict) and "pmv" in result:
            return float(result["pmv"])
        raise ValueError("Selected output PMV is not available in model result.")

    if output is Output.PPD:
        if hasattr(result, "ppd"):
            return float(result.ppd)
        if isinstance(result, dict) and "ppd" in result:
            return float(result["ppd"])
        raise ValueError("Selected output PPD is not available in model result.")

    if output is Output.UTCI:
        if hasattr(result, "utci"):
            return float(result.utci)
        if isinstance(result, dict) and "utci" in result:
            return float(result["utci"])
        try:
            return float(result)
        except Exception as exc:
            raise ValueError(
                "Selected output UTCI could not be extracted from result."
            ) from exc

    msg = f"Unsupported output: {output}"
    raise ValueError(msg)


def _default_region_colors(n_regions: int) -> list[str]:
    """Return default region colors in a cool-neutral-warm progression."""
    if n_regions < 1:
        raise ValueError("n_regions must be at least 1.")
    if n_regions == 1:
        return ["#f2f2f2"]
    if n_regions == 2:
        return ["#4c78a8", "#e15759"]
    if n_regions == 3:
        return ["#4c78a8", "#f2f2f2", "#e15759"]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "range_scene_blue_neutral_red",
        ["#4c78a8", "#f2f2f2", "#e15759"],
    )
    positions = np.linspace(0.0, 1.0, n_regions)
    return [mcolors.to_hex(cmap(v)) for v in positions]


class RangeScene:
    def __init__(self, model_func: Any, allow_aliases: bool = False) -> None:
        self.model_func = model_func
        self._allow_aliases = allow_aliases

        self.x_axis: AxisConfig | None = None
        self.y_axis: AxisConfig | None = None
        self._default_links: dict[str, str] = {"tr": "tdb"}
        self.fixed_values: dict[str, Any] = {}

        self.line_artists: list[Line2D] = []
        self.fill_artists: list[PolyCollection] = []

        self._signature: inspect.Signature | None = None
        self._allowed_args: set[str] = set()
        self._required_args: set[str] = set()
        self._accepts_var_kwargs: bool = False

    def allowed_parameters(self) -> list[str]:
        """Return sorted parameter names accepted by the model signature."""
        self._read_model_signature()
        return sorted(self._allowed_args)

    def required_parameters(self) -> list[str]:
        """Return sorted required parameter names from the model signature."""
        self._read_model_signature()
        return sorted(self._required_args)

    def axes(self, **axes_ranges: Any) -> RangeScene:
        self._read_model_signature()

        if len(axes_ranges) != 2:
            raise ValueError("axes() requires exactly two keyword ranges.")

        (x_name, x_range), (y_name, y_range) = axes_ranges.items()

        if x_name == y_name:
            raise ValueError("x and y axis parameters must be different.")

        invalid = [name for name in (x_name, y_name) if name not in self._allowed_args]
        if invalid:
            invalid_str = ", ".join(invalid)
            msg = (
                f"axes() received invalid parameter(s): {invalid_str}. "
                "Use model argument names from the selected function."
            )
            raise ValueError(msg)
        # extracting the min and max from the ranges using the already defined parse function
        x_min, x_max = _parse_axis_range(x_name, x_range)
        y_min, y_max = _parse_axis_range(y_name, y_range)

        axis_conflicts = [
            name for name in (x_name, y_name) if name in self.fixed_values
        ]
        if axis_conflicts:
            conflict_str = ", ".join(axis_conflicts)
            msg = (
                f"fixed() already contains axis parameter(s): {conflict_str}. "
                "Remove them from fixed() when using axes()."
            )
            raise ValueError(msg)

        self.x_axis = AxisConfig(name=x_name, min_val=x_min, max_val=x_max)
        self.y_axis = AxisConfig(name=y_name, min_val=y_min, max_val=y_max)
        return self

    def fixed(self, **kwargs: Any) -> RangeScene:
        """Set fixed model parameters that are not used as axes.

        Example:
            scene.fixed(vr=0.1, met=1.2, clo=0.5)
        """
        self._read_model_signature()
        fixed_kwargs = self._apply_v_vr_alias(kwargs)

        if not self._accepts_var_kwargs:
            invalid = sorted(k for k in fixed_kwargs if k not in self._allowed_args)
            if invalid:
                invalid_str = ", ".join(invalid)
                msg = (
                    f"fixed() received invalid parameter(s): {invalid_str}. "
                    "Use scene.allowed_parameters() to inspect valid keys."
                )
                raise ValueError(msg)

        x_name = self.x_axis.name if self.x_axis is not None else None
        y_name = self.y_axis.name if self.y_axis is not None else None
        for key, value in fixed_kwargs.items():
            if key in (x_name, y_name):
                msg = f"fixed() cannot set axis parameter '{key}'. Set it through axes(...)."
                raise ValueError(msg)
            self.fixed_values[key] = value
        return self

    def _read_model_signature(self) -> None:
        """Read model signature once and cache required/allowed argument names."""
        if self._signature is not None:
            return

        self._signature = inspect.signature(self.model_func)
        self._allowed_args = set(self._signature.parameters.keys())
        self._accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in self._signature.parameters.values()
        )

        self._required_args = {
            name
            for name, p in self._signature.parameters.items()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
            and p.default is inspect._empty
        }

    def _apply_v_vr_alias(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Handle v/vr aliasing explicitly based on model expectations."""
        self._read_model_signature()
        out = dict(kwargs)

        expects_v = "v" in self._allowed_args
        expects_vr = "vr" in self._allowed_args

        if expects_vr and not expects_v:
            if "v" in out and "vr" in out:
                raise ValueError(
                    "Model expects 'vr' and does not accept 'v'. "
                    "Provide only vr=... for this model."
                )
            if "v" in out and "vr" not in out:
                if not self._allow_aliases:
                    raise ValueError(
                        "Model expects 'vr'. Use vr=... for this model. "
                        "To auto-convert v->vr, initialize RangeScene(..., allow_aliases=True)."
                    )
                warnings.warn(
                    "Converting fixed/model kwargs key 'v' to 'vr' (allow_aliases=True).",
                    stacklevel=2,
                )
                out["vr"] = out.pop("v")
        elif expects_v and not expects_vr:
            if "vr" in out and "v" in out:
                raise ValueError(
                    "Model expects 'v' and does not accept 'vr'. "
                    "Provide only v=... for this model."
                )
            if "vr" in out and "v" not in out:
                if not self._allow_aliases:
                    raise ValueError(
                        "Model expects 'v'. Use v=... for this model. "
                        "To auto-convert vr->v, initialize RangeScene(..., allow_aliases=True)."
                    )
                warnings.warn(
                    "Converting fixed/model kwargs key 'vr' to 'v' (allow_aliases=True).",
                    stacklevel=2,
                )
                out["v"] = out.pop("vr")

        return out

    def _validate_call_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Validate model-call kwargs against signature."""
        self._read_model_signature()

        if not self._accepts_var_kwargs:
            invalid = sorted(k for k in kwargs if k not in self._allowed_args)
            if invalid:
                invalid_str = ", ".join(invalid)
                msg = f"Model does not accept parameter(s): {invalid_str}"
                raise ValueError(msg)

        missing = sorted(k for k in self._required_args if k not in kwargs)
        if missing:
            missing_str = ", ".join(missing)
            msg = f"Missing required parameter(s): {missing_str}"
            raise ValueError(msg)

    def _build_call_kwargs(self, x_value: float, y_value: float) -> dict[str, Any]:
        """Build model kwargs for one x/y point."""
        if self.x_axis is None or self.y_axis is None:
            raise ValueError("Axes are not set. Call axes(...) first.")
        self._read_model_signature()

        call_kwargs: dict[str, Any] = dict(self.fixed_values)
        call_kwargs[self.x_axis.name] = float(x_value)
        call_kwargs[self.y_axis.name] = float(y_value)

        for target, source in self._default_links.items():
            if (
                target in self._allowed_args
                and source in self._allowed_args
                and target not in call_kwargs
                and source in call_kwargs
            ):
                call_kwargs[target] = call_kwargs[source]

        # Apply guided v/vr handling, then validate complete call kwargs.
        call_kwargs = self._apply_v_vr_alias(call_kwargs)
        self._validate_call_kwargs(call_kwargs)
        return call_kwargs

    def _compute_threshold_curves(
        self,
        *,
        output: Output,
        thresholds: list[float],
        x_step: float,
        y_step: float,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Compute x(y) threshold curves by scanning temperature for each RH."""
        if self.x_axis is None or self.y_axis is None:
            raise ValueError("Axes are not set. Call axes(...) first.")
        if x_step <= 0 or y_step <= 0:
            raise ValueError("x_step and y_step must be positive.")

        if (
            self.x_axis.min_val is None
            or self.x_axis.max_val is None
            or self.y_axis.min_val is None
            or self.y_axis.max_val is None
        ):
            raise ValueError("Axes are not fully set. Call axes(...) first.")

        x_values = np.arange(self.x_axis.min_val, self.x_axis.max_val + 1e-12, x_step)
        y_values = np.arange(self.y_axis.min_val, self.y_axis.max_val + 1e-12, y_step)

        fill_curves: list[np.ndarray] = []
        line_curves: list[np.ndarray] = []
        x_min = float(self.x_axis.min_val)
        x_max = float(self.x_axis.max_val)
        has_any_valid_output = False

        for _ in thresholds:
            fill_curves.append(np.full(len(y_values), np.nan, dtype=float))
            line_curves.append(np.full(len(y_values), np.nan, dtype=float))

        for i, y in enumerate(y_values):
            z = np.full(len(x_values), np.nan, dtype=float)
            for j, x in enumerate(x_values):
                try:
                    kwargs = self._build_call_kwargs(float(x), float(y))
                    result = self.model_func(**kwargs)
                    z_val = _extract_output_value(result, output)
                    z[j] = z_val
                    if np.isfinite(z_val):
                        has_any_valid_output = True
                except Exception:
                    z[j] = np.nan
                    continue

            if not np.isfinite(z).any():
                continue

            for k, threshold in enumerate(thresholds):
                r = z - float(threshold)

                crossing_x = np.nan
                found_crossing = False
                for j in range(len(x_values) - 1):
                    x0, x1 = x_values[j], x_values[j + 1]
                    r0, r1 = r[j], r[j + 1]

                    if not np.isfinite(r0) or not np.isfinite(r1):
                        continue

                    if r0 == 0.0:
                        crossing_x = float(x0)
                        found_crossing = True
                        break

                    if r0 * r1 < 0.0:
                        crossing_x = float(x0 + (-r0) * (x1 - x0) / (r1 - r0))
                        found_crossing = True
                        break

                    if r1 == 0.0:
                        crossing_x = float(x1)
                        found_crossing = True
                        break

                if found_crossing:
                    fill_curves[k][i] = crossing_x
                    line_curves[k][i] = crossing_x
                    continue

                finite = np.isfinite(r)
                if not finite.any():
                    continue

                rf = r[finite]
                if np.all(rf < 0.0):
                    fill_curves[k][i] = x_max
                elif np.all(rf > 0.0):
                    fill_curves[k][i] = x_min
                else:
                    continue

        if not has_any_valid_output:
            raise ValueError(
                "No valid model outputs could be computed for this plot. "
                "Check fixed parameters and axis ranges."
            )

        return y_values, fill_curves, line_curves

    def plot(
        self,
        *,
        output: Output,
        levels: list[float],
        colors: list[str] | None = None,
        alpha: float = 0.65,
        x_step: float,
        y_step: float,
        ax: plt.Axes | None = None,
        legend: bool = True,
        legend_title: str = "Regions",
        legend_loc: str = "upper left",
        **line_kwargs: Any,
    ) -> plt.Axes:
        if self.x_axis is None or self.y_axis is None:
            raise ValueError("Axes are not set. Call axes(...) first.")
        if x_step <= 0 or y_step <= 0:
            raise ValueError("x_step and y_step must be positive.")
        if len(levels) == 0:
            raise ValueError("levels must contain at least one threshold.")
        if (
            self.x_axis.min_val is None
            or self.x_axis.max_val is None
            or self.y_axis.min_val is None
            or self.y_axis.max_val is None
        ):
            raise ValueError("Axes are not fully set. Call axes(...) first.")

        sorted_levels = sorted(float(v) for v in levels)

        n_regions = len(sorted_levels) + 1
        if colors is None:
            region_colors = _default_region_colors(n_regions)
        else:
            if len(colors) != n_regions:
                msg = f"colors must have length {n_regions} (got {len(colors)})."
                raise ValueError(msg)
            region_colors = colors

        if ax is None:
            _, ax = plt.subplots(figsize=(9, 6))

        y_values, fill_curves, line_curves = self._compute_threshold_curves(
            output=output,
            thresholds=sorted_levels,
            x_step=x_step,
            y_step=y_step,
        )

        x_lo = float(self.x_axis.min_val)
        x_hi = float(self.x_axis.max_val)
        left_const = np.full_like(y_values, x_lo, dtype=float)
        right_const = np.full_like(y_values, x_hi, dtype=float)

        self.fill_artists = []
        self.line_artists = []

        region_pairs: list[tuple[np.ndarray, np.ndarray]] = [
            (left_const, fill_curves[0]),
            *[
                (fill_curves[i], fill_curves[i + 1])
                for i in range(len(fill_curves) - 1)
            ],
            (fill_curves[-1], right_const),
        ]

        for i, (x_left, x_right) in enumerate(region_pairs):
            valid = np.isfinite(x_left) & np.isfinite(x_right)
            if valid.any():
                poly = ax.fill_betweenx(
                    y_values[valid],
                    x_left[valid],
                    x_right[valid],
                    color=region_colors[i],
                    alpha=alpha,
                )
                self.fill_artists.append(poly)

        for curve in line_curves:
            valid = np.isfinite(curve)
            if valid.any():
                (line,) = ax.plot(
                    curve[valid],
                    y_values[valid],
                    **line_kwargs,
                )
                self.line_artists.append(line)

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(float(self.y_axis.min_val), float(self.y_axis.max_val))
        ax.set_xlabel(self.x_axis.name)
        ax.set_ylabel(self.y_axis.name)

        # Auto-legend for filled threshold regions.
        if legend:
            out_name = output.value.upper()
            labels: list[str] = []
            for i in range(len(region_colors)):
                if i == 0:
                    labels.append(f"{out_name} < {sorted_levels[0]:g}")
                elif i == len(region_colors) - 1:
                    labels.append(f"{out_name} > {sorted_levels[-1]:g}")
                else:
                    lo = sorted_levels[i - 1]
                    hi = sorted_levels[i]
                    labels.append(f"{lo:g} <= {out_name} <= {hi:g}")

            handles = [
                Patch(facecolor=region_colors[i], alpha=alpha, label=labels[i])
                for i in range(len(region_colors))
            ]
            ax.legend(handles=handles, title=legend_title, loc=legend_loc)

        return ax

    def adjust_lines(self, **kwargs: Any) -> RangeScene:
        """Apply native Matplotlib line kwargs to all threshold lines."""
        if not self.line_artists:
            raise ValueError("Call plot() first.")
        for line in self.line_artists:
            line.set(**kwargs)
        return self

    def adjust_fills(self, **kwargs: Any) -> RangeScene:
        """Apply native Matplotlib fill kwargs to all threshold regions."""
        if not self.fill_artists:
            raise ValueError("Call plot() first.")
        for fill in self.fill_artists:
            fill.set(**kwargs)
        return self
