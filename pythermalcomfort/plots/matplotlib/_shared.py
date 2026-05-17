"""Shared helpers for Matplotlib threshold and summary plots."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Number
from typing import Any

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# ── axis helpers ───────────────────────────────────────────────────────────


@dataclass
class _AxisConfig:
    """Axis plotting configuration."""

    name: str
    min_val: float
    max_val: float
    resolution: float


def _parse_axis_range(min_val: Any, max_val: Any) -> tuple[float, float]:
    """Validate and normalize axis bounds."""
    try:
        min_float = float(min_val)
        max_float = float(max_val)
    except (TypeError, ValueError) as exc:
        raise ValueError("Axis range values must be numeric.") from exc

    if min_float >= max_float:
        msg = f"Axis requires min < max (got {min_float} >= {max_float})."
        raise ValueError(msg)

    return min_float, max_float


def _validate_resolution(resolution: Any) -> float:
    """Validate axis resolution."""
    try:
        resolution_float = float(resolution)
    except (TypeError, ValueError) as exc:
        raise ValueError("Axis resolution must be numeric.") from exc
    if resolution_float <= 0:
        raise ValueError("Axis resolution must be positive.")
    return resolution_float


# ── visual defaults ────────────────────────────────────────────────────────


class _PlotDefaults:
    """Central registry of visual defaults shared across all Matplotlib plot types.

    Top-level attributes are shared across all plot types for a consistent
    look and feel.  Nested classes group plot-specific defaults by type.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.plots.matplotlib._shared import _PlotDefaults

        color = _PlotDefaults.color_out_of_model
        size = _PlotDefaults.figsize
        alpha = _PlotDefaults.fill_alpha
        fsize = _PlotDefaults.title_fontsize
    """

    # ── shared across all plot types ───────────────────────────────────────
    color_out_of_model: str = "#bdbdbd"
    parameter_links: dict = {"tr": "tdb", "tdb": "tr"}
    figsize: tuple = (7, 4)
    fill_alpha: float = 1.0
    title_fontsize: int = 13
    # When legend and title are both shown, the legend sits just above the axes
    # and the title floats above the legend.
    legend_bbox_to_anchor_with_title: tuple = (0.5, 1.05)
    title_y_with_legend: float = 1.15

    class Threshold:
        """Defaults specific to :class:`ThresholdPlot`."""

        fill_corner_mask: bool = False
        line_color: str = "black"
        line_linewidth: float = 1.0
        legend_loc: str = "lower center"
        legend_bbox_to_anchor: tuple = (0.5, 1.02)
        legend_ncol_max: int = 4
        zorder_invalid: float = 1.5

    class Adaptive:
        """Defaults specific to :class:`AdaptivePlot`."""

        n_points: int = 200
        center_line_label: str = "Comfort Temperature"
        center_line_defaults: dict = {
            "color": "#333333",
            "linewidth": 1.5,
            "linestyle": "--",
        }
        legend_loc: str = "lower right"
        # fixme the legend height in this plot is not the same as in the psychrometric chart
        legend_ncol: int = 3

    class Psychrometric:
        """Defaults specific to :class:`PsychrometricPlot`."""

        p_atm: float = 101325.0
        n_tdb_points: int = 500
        rh_line_color: str = "#a0a0a0"
        rh_line_linewidth: float = 0.8
        rh_label_fontsize: int = 8
        rh_label_offset_fraction: float = 0.01
        rh_curve_step: int = 10
        zorder_rh_mask: float = 4.5  
        zorder_rh_lines: float = 5.0

    class Summary:
        """Defaults specific to :class:`SummaryPlot`."""

        bar_edgecolor: str = "none"
        bar_linewidth: float = 0.0
        percentage_fontsize: int = 12
        label_fontsize: int = 11
        pct_min_to_show: float = 5.0
        h_xlim: tuple = (0.0, 100.0)
        h_ylim: tuple = (-0.25, 0.42)          
        h_ylim_legend: tuple = (-0.25, 0.25)   
        h_bar_y: float = 0.0
        h_bar_height: float = 0.5  
        h_label_y: float = 0.29
        v_xlim: tuple = (-0.50, 0.90)  # with side labels
        v_xlim_legend: tuple = (-0.50, 0.50)  # legend handles labels: bar at ~80%
        v_ylim: tuple = (0.0, 100.0)
        v_bar_x: float = 0.0
        v_bar_width: float = 0.80  # bar fills most of the horizontal span
        v_label_x_offset: float = 0.42  # just past bar right edge (0.40)


# ── package-wide Matplotlib style ─────────────────────────────────────────

#: rcParams applied via ``mpl.rc_context`` inside every ``plot()`` call.
_PYTHERMALCOMFORT_RC: dict[str, Any] = {
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,
}

# ── internal resolved container ────────────────────────────────────────────


@dataclass
class RegionConfig:
    """Fully-resolved region configuration (internal use only).

    Attributes
    ----------
    output_name : str
        Validated output column / field name.
    thresholds : list of float
        Sorted, finite threshold boundary values.
    labels : list of str
        Human-readable label for every region (length = ``len(thresholds) + 1``).
    colors : list of str
        Matplotlib-compatible color for every region
        (length = ``len(thresholds) + 1``).
    """

    output_name: str
    thresholds: list[float]
    labels: list[str]
    colors: list[str]


@dataclass
class BasePlotResult:
    """Minimal result handle shared by all plot types.

    Attributes
    ----------
    fig : Figure
        Matplotlib figure containing the rendered plot.
    ax : Axes
        Matplotlib axis containing the rendered plot.
    """

    fig: Figure
    ax: Axes


# ── model-signature helpers ────────────────────────────────────────────────


def _inspect_model_signature(
    model_func: Any,
) -> tuple[inspect.Signature, set[str], set[str], bool]:
    """Inspect a model callable and return argument metadata."""
    signature = inspect.signature(model_func)
    allowed_args = set(signature.parameters.keys())
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    required_args = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
        and parameter.default is inspect.Signature.empty
    }
    return signature, allowed_args, required_args, accepts_var_kwargs


def _validate_model_kwargs(
    kwargs: Mapping[str, Any],
    *,
    allowed_args: set[str],
    required_args: set[str],
    accepts_var_kwargs: bool,
) -> None:
    """Validate kwargs against a model signature contract."""
    if not accepts_var_kwargs:
        invalid = sorted(key for key in kwargs if key not in allowed_args)
        if invalid:
            invalid_str = ", ".join(invalid)
            msg = f"Model does not accept parameter(s): {invalid_str}"
            raise ValueError(msg)

    missing = sorted(key for key in required_args if key not in kwargs)
    if missing:
        missing_str = ", ".join(missing)
        msg = f"Missing required parameter(s): {missing_str}"
        raise ValueError(msg)


# ── output extraction ──────────────────────────────────────────────────────


def _extract_output_by_name(result: Any, output: str) -> Any:
    """Extract an output payload from a model result by name."""
    output_name = output.strip()
    if not output_name:
        raise ValueError("output must be a non-empty string.")

    candidates = [output_name]
    lowered = output_name.lower()
    if lowered != output_name:
        candidates.append(lowered)

    for candidate in candidates:
        if hasattr(result, candidate):
            return getattr(result, candidate)

    if isinstance(result, Mapping):
        for candidate in candidates:
            if candidate in result:
                return result[candidate]

    if isinstance(result, Number) and not isinstance(result, bool):
        if lowered in {"value", "scalar"}:
            return result
        msg = (
            f"Could not extract output '{output_name}' from scalar model result. "
            "Use output='value' for scalar-returning models."
        )
        raise ValueError(msg)

    available_outputs: list[str] = []
    if isinstance(result, Mapping):
        available_outputs = [str(key) for key in result.keys()]
    else:
        result_dict = getattr(result, "__dict__", None)
        if isinstance(result_dict, dict):
            available_outputs = [
                name
                for name, value in result_dict.items()
                if not name.startswith("_") and not callable(value)
            ]

    msg = f"Could not extract output '{output_name}' from model result."
    if available_outputs:
        shown = ", ".join(sorted(available_outputs)[:6])
        if len(available_outputs) > 6:
            shown = f"{shown}, ..."
        msg = f"{msg} Available outputs: {shown}."
    raise ValueError(msg)


# ── threshold / level helpers ──────────────────────────────────────────────


def _normalize_levels(levels: Sequence[float]) -> list[float]:
    """Validate and normalize threshold levels."""
    if len(levels) == 0:
        raise ValueError("thresholds must contain at least one threshold.")

    try:
        normalized = sorted(float(level) for level in levels)
    except (TypeError, ValueError) as exc:
        raise ValueError("thresholds must contain only numeric values.") from exc

    if not all(np.isfinite(normalized)):
        raise ValueError("thresholds must contain only finite values.")

    if any(
        right <= left for left, right in zip(normalized, normalized[1:], strict=False)
    ):
        raise ValueError("thresholds must be strictly increasing after sorting.")

    return normalized


def _build_region_labels(
    *,
    output: str,
    levels: Sequence[float],
    labels: Sequence[str] | None = None,
) -> list[str]:
    """Build labels for threshold regions from already-normalized levels."""
    output_name = output.strip()
    if not output_name:
        raise ValueError("output must be a non-empty string.")

    n_regions = len(levels) + 1
    if labels is not None:
        if len(labels) == 0:
            return [""] * n_regions
        if len(labels) != n_regions:
            msg = f"labels must have length {n_regions} (got {len(labels)})."
            raise ValueError(msg)
        return [str(label) for label in labels]

    out_name = output_name.upper()
    region_labels = [f"{out_name} < {levels[0]:g}"]
    for lower, upper in zip(levels, levels[1:], strict=False):
        region_labels.append(f"{lower:g} ≤ {out_name} < {upper:g}")
    region_labels.append(f"{out_name} ≥ {levels[-1]:g}")
    return region_labels


# ── color helpers ──────────────────────────────────────────────────────────


def _default_region_colors(n_regions: int) -> list[str]:
    """Return default region colors in a cool-neutral-warm progression."""
    if n_regions < 1:
        raise ValueError("n_regions must be at least 1.")
    if n_regions == 1:
        return ["#008D3D"]
    if n_regions == 2:
        return ["#0067B2", "#C40025"]
    if n_regions == 3:
        return ["#A3D1FF", "#A8E6CF", "#FFB7B2"]

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "summary_blue_neutral_red",
        [
            "#8DA2D8",
            "#A9D4F5",
            "#D1EAFA",
            "#D2E8BF",
            "#F0BECB",
            "#DE7B6A",
            "#A65558",
        ],
    )
    positions = np.linspace(0.0, 1.0, n_regions)
    return [mcolors.to_hex(cmap(value)) for value in positions]


def _resolve_region_colors(
    *,
    n_regions: int,
    colors: Sequence[str] | None = None,
) -> list[str]:
    """Resolve user or default region colors and validate length/format."""
    if colors is None:
        return _default_region_colors(n_regions)

    resolved = [str(color) for color in colors]
    if len(resolved) != n_regions:
        msg = f"colors must have length {n_regions} (got {len(resolved)})."
        raise ValueError(msg)
    invalid = [color for color in resolved if not mcolors.is_color_like(color)]
    if invalid:
        msg = f"Invalid color value(s): {', '.join(invalid)}."
        raise ValueError(msg)
    return resolved


# ── region configuration factory ───────────────────────────────────────────


def _configure_regions(
    *,
    output: str,
    thresholds: Sequence[float],
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
) -> RegionConfig:
    """Validate inputs and build a :class:`RegionConfig`.

    This is the single source of truth for region configuration shared by
    :class:`SummaryPlot` and :class:`ThresholdPlot`.

    Parameters
    ----------
    output : str
        Output column / field name.
    thresholds : sequence of float
        Boundary values that divide the output range into regions.
    labels : sequence of str, optional
        Human-readable label for every region.  Must have length
        ``len(thresholds) + 1`` when provided.
    colors : sequence of str, optional
        Matplotlib-compatible color for every region.  Must have length
        ``len(thresholds) + 1`` when provided.

    Returns
    -------
    RegionConfig
        A fully validated :class:`RegionConfig`.

    Raises
    ------
    TypeError
        If *output* is not a string.
    ValueError
        If *output* is empty, or thresholds / labels / colors are invalid.
    """
    if not isinstance(output, str):
        raise TypeError("output must be a string.")
    output_name = output.strip()
    if not output_name:
        raise ValueError("output must be a non-empty string.")

    normalized_levels = _normalize_levels(thresholds)
    region_labels = _build_region_labels(
        output=output_name,
        levels=normalized_levels,
        labels=labels,
    )
    region_colors = _resolve_region_colors(
        n_regions=len(normalized_levels) + 1,
        colors=colors,
    )

    return RegionConfig(
        output_name=output_name,
        thresholds=normalized_levels,
        labels=region_labels,
        colors=region_colors,
    )


# ── default-link helpers ───────────────────────────────────────────────────


def _apply_default_links_to_kwargs(
    kwargs: dict[str, Any],
    *,
    allowed_args: set[str],
    default_links: Mapping[str, str],
) -> dict[str, Any]:
    """Apply implicit parameter links (e.g. tr <-> tdb) to *kwargs*."""
    resolved = dict(kwargs)
    for target, source in default_links.items():
        if (
            (not allowed_args or target in allowed_args)
            and (not allowed_args or source in allowed_args)
            and target not in resolved
            and source in resolved
        ):
            resolved[target] = resolved[source]
    return resolved


# ── colour utilities ───────────────────────────────────────────────────────


def _is_light_color(color: str) -> bool:
    """Return ``True`` when *color* has a perceived luminance above 0.7.

    Uses WCAG 2.0 channel coefficients applied directly to sRGB values
    (gamma linearisation is intentionally skipped for simplicity).
    Accurate enough for choosing contrasting text colour (black vs. white).
    """
    red, green, blue = mcolors.to_rgb(color)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return luminance > 0.7

# ── dynamic layout helpers ─────────────────────────────────────────────────

def _resolve_layout(title: str | None, labels: Sequence[str], default_ncol: int, is_summary: bool = False) -> tuple[int, float, float]:
    """Dynamically calculate legend columns, legend Y anchor, and title pad to prevent overlaps."""
    n_labels = len(labels)
    if n_labels == 0:
        ncol = default_ncol
        n_legend_rows = 0
    else:
        total_chars = sum(len(str(lbl)) for lbl in labels)
        if total_chars > 45 and n_labels > 2:
            ncol = max(2, (n_labels + 1) // 2)
        else:
            ncol = min(n_labels, default_ncol)
        n_legend_rows = (n_labels + ncol - 1) // ncol

    # The legend is uniformly anchored above the upper edge of the coordinate axis
    legend_anchor_y = 1.01 if is_summary else 1.02

    if n_legend_rows == 0:
        base_pad = 6.0 if is_summary else 8.0
    elif n_legend_rows == 1:
        base_pad = 32.0 if is_summary else 34.0  
    else:
        base_pad = (32.0 if is_summary else 34.0) + (n_legend_rows - 1) * (12.0 if is_summary else 16.0)

    if title:
        newlines = title.count('\n')
        title_pad = base_pad + (newlines * (10.0 if is_summary else 12.0))
    else:
        title_pad = base_pad

    return ncol, legend_anchor_y, title_pad