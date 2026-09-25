"""Shared helpers for Matplotlib threshold and summary plots."""

from __future__ import annotations

import contextlib
import inspect
import math
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Number
from types import MappingProxyType
from typing import Any

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import BboxBase

# ── axis helpers ───────────────────────────────────────────────────────────


@dataclass
class _AxisConfig:
    """Axis plotting configuration."""

    name: str
    min_val: float
    max_val: float
    resolution: float | None


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
    color_out_of_model: str = "#C4C9CC"
    parameter_links: MappingProxyType = MappingProxyType({"tr": "tdb", "tdb": "tr"})
    figsize: tuple = (7, 4)
    fill_alpha: float = 1.0
    title_fontsize: int = 13
    max_labeled_ticks: int = 6
    # When legend and title are both shown, the legend sits just above the axes
    # and the title floats above the legend.
    legend_bbox_to_anchor_with_title: tuple = (0.5, 1.05)
    title_y_with_legend: float = 1.15
    #: Height of one legend row in axes coordinates.  A title sitting above a
    #: legend has to clear every row, so the offset is per-row rather than
    #: fixed: a five-region chart wraps its legend onto two rows and the old
    #: fixed offset put the title straight through it.  Measured rather than
    #: guessed -- 0.10 left the title overlapping the legend by 0.015 even at
    #: one row, so titles were always very slightly clipped.  Deliberately a
    #: fixed model rather than measuring the drawn legend: callers who apply
    #: ``constrained_layout`` or ``tight_layout`` afterwards move everything,
    #: which would leave a measured position stale.  0.12 is the smallest
    #: value that clears the legend at both one and two rows with a little
    #: margin; 0.115 just clears it, and 0.10 overlapped even at one row.
    title_legend_row_height: float = 0.12

    class Threshold:
        """Defaults specific to :class:`ThresholdPlot`."""

        fill_corner_mask: bool = False
        # Rows and scan samples used when solving boundaries.  The scan samples
        # only have to separate one crossing from the next -- bisection
        # supplies the precision -- while the rows set how finely the boundary
        # curves themselves are sampled.
        curve_min_rows: int = 200
        # Doubled from 65: the scan spacing sets the smallest feature the
        # solver can see, so a finer floor halves the width of an invalid
        # pocket or a close pair of crossings that could slip between samples.
        # Costs ~20% on pmv_ppd_iso and ~50% on set_tmp, both still well under
        # a third of a second.
        curve_min_scan_samples: int = 129
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
        center_line_defaults: MappingProxyType = MappingProxyType(
            {
                "color": "#333333",
                "linewidth": 1.5,
                "linestyle": "--",
            }
        )
        legend_loc: str = "lower right"
        # fixme the legend height in this plot is not the same as in the psychrometric chart
        legend_ncol: int = 3

    class Psychrometric:
        """Defaults specific to :class:`PsychrometricPlot`."""

        p_atm: float = 101325.0
        n_tdb_points: int = 500
        rh_line_color: str = "#a0a0a0"
        rh_line_linewidth: float = 0.8
        #: Labels are darker than their curves: the line can be faint because
        #: it is background, but the text has to be read.
        rh_label_color: str = "#6b6b6b"
        #: Half-width of the gap left for the label, as a fraction of the
        #: curve's *visible* length.  A fixed sample count would blank most of
        #: a short curve: on a sub-1 g/kg chart -- legitimate for cold air --
        #: the 100 % curve keeps only ~49 of its 500 samples, and 11 samples
        #: either side of the label erased almost half of it.
        rh_label_gap_fraction: float = 0.023
        #: Where along each visible RH curve its label sits, as a fraction of
        #: the curve's in-range span.  Just short of the end keeps the label
        #: inside the axes while staying out of the busy lower-left corner.
        rh_label_position: float = 0.93
        rh_curve_step: int = 25
        zorder_rh_mask: float = 1.6
        zorder_rh_lines: float = 2.0

    class Summary:
        """Defaults specific to :class:`SummaryPlot`."""

        bar_edgecolor: str = "white"
        bar_linewidth: float = 1.0
        layout_pad: float = 0.2
        legend_ncol_max: int = 3
        title_legend_gap: float = 5.0
        title_spacing_iterations: int = 3
        vertical_legend_ncol: int = 1
        figsize_horizontal_legend: tuple = (6.4, 1.8)
        figsize_horizontal_labels: tuple = (6.4, 1.4)
        figsize_horizontal_plain: tuple = (6.4, 1.1)
        figsize_vertical_labels: tuple = (3.2, 4.0)
        figsize_vertical_legend: tuple = (2.8, 4.0)
        figsize_vertical_plain: tuple = (2.2, 4.0)
        percentage_fontsize: int = 12
        label_fontsize: int = 11
        pct_min_to_show: float = 5.0
        h_xlim: tuple = (0.0, 100.0)
        h_ylim: tuple = (-0.48, 0.72)  # with side labels: bar at 60% of height
        h_ylim_legend: tuple = (-0.48, 0.48)  # legend handles labels: bar at ~75%
        h_bar_y: float = 0.0
        h_bar_height: float = 0.72  # bar fills most of the vertical span
        h_label_y: float = 0.38  # sits just above bar top (0.36)
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


def _legend_anchor_y(
    bbox_to_anchor: Any, *, ax: Axes, bbox_transform: Any = None
) -> float:
    """Read a ``bbox_to_anchor`` y coordinate, in axes coordinates.

    ``ax.legend`` takes a 2-tuple, a 4-tuple or a ``BboxBase``, and only the
    tuples are subscriptable, so a caller passing a ``Bbox`` used to crash here
    before their legend was ever drawn.

    It also takes a ``bbox_transform``, which says what the anchor is measured
    in.  A caller anchoring the legend in figure coordinates was getting that
    raw figure value back, while ``ax.set_title(y=...)`` reads axes
    coordinates, so the title landed somewhere unrelated to the legend.

    Parameters
    ----------
    bbox_to_anchor : BboxBase or tuple
        The anchor as passed to ``ax.legend``.
    ax : Axes
        Axis the legend belongs to.
    bbox_transform : Transform, optional
        The anchor's coordinate system.  ``None`` means axes coordinates,
        which is what ``ax.legend`` itself assumes.

    Returns
    -------
    float
        The anchor's lower y coordinate, in axes coordinates.
    """
    if isinstance(bbox_to_anchor, BboxBase):
        anchor_x, anchor_y = float(bbox_to_anchor.x0), float(bbox_to_anchor.y0)
    else:
        anchor_x, anchor_y = float(bbox_to_anchor[0]), float(bbox_to_anchor[1])

    if bbox_transform is None or bbox_transform is ax.transAxes:
        return anchor_y

    # Round-trip through display coordinates, since a transform need not treat
    # the two axes independently.
    display = bbox_transform.transform((anchor_x, anchor_y))
    return float(ax.transAxes.inverted().transform(display)[1])


def _title_y_above_legend(*, n_handles: int, ncol: int, anchor_y: float) -> float:
    """Return a title ``y`` that clears a legend of ``n_handles`` entries.

    The legend grows upward from ``anchor_y``, one row at a time, so a title
    pinned at a fixed height runs through it as soon as the entries wrap onto
    a second row.

    Parameters
    ----------
    n_handles : int
        Number of legend entries.
    ncol : int
        Number of legend columns.
    anchor_y : float
        The legend's ``bbox_to_anchor`` y, in axes coordinates.

    Returns
    -------
    float
        Title ``y`` in axes coordinates.
    """
    rows = max(1, math.ceil(n_handles / max(1, ncol)))
    return anchor_y + rows * _PlotDefaults.title_legend_row_height


#: Matches only warnings raised when a model input is outside its applicability
#: limits. Anchored loosely because the message opens with the parameter name
#: and the offending values, which vary.
_APPLICABILITY_WARNING = r".*outside the applicability limits"


@contextlib.contextmanager
def _suppress_applicability_warnings() -> Iterator[None]:
    """Silence the models' out-of-applicability-limits warnings.

    Every model warns, at length, when an input falls outside the limits its
    standard defines.  That is the right default for someone calling a model
    directly, but a threshold chart deliberately sweeps across those limits --
    finding where they fall is how it draws the out-of-model-limits area -- so
    the warning fires on every evaluation and says nothing the chart is not
    about to show.  One 129-point sweep of ``pmv_ppd_iso`` raises two warnings
    of about 500 characters each; a notebook full of charts drowns in them.

    Only that one message is filtered.  Models raise other warnings of the same
    category that report a calculation going wrong rather than an input being
    out of range -- ``cooling_effect`` when its solver returns zero,
    ``sports_heat_stress_risk`` when an internal solver falls back -- and a
    chart must not swallow those.

    The information is not lost either: out-of-limits areas are shaded and
    carry their own legend entry.  Call the model directly to see the warnings.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=_APPLICABILITY_WARNING, category=UserWarning
        )
        yield


def _apply_axes_style(ax: Axes) -> None:
    """Apply the package's axes styling to *ax* directly.

    ``_PYTHERMALCOMFORT_RC`` is an ``rc_context``, so its settings only reach
    axes that are *created* inside it.  An axes the caller made earlier keeps
    its own frame and grid, which leaves a multi-panel figure styled
    inconsistently -- panels drawn on caller-supplied axes get a full box,
    panels drawn on axes the plot created do not.  Setting the same properties
    on the axes closes that gap.

    Callers who want the grid back can call ``result.ax.grid(True)``.

    Parameters
    ----------
    ax : Axes
        Axis to style in place.
    """
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _limit_labeled_ticks(ax)


def _limit_labeled_ticks(ax: Axes) -> None:
    """Limit each axis to six major tick labels by default."""
    # MaxNLocator counts intervals, so five intervals give at most six labels.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=_PlotDefaults.max_labeled_ticks - 1))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=_PlotDefaults.max_labeled_ticks - 1))


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


_DEFAULT_REGION_COLORS: dict[int, tuple[str, ...]] = {
    1: ("#F1F3F2",),
    2: ("#86AEC8", "#D88B7B"),
    3: ("#86AEC8", "#F1F3F2", "#D88B7B"),
    4: ("#5F8FA9", "#B7D0DE", "#E7B7AC", "#C66B5E"),
    5: ("#5F8FA9", "#B7D0DE", "#F1F3F2", "#E7B7AC", "#C66B5E"),
    6: ("#527F98", "#86AEC8", "#C7DCE6", "#EBCBC3", "#D88B7B", "#B85F55"),
    7: (
        "#527F98",
        "#86AEC8",
        "#C7DCE6",
        "#F1F3F2",
        "#EBCBC3",
        "#D88B7B",
        "#B85F55",
    ),
}


def _default_region_colors(n_regions: int) -> list[str]:
    """Return muted default colors from cool blue to warm terracotta.

    Palettes with an odd number of regions place neutral gray in the central
    band. Even palettes have no true midpoint, so they move directly from
    pale blue to pale terracotta. Larger palettes interpolate between the
    seven defined anchors while preserving that order.
    """
    if n_regions < 1:
        raise ValueError("n_regions must be at least 1.")

    if n_regions in _DEFAULT_REGION_COLORS:
        return list(_DEFAULT_REGION_COLORS[n_regions])

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cool_neutral_warm", _DEFAULT_REGION_COLORS[7]
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
