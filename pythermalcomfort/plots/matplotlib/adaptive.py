"""Adaptive comfort chart plotting for ASHRAE 55 and EN 16798.

Comfort band boundaries are computed as smooth lines directly from the
standard equations.  The slope, intercept, and cooling-effect function are
imported from the underlying model modules so there is one single source of
truth for every numeric constant.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import is_color_like
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from pythermalcomfort.models.adaptive_ashrae import INTERCEPT as _ASHRAE_INTERCEPT
from pythermalcomfort.models.adaptive_ashrae import SLOPE as _ASHRAE_SLOPE
from pythermalcomfort.models.adaptive_en import INTERCEPT as _EN_INTERCEPT
from pythermalcomfort.models.adaptive_en import SLOPE as _EN_SLOPE
from pythermalcomfort.plots.matplotlib._base import BasePlot
from pythermalcomfort.plots.matplotlib._shared import (
    _PYTHERMALCOMFORT_RC,
    BasePlotResult,
    _PlotDefaults,
    _resolve_layout,
)
from pythermalcomfort.utilities import adaptive_cooling_effect

# ── band specification ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class _BandSpec:
    """Static specification of one comfort band."""

    key: str
    lower_offset: float  # below t_cmf (negative)
    upper_offset: float  # above t_cmf (positive, before cooling effect)
    default_label: str
    default_color: str


_STANDARD_CONFIGS: dict[str, dict[str, Any]] = {
    "ashrae": {
        "slope": _ASHRAE_SLOPE,
        "intercept": _ASHRAE_INTERCEPT,
        "t_rm_range": (10.0, 33.5),
        "bands": [
            _BandSpec("80", -3.5, 3.5, "80% Acceptability", "#B3D9FF"),
            _BandSpec("90", -2.5, 2.5, "90% Acceptability", "#6BB3FF"),
        ],
    },
    "en": {
        "slope": _EN_SLOPE,
        "intercept": _EN_INTERCEPT,
        "t_rm_range": (10.0, 33.5),
        "bands": [
            _BandSpec("cat_iii", -5.0, 4.0, "Category III", "#C5E0B4"),
            _BandSpec("cat_ii", -4.0, 3.0, "Category II", "#A9D18E"),
            _BandSpec("cat_i", -3.0, 2.0, "Category I", "#70AD47"),
        ],
    },
}

_MODEL_TO_STANDARD: dict[str, str] = {
    "adaptive_ashrae": "ashrae",
    "adaptive_en": "en",
}

_BAND_KEYS: dict[str, list[str]] = {
    std: [b.key for b in cfg["bands"]] for std, cfg in _STANDARD_CONFIGS.items()
}


# ── public config ──────────────────────────────────────────────────────────


@dataclass
class RegionsConfig:
    """Reusable comfort band configuration for adaptive charts.

    Controls which bands are displayed and their appearance.  Create a
    ``RegionsConfig`` once and pass it to :meth:`AdaptivePlot.set_regions`
    to guarantee consistent band definitions across multiple plots.

    Attributes
    ----------
    show : sequence of str or None
        Band keys to display.  If ``None``, all bands are shown.

        - ASHRAE keys: ``"80"``, ``"90"``
        - EN keys: ``"cat_i"``, ``"cat_ii"``, ``"cat_iii"``

    labels : sequence of str or None
        Custom labels for the visible bands.  Must have the same length as
        *show* (or the total number of bands if *show* is ``None``).  If
        ``None``, default labels are used.
    colors : sequence of str or None
        Custom colors for the visible bands.  Same length rule as *labels*.
        If ``None``, default colors are used.

    Examples
    --------
    .. code-block:: python

        config = RegionsConfig(
            show=["90"],
            labels=["90% Comfort Zone"],
            colors=["#FF6B6B"],
        )
    """

    show: Sequence[str] | None = None
    labels: Sequence[str] | None = None
    colors: Sequence[str] | None = None

    def _validate(self, standard: str) -> None:
        """Validate against a specific standard's band keys."""
        valid_keys = set(_BAND_KEYS[standard])
        n_visible = (
            len(self.show) if self.show is not None else len(_BAND_KEYS[standard])
        )

        if self.show is not None:
            invalid = [k for k in self.show if k not in valid_keys]
            if invalid:
                msg = (
                    f"Invalid band key(s): {', '.join(invalid)}. "
                    f"Valid keys for '{standard}': {', '.join(sorted(valid_keys))}"
                )
                raise ValueError(msg)

        if self.labels is not None:
            if len(self.labels) != n_visible:
                msg = f"labels must have length {n_visible} (got {len(self.labels)})."
                raise ValueError(msg)

        if self.colors is not None:
            if len(self.colors) != n_visible:
                msg = f"colors must have length {n_visible} (got {len(self.colors)})."
                raise ValueError(msg)
            bad = [c for c in self.colors if not is_color_like(c)]
            if bad:
                msg = f"Invalid color value(s): {', '.join(str(c) for c in bad)}"
                raise ValueError(msg)


# ── resolved band (internal) ──────────────────────────────────────────────


@dataclass(frozen=True)
class _ResolvedBand:
    """A band with all overrides applied, ready to render."""

    spec: _BandSpec
    label: str
    color: str


# ── result container ───────────────────────────────────────────────────────


@dataclass
class AdaptivePlotResult(BasePlotResult):
    """Result from :meth:`AdaptivePlot.plot`.

    Attributes
    ----------
    fig : Figure
        Matplotlib figure.
    ax : Axes
        Matplotlib axes.
    center_line : Line2D or None
        The comfort temperature center line artist, or ``None``.
    fills : list of PolyCollection
        Filled comfort band artists (outermost first).
    legend : Legend or None
        Legend artist, or ``None``.
    """

    center_line: Line2D | None
    fills: list[PolyCollection]
    legend: Legend | None


# ── main class ─────────────────────────────────────────────────────────────


class AdaptivePlot(BasePlot):
    """Adaptive comfort chart for ASHRAE 55 or EN 16798.

    The chart displays comfort bands as filled regions on a plot of
    operative temperature (y-axis) versus prevailing mean outdoor
    temperature (x-axis).  Band boundaries are smooth lines computed
    directly from the standard equations; all numeric constants are
    imported from the underlying model modules.

    The cooling effect (if any) shifts the **upper** boundary of each band
    upward, but only where that boundary already exceeds 25 °C — matching
    the standard definition.

    Band keys for selection and customization:

    - **ASHRAE**: ``"80"`` (80% acceptability), ``"90"`` (90% acceptability)
    - **EN**: ``"cat_i"`` (Category I), ``"cat_ii"`` (Category II),
      ``"cat_iii"`` (Category III)

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import adaptive_ashrae
        from pythermalcomfort.plots.matplotlib import AdaptivePlot

        result = (
            AdaptivePlot(adaptive_ashrae)
            .set_params(v=0.5)
            .plot(title="Adaptive Comfort (ASHRAE 55)")
        )
    """

    def __init__(self, model_func: Any) -> None:
        """Initialize an adaptive comfort chart builder.

        Parameters
        ----------
        model_func : callable
            The adaptive comfort model function.  Must be ``adaptive_ashrae``
            or ``adaptive_en`` from :mod:`pythermalcomfort.models`.

        Raises
        ------
        ValueError
            If *model_func* is not a recognized adaptive model function.
        """
        super().__init__()
        name = getattr(model_func, "__name__", "")
        if name not in _MODEL_TO_STANDARD:
            valid = ", ".join(sorted(_MODEL_TO_STANDARD))
            msg = (
                f"model_func must be one of the adaptive model functions "
                f"({valid}), got '{name}'."
            )
            raise ValueError(msg)

        self._standard = _MODEL_TO_STANDARD[name]
        self._cfg = _STANDARD_CONFIGS[self._standard]
        self._v: float = 0.1
        self._regions_config: RegionsConfig | None = None
        self._t_rm_range: tuple[float, float] = self._cfg["t_rm_range"]
        self._y_range: tuple[float, float] | None = None

    def set_x_axis(self, min_val: float, max_val: float) -> AdaptivePlot:
        """Set the x-axis (prevailing mean outdoor temperature) display range.

        Parameters
        ----------
        min_val : float
            Minimum prevailing mean outdoor temperature [°C].
        max_val : float
            Maximum prevailing mean outdoor temperature [°C].

        Returns
        -------
        AdaptivePlot
            Self, to support method chaining.

        Raises
        ------
        ValueError
            If *min_val* >= *max_val* or values are non-numeric.
        """
        try:
            lo, hi = float(min_val), float(max_val)
        except (TypeError, ValueError) as exc:
            raise ValueError("set_x_axis values must be numeric.") from exc
        if lo >= hi:
            msg = f"set_x_axis requires min < max (got {lo} >= {hi})."
            raise ValueError(msg)
        self._t_rm_range = (lo, hi)
        return self

    def set_y_axis(self, min_val: float, max_val: float) -> AdaptivePlot:
        """Set the y-axis (operative temperature) display range.

        Parameters
        ----------
        min_val : float
            Minimum operative temperature [°C].
        max_val : float
            Maximum operative temperature [°C].

        Returns
        -------
        AdaptivePlot
            Self, to support method chaining.

        Raises
        ------
        ValueError
            If *min_val* >= *max_val* or values are non-numeric.
        """
        try:
            lo, hi = float(min_val), float(max_val)
        except (TypeError, ValueError) as exc:
            raise ValueError("set_y_axis values must be numeric.") from exc
        if lo >= hi:
            msg = f"set_y_axis requires min < max (got {lo} >= {hi})."
            raise ValueError(msg)
        self._y_range = (lo, hi)
        return self

    def set_params(self, *, v: float) -> AdaptivePlot:
        """Set the air speed used to compute the cooling effect.

        Parameters
        ----------
        v : float
            Air speed in m/s.  Values below 0.6 m/s produce no cooling effect.

        Returns
        -------
        AdaptivePlot
            Self, to support method chaining.
        """
        self._v = float(v)
        return self

    def set_regions(
        self,
        *,
        show: RegionsConfig | Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
        colors: Sequence[str] | None = None,
    ) -> AdaptivePlot:
        """Configure which comfort bands are displayed and their appearance.

        Accepts either a pre-built :class:`RegionsConfig` or raw parameters.
        ``output`` and numeric ``thresholds`` do not apply here — bands are
        identified by key (e.g. ``"80"`` or ``"cat_i"``) and their boundaries
        are defined by the standard.

        Parameters
        ----------
        show : RegionsConfig, sequence of str, or None
            Controls which bands are shown and, optionally, their appearance.

            - **RegionsConfig** — a fully configured instance; *labels* and
              *colors* must not be supplied separately.
            - **list of band keys** — selects which bands to display; use
              *labels* / *colors* for appearance.
            - **None** — all bands are shown (default).

            ASHRAE keys: ``"80"``, ``"90"``.
            EN keys: ``"cat_i"``, ``"cat_ii"``, ``"cat_iii"``.

        labels : sequence of str, optional
            Custom labels for the visible bands.  Must have the same length as
            *show* (or the total band count if *show* is ``None``).
        colors : sequence of str, optional
            Custom colors for the visible bands.  Same length rule as *labels*.

        Returns
        -------
        AdaptivePlot
            Self, to support method chaining.

        Raises
        ------
        ValueError
            If band keys are invalid, or if *labels*/*colors* are supplied
            alongside a :class:`RegionsConfig` instance.

        Examples
        --------
        .. code-block:: python

            # Raw parameters
            .set_regions(show=["90"], labels=["90% Zone"], colors=["#FF6B6B"])

            # RegionsConfig (reusable across plots)
            config = RegionsConfig(show=["90"], labels=["90% Zone"], colors=["#FF6B6B"])
            .set_regions(show=config)
        """
        if isinstance(show, RegionsConfig):
            if labels is not None or colors is not None:
                raise ValueError(
                    "labels and colors must not be provided separately when "
                    "show is a RegionsConfig instance.  Set them inside the "
                    "RegionsConfig instead."
                )
            config = show
        else:
            config = RegionsConfig(show=show, labels=labels, colors=colors)

        config._validate(self._standard)
        self._regions_config = config
        return self

    def _resolve_bands(self) -> list[_ResolvedBand]:
        """Return bands with all user overrides applied."""
        all_specs: list[_BandSpec] = self._cfg["bands"]
        cfg = self._regions_config

        if cfg is not None and cfg.show is not None:
            show_set = set(cfg.show)
            visible_specs = [s for s in all_specs if s.key in show_set]
        else:
            visible_specs = list(all_specs)

        resolved: list[_ResolvedBand] = []
        for i, spec in enumerate(visible_specs):
            label = spec.default_label
            color = spec.default_color
            if cfg is not None:
                if cfg.labels is not None:
                    label = str(cfg.labels[i])
                if cfg.colors is not None:
                    color = str(cfg.colors[i])
            resolved.append(_ResolvedBand(spec=spec, label=label, color=color))
        return resolved

    def plot(
        self,
        *,
        ax: Axes | None = None,
        title: str | None = None,
        xlabel: str | None = "Prevailing Mean Outdoor Temperature [°C]",
        ylabel: str | None = "Operative Temperature [°C]",
        legend: bool = True,
        grid: bool = True,
        show_center_line: bool = True,
        center_line_kws: Mapping[str, Any] | None = None,
        fill_kws: Mapping[str, Any] | None = None,
        legend_kws: Mapping[str, Any] | None = None,
    ) -> AdaptivePlotResult:
        """Render the adaptive comfort chart.

        Parameters
        ----------
        ax : Axes, optional
            Existing axes.  If ``None``, a new figure is created with a
            default size of ``(7, 4)`` inches.
        title : str, optional
            Optional chart title.
        xlabel : str or None
            X-axis label.  ``None`` to omit.
        ylabel : str or None
            Y-axis label.  ``None`` to omit.
        legend : bool
            Whether to draw a legend.
        grid : bool
            Whether to display background grid lines.
        show_center_line : bool
            Whether to draw the comfort temperature center line.
        center_line_kws : dict, optional
            Overrides for the center line (``ax.plot``).
        fill_kws : dict, optional
            Shared overrides for all bands (``ax.fill_between``).  Per-band
            colors are set via :meth:`set_regions`.
        legend_kws : dict, optional
            Overrides for the legend (``ax.legend``).

        Returns
        -------
        AdaptivePlotResult
            Result with figure, axes, and artists.
        """
        with mpl.rc_context(_PYTHERMALCOMFORT_RC):
            bands = self._resolve_bands()

            slope: float = self._cfg["slope"]
            intercept: float = self._cfg["intercept"]

            if ax is None:
                fig, ax = plt.subplots(figsize=_PlotDefaults.figsize)
            else:
                fig = ax.figure

            fill_opts = dict(fill_kws or {})
            fill_opts.setdefault("alpha", _PlotDefaults.fill_alpha)

            slope: float = self._cfg["slope"]
            intercept: float = self._cfg["intercept"]
            t_lo, t_hi = self._t_rm_range

            ce = adaptive_cooling_effect(self._v, np.array([26.0]))[0]

            fills: list[PolyCollection] = []
            for band in bands:
                t_rm_transition = (25.0 - intercept - band.spec.upper_offset) / slope

                has_transition = ce > 0.0 and t_lo < t_rm_transition < t_hi

                if has_transition:
                    # 4 points: start, transition(no ce), transition(with ce), end
                    x = np.array([t_lo, t_rm_transition, t_rm_transition, t_hi])
                else:
                    # 2 points: start, end
                    x = np.array([t_lo, t_hi])

                t_cmf_at_x = slope * x + intercept
                lower = t_cmf_at_x + band.spec.lower_offset
                upper_base = t_cmf_at_x + band.spec.upper_offset

                if has_transition:
                    # [no ce, no ce, with ce, with ce]
                    upper = np.array(
                        [
                            upper_base[0],
                            25.0,
                            25.0 + ce,
                            upper_base[3] + ce,
                        ]
                    )
                else:
                    upper = upper_base + adaptive_cooling_effect(self._v, upper_base)

                fill = ax.fill_between(x, lower, upper, color=band.color, **fill_opts)
                fills.append(fill)

            center_line_artist: Line2D | None = None
            if show_center_line:
                cl_opts = dict(_PlotDefaults.Adaptive.center_line_defaults)
                if center_line_kws:
                    cl_opts.update(center_line_kws)
                t_lo, t_hi = self._t_rm_range
                x = [t_lo, t_hi]
                y = [slope * t_lo + intercept, slope * t_hi + intercept]
                (center_line_artist,) = ax.plot(x, y, **cl_opts)

            legend_artist: Legend | None = None
            if legend:
                handles: list[Any] = []
                for band in reversed(bands):
                    handles.append(
                        Patch(
                            facecolor=band.color,
                            alpha=fill_opts.get("alpha", _PlotDefaults.fill_alpha),
                            label=band.label,
                        )
                    )
                if center_line_artist is not None:
                    handles.append(
                        Line2D(
                            [0],
                            [0],
                            label=_PlotDefaults.Adaptive.center_line_label,
                            **dict(_PlotDefaults.Adaptive.center_line_defaults),
                        )
                    )

                labels_text = [h.get_label() for h in handles]
                dynamic_ncol, dynamic_anchor, dynamic_title_pad = _resolve_layout(
                    title=title, 
                    labels=labels_text, 
                    default_ncol=_PlotDefaults.Adaptive.legend_ncol
                )

                lg_opts = dict(legend_kws or {})
                if title is not None:
                    lg_opts.setdefault("loc", "lower center")
                    lg_opts.setdefault("bbox_to_anchor", (0.5, dynamic_anchor))
                else:
                    lg_opts.setdefault("loc", _PlotDefaults.Adaptive.legend_loc)

                lg_opts.setdefault("ncol", dynamic_ncol)
                legend_artist = ax.legend(handles=handles, **lg_opts)
            else:
                _, _, dynamic_title_pad = _resolve_layout(title, [], 1)

            if grid:
                ax.grid(True)

            if xlabel is not None:
                ax.set_xlabel(xlabel)
            if ylabel is not None:
                ax.set_ylabel(ylabel)
            if title is not None:
                ax.set_title(title, pad=dynamic_title_pad)

            ax.set_xlim(self._t_rm_range)
            if self._y_range is not None:
                ax.set_ylim(self._y_range)

            return AdaptivePlotResult(
                fig=fig,
                ax=ax,
                center_line=center_line_artist,
                fills=fills,
                legend=legend_artist,
            )