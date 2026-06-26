"""Plot Croquet and Bowls sports heat stress risk comparisons."""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

OUTPUT_DIR = Path(__file__).resolve().parent / "sports_heat_stress_comparison"
PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "pythermalcomfort-matplotlib"),
)


def _configure_matplotlib_backend() -> None:
    """Configure Matplotlib for headless image generation."""
    import matplotlib

    matplotlib.use("Agg")


_configure_matplotlib_backend()


def _cell_centers(edges: np.ndarray) -> np.ndarray:
    """Return cell centers for a monotonic array of cell edges."""
    return (edges[:-1] + edges[1:]) / 2.0


TA_EDGES = np.arange(20.0, 45.0 + 1.0, 1.0)
RH_EDGES = np.arange(0.0, 100.0 + 5.0, 5.0)
TA_VALUES = _cell_centers(TA_EDGES)
RH_VALUES = _cell_centers(RH_EDGES)

RISK_LEVELS = [1.0, 2.0, 3.0, 4.0, 4.9]
RISK_COLORS = ["#1a9850", "#fee08b", "#f46d43", "#d73027"]
RISK_LABELS = ["Low (1-2)", "Moderate (2-3)", "High (3-4)", "Extreme (4-4.9)"]
RISK_TICKS = [1.5, 2.5, 3.5, 4.45]

DIFF_LIMIT = 1.0
DIFF_TICKS = [-1.0, -0.5, 0.0, 0.5, 1.0]

SCENARIOS = [
    {
        "title": "ta = tr, vr = 0.5 m/s",
        "filename": "croquet_bowls_ta_eq_tr_vr_0p5.png",
        "tr_offset": 0.0,
        "vr": 0.5,
    },
    {
        "title": "ta = tr, vr = 2.0 m/s",
        "filename": "croquet_bowls_ta_eq_tr_vr_2p0.png",
        "tr_offset": 0.0,
        "vr": 2.0,
    },
    {
        "title": "tr = ta + 10 C, vr = 0.5 m/s",
        "filename": "croquet_bowls_tr_ta_plus_10_vr_0p5.png",
        "tr_offset": 10.0,
        "vr": 0.5,
    },
    {
        "title": "tr = ta + 10 C, vr = 2.0 m/s",
        "filename": "croquet_bowls_tr_ta_plus_10_vr_2p0.png",
        "tr_offset": 10.0,
        "vr": 2.0,
    },
]


def _risk_grid(
    *,
    ta_grid: np.ndarray,
    tr_grid: np.ndarray,
    rh_grid: np.ndarray,
    vr: float,
    sport: object,
) -> np.ndarray:
    """Return interpolated sports heat stress risk for one sport."""
    from pythermalcomfort.models.sports_heat_stress_risk import sports_heat_stress_risk

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Solver did not find a solution.*",
            category=UserWarning,
        )
        result = sports_heat_stress_risk(
            tdb=ta_grid,
            tr=tr_grid,
            rh=rh_grid,
            vr=vr,
            sport=sport,
        )
    return np.asarray(result.risk_level_interpolated, dtype=float)


def _plot_risk_panel(
    ax: Any,
    *,
    ta_edges: np.ndarray,
    rh_edges: np.ndarray,
    risk: np.ndarray,
    title: str,
) -> Any:
    """Plot a sport risk panel and return the color mesh."""
    from matplotlib.colors import BoundaryNorm, ListedColormap

    risk_cmap = ListedColormap(RISK_COLORS)
    risk_norm = BoundaryNorm(RISK_LEVELS, risk_cmap.N, clip=True)
    mesh = ax.pcolormesh(
        ta_edges,
        rh_edges,
        risk,
        cmap=risk_cmap,
        norm=risk_norm,
        shading="flat",
    )
    ax.set_title(title)
    ax.set_xlabel("Air temperature, ta [C]")
    ax.set_ylabel("Relative humidity [%]")
    return mesh


def _plot_diff_panel(
    ax: Any,
    *,
    ta_edges: np.ndarray,
    rh_edges: np.ndarray,
    diff: np.ndarray,
) -> Any:
    """Plot Bowls minus Croquet risk difference as color blocks."""
    mesh = ax.pcolormesh(
        ta_edges,
        rh_edges,
        diff,
        cmap="coolwarm",
        vmin=-DIFF_LIMIT,
        vmax=DIFF_LIMIT,
        shading="flat",
    )
    ax.set_title("Bowls - Croquet")
    ax.set_xlabel("Air temperature, ta [C]")
    ax.set_ylabel("Relative humidity [%]")
    return mesh


def plot_scenario(
    *,
    title: str,
    filename: str,
    tr_offset: float,
    vr: float,
) -> Path:
    """Generate one three-panel comparison figure and return its path."""
    import matplotlib.pyplot as plt

    from pythermalcomfort.models.sports_heat_stress_risk import Sports

    ta_grid, rh_grid = np.meshgrid(TA_VALUES, RH_VALUES)
    tr_grid = ta_grid + tr_offset

    croquet_risk = _risk_grid(
        ta_grid=ta_grid,
        tr_grid=tr_grid,
        rh_grid=rh_grid,
        vr=vr,
        sport=Sports.CROQUET,
    )
    bowls_risk = _risk_grid(
        ta_grid=ta_grid,
        tr_grid=tr_grid,
        rh_grid=rh_grid,
        vr=vr,
        sport=Sports.BOWLS,
    )
    diff = bowls_risk - croquet_risk

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13, 4),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    fig.suptitle(title)

    risk_mesh = _plot_risk_panel(
        axes[0],
        ta_edges=TA_EDGES,
        rh_edges=RH_EDGES,
        risk=croquet_risk,
        title="Croquet",
    )
    _plot_risk_panel(
        axes[1],
        ta_edges=TA_EDGES,
        rh_edges=RH_EDGES,
        risk=bowls_risk,
        title="Bowls",
    )
    diff_mesh = _plot_diff_panel(
        axes[2],
        ta_edges=TA_EDGES,
        rh_edges=RH_EDGES,
        diff=diff,
    )

    risk_cbar = fig.colorbar(
        risk_mesh,
        ax=axes[:2],
        ticks=RISK_TICKS,
        shrink=0.9,
    )
    risk_cbar.ax.set_yticklabels(RISK_LABELS)
    risk_cbar.set_label("Risk level")

    diff_cbar = fig.colorbar(
        diff_mesh,
        ax=axes[2],
        ticks=DIFF_TICKS,
        shrink=0.9,
    )
    diff_cbar.set_label("Risk level difference")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> None:
    """Generate all Croquet versus Bowls comparison figures."""
    for scenario in SCENARIOS:
        output_path = plot_scenario(**scenario)
        print(output_path)


if __name__ == "__main__":
    main()
