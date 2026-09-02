"""Example 2: Beijing hourly UTCI analysis from an EPW climate file.

Parses beijing.epw, calculates the Universal Thermal Climate Index (UTCI) for
every hour using pythermalcomfort, and produces a heatmap of hourly UTCI values
and a monthly stress-category distribution.

Mean radiant temperature (tr) is set equal to dry-bulb air temperature (tdb),
representing a person sheltered from direct solar radiation.

Reproduces the illustrative example 2 figure of the "pythermalcomfort"
Building Simulation manuscript.

Usage
-----
    python3 examples/manuscript-v4/example-epw-analysis.py
    python3 examples/manuscript-v4/example-epw-analysis.py <path/to/file.epw>
"""

import csv
import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pythermalcomfort.models import utci
from pythermalcomfort.plots.matplotlib import SummaryPlot

# ---------------------------------------------------------------------------
# UTCI stress categories, thresholds (degC), and colors for plotting
# ---------------------------------------------------------------------------
UTCI_THRESHOLDS = [-40, -27, -13, 0, 9, 26, 32, 38, 46]
UTCI_LABELS = [
    "Extreme cold stress",
    "Very strong cold stress",
    "Strong cold stress",
    "Moderate cold stress",
    "Slight cold stress",
    "No thermal stress",
    "Moderate heat stress",
    "Strong heat stress",
    "Very strong heat stress",
    "Extreme heat stress",
]
UTCI_COLORS = [
    "#053061",  # extreme cold
    "#2166ac",  # very strong cold
    "#4393c3",  # strong cold
    "#92c5de",  # moderate cold
    "#74add1",  # slight cold
    "#abdda4",  # no stress
    "#fee08b",  # moderate heat
    "#f46d43",  # strong heat
    "#d73027",  # very strong heat
    "#d73027",  # extreme heat
]
MONTH_LABELS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]
LEGEND_MIN_MONTHLY_PCT = 1.0
EPW_HEADER_ROWS = 8

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EPW_PATH = os.path.join(SCRIPT_DIR, "data", "beijing.epw")


def parse_epw(path):
    """Return month, day, hour (0-based), tdb (degC), v (m/s) and rh (%) from an EPW file."""
    months, days, hours, tdb_vals, v_vals, rh_vals = [], [], [], [], [], []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for _ in range(EPW_HEADER_ROWS):
            next(reader)
        for row in reader:
            if len(row) < 22:
                continue
            try:
                months.append(int(row[1]))
                days.append(int(row[2]))
                hours.append(int(row[3]) - 1)  # EPW hours are 1-24
                tdb_vals.append(float(row[6]))
                rh_vals.append(float(row[8]))
                v_vals.append(float(row[21]))
            except (ValueError, IndexError):
                continue
    return (
        np.array(months),
        np.array(days),
        np.array(hours),
        np.array(tdb_vals),
        np.array(v_vals),
        np.array(rh_vals),
    )


def main():
    epw_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EPW_PATH
    if not os.path.isfile(epw_path):
        msg = f"EPW file not found: {epw_path}"
        raise FileNotFoundError(msg)

    print(f"Parsing EPW: {epw_path}")
    month, day, hour, tdb, v, rh = parse_epw(epw_path)

    # tr = tdb: person assumed to be sheltered from direct solar radiation
    result = utci(tdb=tdb, tr=tdb, v=v, rh=rh, limit_inputs=False)
    utci_vals = np.array(result.utci, dtype=float)
    stress_categories = np.array(result.stress_category)

    n = len(utci_vals)

    print("\nAnnual UTCI stress category distribution:")
    for label in UTCI_LABELS:
        count = int(np.sum(stress_categories == label.lower()))
        print(f"  {label:<35s}: {count:5d} h  ({100 * count / n:.1f}%)")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    n_full_days = n // 24
    n_used = n_full_days * 24
    utci_grid = utci_vals[:n_used].reshape(n_full_days, 24).T
    day_month = month[:n_used].reshape(n_full_days, 24)[:, 0]

    cmap = mcolors.ListedColormap(UTCI_COLORS)
    bounds = [-50] + UTCI_THRESHOLDS + [60]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(9, 7), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[7, 1])
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_sum = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])

    # Top left: hourly UTCI heatmap
    ax_heat.imshow(
        utci_grid,
        aspect="auto",
        cmap=cmap,
        norm=norm,
        origin="lower",
        extent=[0.5, n_full_days + 0.5, -0.5, 23.5],
    )
    ax_heat.set_xlabel("Day of year")
    ax_heat.set_ylabel("Hour of day")
    ax_heat.set_yticks(range(0, 24, 3))
    ax_heat.set_title("Hourly UTCI (shade)")

    month_ticks, month_tick_labels = [], []
    for month_id, month_label in enumerate(MONTH_LABELS, start=1):
        idx = np.where(day_month == month_id)[0]
        if idx.size > 0:
            month_ticks.append(int(idx[idx.size // 2]) + 1)
            month_tick_labels.append(month_label)
    ax_heat.set_xticks(month_ticks)
    ax_heat.set_xticklabels(month_tick_labels)
    ax_heat.grid(False)

    # Top right: annual UTCI stress category distribution
    df_utci = pd.DataFrame({"utci": utci_vals})
    (
        SummaryPlot(df_utci)
        .set_regions(
            output="utci",
            thresholds=UTCI_THRESHOLDS,
            labels=[],
            colors=UTCI_COLORS,
        )
        .plot(ax=ax_sum, vertical=True, legend=False)
    )
    ax_sum.text(
        0.8,
        0.5,
        "Annual distribution",
        ha="center",
        va="center",
        transform=ax_sum.transAxes,
        rotation=90,
        fontsize=12,
    )

    # Bottom: monthly stacked bar of UTCI stress categories
    dark_labels = {
        "extreme cold stress",
        "very strong cold stress",
        "strong cold stress",
        "strong heat stress",
        "very strong heat stress",
        "extreme heat stress",
    }
    monthly_counts = np.zeros((12, len(UTCI_LABELS)), dtype=float)
    for m in range(1, 13):
        m_mask = month == m
        for i, label in enumerate(UTCI_LABELS):
            monthly_counts[m - 1, i] = np.sum(
                m_mask & (stress_categories == label.lower())
            )

    month_totals = monthly_counts.sum(axis=1, keepdims=True)
    monthly_pct = np.divide(
        monthly_counts * 100.0,
        month_totals,
        out=np.zeros_like(monthly_counts),
        where=month_totals > 0,
    )

    x = np.arange(12)
    bottom = np.zeros(12)
    legend_handles, legend_labels_plot = [], []
    for i, (label, color) in enumerate(zip(UTCI_LABELS, UTCI_COLORS, strict=True)):
        vals = monthly_pct[:, i]
        bars = ax_bar.bar(
            x,
            vals,
            bottom=bottom,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            label=label,
        )
        for j, bar in enumerate(bars):
            if vals[j] >= 5.0:
                ax_bar.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bottom[j] + vals[j] / 2.0,
                    f"{vals[j]:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if label.lower() in dark_labels else "black",
                )
        if np.any(vals >= LEGEND_MIN_MONTHLY_PCT):
            legend_handles.append(bars[0])
            legend_labels_plot.append(label)
        bottom += vals

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(MONTH_LABELS)
    ax_bar.set_ylim(0, 100)
    ax_bar.set_ylabel("Percentage of time (%)")
    ax_bar.set_title("Monthly UTCI (shade)")
    ax_bar.grid(False)
    ax_bar.legend(
        legend_handles,
        legend_labels_plot,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=min(4, max(1, len(legend_labels_plot))),
        fontsize=9,
        frameon=False,
    )

    outdir = os.path.join(SCRIPT_DIR, "output")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "example2_epw_utci.pdf")
    fig.savefig(out)
    plt.show()
    print(f"\nFigure saved to {out}")


if __name__ == "__main__":
    main()
