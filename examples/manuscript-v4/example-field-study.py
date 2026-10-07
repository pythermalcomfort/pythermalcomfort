"""Illustrative fixed-air-speed PMV calculation from Cozie wearable data.

Reads a Cozie field-study dataset and computes PMV (ISO 7730) for each retained
observation at an assumed air speed of 0.1 m/s. The resulting per-participant
distribution illustrates PMV categories; it is not a comfort-validation analysis.

Usage
-----
    python3 examples/manuscript-v4/example-field-study.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from pythermalcomfort.models import pmv_ppd_iso

# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------
MET_MAP = {"sitting": 1.1, "resting": 0.8, "standing": 1.4, "exercising": 3.0}
CLO_MAP = {"Very light": 0.3, "Light": 0.5, "Medium": 0.7, "Heavy": 1.0}

AIR_SPEED = 0.1  # m/s, fixed illustrative assumption

# PMV numerical categories for the fixed-air-speed calculation.
PMV_LABELS = [
    "PMV < -0.5",
    "-0.5 ≤ PMV ≤ 0.5",
    "PMV > 0.5",
]
PMV_COLORS = ["#74add1", "#abdda4", "#f46d43"]

# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

df = pd.read_csv(os.path.join(DATA_DIR, "df_cozie_env.csv"))

df["met"] = df["met"].map(MET_MAP)
df["clothing"] = df["clothing"].map(CLO_MAP)

# Drop rows missing required inputs
df_pmv = df.dropna(subset=["t-env", "rh-env", "met", "clothing"]).copy()

# ---------------------------------------------------------------------------
# PMV (vectorised) at the fixed illustrative air speed of 0.1 m/s.
# This calculation illustrates model categories and does not validate comfort
# against participant-reported outcomes.
# ---------------------------------------------------------------------------
_common = dict(
    tdb=df_pmv["t-env"].values,
    tr=df_pmv["t-env"].values,
    rh=df_pmv["rh-env"].values,
    met=df_pmv["met"].values,
    clo=df_pmv["clothing"].values,
    model="7730-2005",
    limit_inputs=False,
)
df_pmv["pmv"] = np.array(pmv_ppd_iso(vr=AIR_SPEED, **_common).pmv, dtype=float)
df_pmv["pmv_cat"] = np.select(
    [df_pmv["pmv"] < -0.5, df_pmv["pmv"] <= 0.5],
    [PMV_LABELS[0], PMV_LABELS[1]],
    default=PMV_LABELS[2],
)
df_pmv["pmv_cat"] = pd.Categorical(df_pmv["pmv_cat"], categories=PMV_LABELS)

print(f"Rows with valid data: {len(df_pmv)}")
print(f"\nPMV category distribution:\n{df_pmv['pmv_cat'].value_counts()}")

# ---------------------------------------------------------------------------
# Per-participant percentage table
# ---------------------------------------------------------------------------
PARTICIPANTS = sorted(df_pmv["userid"].unique())


def participant_pct(data, col, categories):
    """Return (n_participants x n_categories) percentage array."""
    pct = np.zeros((len(PARTICIPANTS), len(categories)))
    for i, pid in enumerate(PARTICIPANTS):
        sub = data[data["userid"] == pid]
        totals = len(sub)
        if totals > 0:
            for j, cat in enumerate(categories):
                pct[i, j] = (sub[col] == cat).sum() / totals * 100
    return pct


pct_pmv = participant_pct(df_pmv, "pmv_cat", PMV_LABELS)

# ---------------------------------------------------------------------------
# Figure: single stacked bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 3.2))

x = np.arange(len(PARTICIPANTS))
x_labels = [f"P{p}" for p in PARTICIPANTS]

bottom = np.zeros(len(PARTICIPANTS))
for j, (label, color) in enumerate(zip(PMV_LABELS, PMV_COLORS, strict=True)):
    vals = pct_pmv[:, j]
    bars = ax.bar(
        x,
        vals,
        bottom=bottom,
        color=color,
        edgecolor="white",
        linewidth=0.4,
        label=label,
    )
    for k, bar in enumerate(bars):
        if vals[k] >= 8.0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bottom[k] + vals[k] / 2.0,
                f"{vals[k]:.0f}%",
                ha="center",
                va="center",
                fontsize=7,
                color="k",
            )
    bottom += vals

ax.set_ylim(0, 100)
ax.set_ylabel("Observations (%)")
ax.grid(False)
legend_handles = [
    Patch(facecolor=c, label=lbl) for c, lbl in zip(PMV_COLORS, PMV_LABELS, strict=True)
]
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Title and legend are placed in figure (not axes) coordinates and reserved
# their own margin above the axes, so they never overlap a full-height bar
# regardless of the data (axes-fraction placement like `ax.legend(...,
# bbox_to_anchor=(0.5, 1.10))` sits a fixed 10% of the *axes* height above
# the top of the bars, which is smaller than the legend's own rendered
# height and so overlaps it).
fig.suptitle(
    "PMV categories (ISO 7730; assumed air speed 0.1 m/s)",
    fontsize=10,
    y=0.94,
)
fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.90),
    fontsize=8,
    frameon=False,
    ncol=3,
)

fig.tight_layout(rect=(0, 0, 1, 0.95))

outdir = os.path.join(SCRIPT_DIR, "output")
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, "example_field_study.pdf")
fig.savefig(out)
print(f"\nFigure saved to {out}")
