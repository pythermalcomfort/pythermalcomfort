"""Field study example: PMV, adaptive comfort, and thermal preference vote analysis from Cozie wearable data.

Reads a Cozie field-study dataset, computes PMV (ISO 7730) and the ASHRAE 55 adaptive
comfort classification for each observation, and produces a three-panel figure showing
per-participant distributions of adaptive comfort category, PMV comfort category, and
self-reported thermal preference vote.

Reproduces the illustrative example 3 figure of the "pythermalcomfort"
Building Simulation manuscript.

Usage
-----
    python3 examples/manuscript-v4/example-field-study.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from pythermalcomfort.models import adaptive_ashrae, pmv_ppd_iso

# ---------------------------------------------------------------------------
# Mappings
# ---------------------------------------------------------------------------
MET_MAP = {"sitting": 1.1, "resting": 0.8, "standing": 1.4, "exercising": 3.0}
CLO_MAP = {"Very light": 0.3, "Light": 0.5, "Medium": 0.7, "Heavy": 1.0}

# Adaptive comfort fixed bounds (operative temperature, degC)
AIR_SPEED_MIN = 0.1  # m/s
AIR_SPEED_MAX = 1  # m/s

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

df_singapore = pd.read_csv(os.path.join(DATA_DIR, "df_clima_singapore.csv"))
df_singapore.adaptive_cmf_rmt.describe()
RUNNING_MEAN = 27.4  # degC (average value from CBE Clima dataframe Paya Lebar)

ADAPTIVE_LOW = adaptive_ashrae(
    tdb=25, tr=25, t_running_mean=RUNNING_MEAN, v=AIR_SPEED_MIN
).tmp_cmf_90_low
ADAPTIVE_HIGH = adaptive_ashrae(
    tdb=25, tr=25, t_running_mean=RUNNING_MEAN, v=AIR_SPEED_MAX
).tmp_cmf_90_up
ADAPTIVE_LABELS = [
    "Cool (< 90% comfort region lower band)",
    "Comfortable",
    "Warm (> 90% comfort region upper band)",
]
ADAPTIVE_COLORS = ["#74add1", "#abdda4", "#f46d43"]

# PMV comfort categories
PMV_LABELS = ["Cool (PMV < -0.5)", "Comfortable", "Warm (PMV > 0.5)"]
PMV_COLORS = ["#74add1", "#abdda4", "#f46d43"]

# Thermal sensation vote categories
TV_LABELS = ["Warmer", "No Change", "Cooler"]
TV_COLORS = ["#74add1", "#abdda4", "#f46d43"]


# ---------------------------------------------------------------------------
# Load and prepare data
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(DATA_DIR, "df_cozie_env.csv"))

df["met"] = df["met"].map(MET_MAP)
df["clothing"] = df["clothing"].map(CLO_MAP)

# Drop rows missing required inputs
df_pmv = df.dropna(subset=["t-env", "rh-env", "met", "clothing"]).copy()

# ---------------------------------------------------------------------------
# PMV (vectorised) -- range-based classification
# Too cool : PMV < -0.5 at vr = 0.1 m/s
# Too warm : PMV >  0.5 at vr = 1.0 m/s (elevated air movement still too warm)
# Comfortable: everything in between
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
pmv_low = np.array(pmv_ppd_iso(vr=AIR_SPEED_MIN, **_common).pmv, dtype=float)
pmv_high = np.array(pmv_ppd_iso(vr=AIR_SPEED_MAX, **_common).pmv, dtype=float)

df_pmv["pmv"] = pmv_low  # reference column (vr = 0.1)
df_pmv["pmv_cat"] = np.where(
    pmv_low < -0.5,
    PMV_LABELS[0],
    np.where(pmv_high > 0.5, PMV_LABELS[2], PMV_LABELS[1]),
)
df_pmv["pmv_cat"] = pd.Categorical(df_pmv["pmv_cat"], categories=PMV_LABELS)

# ---------------------------------------------------------------------------
# Adaptive comfort (fixed bounds on operative temperature)
# ---------------------------------------------------------------------------
df_pmv["adaptive_cat"] = pd.Categorical(
    np.where(
        df_pmv["t-env"] < ADAPTIVE_LOW,
        ADAPTIVE_LABELS[0],
        np.where(
            df_pmv["t-env"] > ADAPTIVE_HIGH, ADAPTIVE_LABELS[2], ADAPTIVE_LABELS[1]
        ),
    ),
    categories=ADAPTIVE_LABELS,
)

print(f"Rows with valid data: {len(df_pmv)}")
print(f"\nAdaptive comfort distribution:\n{df_pmv['adaptive_cat'].value_counts()}")
print(f"\nPMV category distribution:\n{df_pmv['pmv_cat'].value_counts()}")

# ---------------------------------------------------------------------------
# Thermal vote (keep string labels; filter out blank rows)
# ---------------------------------------------------------------------------
df_tv = df[df["thermal"].isin(TV_LABELS)].copy()
df_tv["thermal"] = pd.Categorical(df_tv["thermal"], categories=TV_LABELS)

# ---------------------------------------------------------------------------
# Per-participant percentage tables
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


pct_adaptive = participant_pct(df_pmv, "adaptive_cat", ADAPTIVE_LABELS)
pct_pmv = participant_pct(df_pmv, "pmv_cat", PMV_LABELS)
pct_tv = participant_pct(df_tv, "thermal", TV_LABELS)

# ---------------------------------------------------------------------------
# Figure: three stacked bar charts
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

x = np.arange(len(PARTICIPANTS))
x_labels = [f"P{p}" for p in PARTICIPANTS]


def draw_stacked_bar(ax, pct, labels, colors, ylabel, title):
    bottom = np.zeros(len(PARTICIPANTS))
    for j, (label, color) in enumerate(zip(labels, colors, strict=True)):
        vals = pct[:, j]
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
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, y=1.1)
    ax.grid(False)
    legend_handles = [
        Patch(facecolor=color, label=label)
        for color, label in zip(colors, labels, strict=True)
    ]
    ax.legend(
        handles=legend_handles,
        loc="center",
        bbox_to_anchor=(0.5, 1.05),
        fontsize=8,
        frameon=False,
        ncol=3,
    )


draw_stacked_bar(
    axes[0],
    pct_adaptive,
    ADAPTIVE_LABELS,
    ADAPTIVE_COLORS,
    "Time (%)",
    f"Adaptive (ASHRAE 55) - air speed {AIR_SPEED_MIN} to {AIR_SPEED_MAX} m/s - "
    f"t_pma(out) = {RUNNING_MEAN} degC",
)
draw_stacked_bar(
    axes[1],
    pct_pmv,
    PMV_LABELS,
    PMV_COLORS,
    "Time (%)",
    f"PMV (ISO 7730) - air speed {AIR_SPEED_MIN} to {AIR_SPEED_MAX} m/s",
)
draw_stacked_bar(
    axes[2],
    pct_tv,
    TV_LABELS,
    TV_COLORS,
    "Votes (%)",
    "Thermal preference vote - self reported by participant",
)

axes[2].set_xticks(x)
axes[2].set_xticklabels(x_labels, fontsize=8)

fig.tight_layout()

outdir = os.path.join(SCRIPT_DIR, "output")
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, "example_field_study.pdf")
fig.savefig(out, bbox_inches="tight")
plt.show()
print(f"\nFigure saved to {out}")
