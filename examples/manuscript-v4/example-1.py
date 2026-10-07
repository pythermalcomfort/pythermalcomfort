"""Example 1: Compare PMV, UTCI, and Heat Index Lu across a T-RH grid,
and display PMV comfort zones on a psychrometric chart with scatter data.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pythermalcomfort.models import heat_index_lu, pmv_ppd_iso, utci
from pythermalcomfort.plots.matplotlib import (
    PsychrometricPlot,
    SummaryPlot,
    ThresholdPlot,
)
from pythermalcomfort.psychrometrics import psy_ta_rh

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTDIR, exist_ok=True)


RNG = np.random.default_rng(42)

# Colour palette shared across both figures for visual consistency
C_COOL = "#74add1"
C_NEUTRAL = "#abdda4"
C_CAUTION = "#fee08b"
C_STRONG = "#f46d43"
C_EXTREME = "#d73027"

# ranges
T_MIN, T_MAX = 23, 37.5
RH_MIN, RH_MAX = 10.0, 95.0
RESOLUTION_T, RESOLUTION_RH = 1, 7.0

legend_kws = {"loc": "lower center", "bbox_to_anchor": (0.5, 1), "ncol": 4}
Y_TITLE_OFFSET = 1.2  # vertical offset for panel titles

# Figure 1: Three-model comparison

fig, axes = plt.subplots(
    3, 1, figsize=(7, 8.5), sharex=True, sharey=True, constrained_layout=True
)

# Panel A -- PMV (ISO 7730)
# limit_inputs=False extends the model beyond its standard applicability
# domain (tdb > 30 $^\circ$C) so the full 23--37.5 $^\circ$C range can be compared
# against the other two models. See caption for details.
(
    ThresholdPlot(pmv_ppd_iso)
    .set_x_axis("tdb", T_MIN, T_MAX, resolution=RESOLUTION_T)
    .set_y_axis("rh", 10.0, 95.0, resolution=RESOLUTION_RH)
    .set_params(vr=0.5, met=1.2, clo=0.5, wme=0.0, limit_inputs=False)
    .set_regions(
        output="pmv",
        thresholds=[-0.5, 0.5, 3],
        colors=[C_COOL, C_NEUTRAL, C_STRONG, C_EXTREME],
    )
    .plot(ax=axes[0], legend_kws=legend_kws)
)
axes[0].set_ylabel("Relative humidity (%)")
axes[0].set_xlabel("")
axes[0].set_title("PMV (ISO 7730)", y=Y_TITLE_OFFSET)

legend_kws.update({"ncol": 2})

# Panel B -- UTCI
# tr auto-links to tdb (person in shade). At 23--37.5 $^\circ$C the relevant UTCI
# categories are no-stress through very strong heat stress.
(
    ThresholdPlot(utci)
    .set_x_axis("tdb", T_MIN, T_MAX, resolution=RESOLUTION_T)
    .set_y_axis("rh", 10.0, 95.0, resolution=RESOLUTION_RH)
    .set_params(v=0.5)
    .set_regions(
        output="utci",
        thresholds=[26, 32, 38],
        labels=[
            r"No thermal stress (UTCI<26$^\circ$C)",
            "Moderate heat stress",
            "Strong heat stress",
            r"Very strong heat stress (UTCI>38$^\circ$C)",
        ],
        colors=[C_NEUTRAL, C_CAUTION, C_STRONG, C_EXTREME],
    )
    .plot(ax=axes[1], legend_kws=legend_kws)
)
axes[1].set_ylabel("Relative humidity (%)")
axes[1].set_xlabel("")
axes[1].set_title("UTCI", y=Y_TITLE_OFFSET + 0.15)

# Panel C -- Heat Index (Lu and Romps 2022)
# heat_index_lu only requires tdb and rh; no set_params needed.
# Standard Heat Index intervals: 27, 32, 41 (degrees Celsius)
(
    ThresholdPlot(heat_index_lu)
    .set_x_axis("tdb", T_MIN, T_MAX, resolution=RESOLUTION_T)
    .set_y_axis("rh", 10.0, 95.0, resolution=RESOLUTION_RH)
    .set_regions(
        output="hi",
        thresholds=[27, 32, 41],
        labels=[
            r"No risk (HI<27$^\circ$C)",
            r"Caution (27≤HI<32$^\circ$C)",
            r"Extreme caution (32≤HI<41$^\circ$C)",
            r"Danger (HI≥41$^\circ$C)",
        ],
        colors=[C_NEUTRAL, C_CAUTION, C_STRONG, C_EXTREME],
    )
    .plot(ax=axes[2], legend_kws=legend_kws)
)
axes[2].set_xlabel(r"Dry-bulb temperature ($^\circ$C)")
axes[2].set_ylabel("Relative humidity (%)")
axes[2].set_title("Heat Index", y=Y_TITLE_OFFSET + 0.15)

for ax in axes:
    ax.grid(False)

fig.savefig(os.path.join(OUTDIR, "example_1.pdf"))
plt.show()

# Figure 2: PsychrometricPlot + SummaryPlot

legend_kws = {"loc": "lower center", "bbox_to_anchor": (0.5, 1), "ncol": 2}

# Synthetic indoor measurements (seeded for reproducibility)
n = 60
tdb_meas = RNG.uniform(20.0, 28.0, n)
rh_meas = RNG.uniform(25.0, 75.0, n)
tr_meas = tdb_meas + RNG.uniform(-1.0, 2.0, n)
vr_meas = np.clip(RNG.normal(0.1, 0.05, n), 0.05, 0.3)
met_meas = np.clip(RNG.normal(1.2, 0.1, n), 1.0, 1.5)
clo_meas = np.clip(RNG.normal(0.6, 0.1, n), 0.4, 0.9)

# PMV computed from the full input set (tr, vr, met, clo vary per point).
# This value is used by SummaryPlot and reflects each observation's true
# comfort state - information a 2-D psychrometric plot cannot capture.
pmv_vals = np.array(
    pmv_ppd_iso(
        tdb=tdb_meas.tolist(),
        tr=tr_meas.tolist(),
        vr=vr_meas.tolist(),
        rh=rh_meas.tolist(),
        met=met_meas.tolist(),
        clo=clo_meas.tolist(),
        limit_inputs=False,
    ).pmv
)
df_meas = pd.DataFrame({"pmv": pmv_vals})

PMV_COLORS = [C_COOL, C_NEUTRAL, C_STRONG]
PMV_LABELS = ["Cool (PMV<-0.5)", "Neutral", r"Warm (PMV$\geq$0.5)"]

fig2, (ax_psy, ax_sum) = plt.subplots(
    1, 2, figsize=(7, 4), width_ratios=[6, 1.5], constrained_layout=True
)

# Psychrometric chart: PMV comfort zones at reference conditions
# (tr = tdb auto-linked, vr = 0.1 m/s, met = 1.2, clo = 0.5).
(
    PsychrometricPlot(pmv_ppd_iso)
    .set_x_axis("tdb", 19, 29, resolution=1)
    .set_y_axis("hr", 0.0, 25.0, resolution=1.5)
    .set_params(vr=0.1, met=1.2, clo=0.5, wme=0.0)
    .set_regions(
        output="pmv",
        thresholds=[-0.5, 0.5],
        labels=PMV_LABELS,
        colors=PMV_COLORS,
    )
    .plot(ax=ax_psy, legend=True, legend_kws=legend_kws)
)
ax_psy.set_xlabel(r"Dry-bulb temperature ($^\circ$C)")
ax_psy.grid(False)
ax_psy.spines["top"].set_visible(False)
ax_psy.spines["right"].set_visible(False)
ax_psy.yaxis.set_label_position("right")
ax_psy.yaxis.tick_right()
# The y-label is supplied by PsychrometricPlot itself and is already in g/kg.

# Overlay scatter measurements. psy_ta_rh returns humidity ratio in kg/kg dry
# air, while the chart's y-axis is in g/kg dry air, so scale before plotting.
hr_meas = psy_ta_rh(tdb_meas, rh_meas).hr * 1000.0
ax_psy.scatter(
    tdb_meas,
    hr_meas,
    s=25,
    c="black",
    zorder=5,
    label="Indoor measurements",
)

# Summary bar: PMV distribution from the full input set.
(
    SummaryPlot(df_meas)
    .set_regions(
        output="pmv",
        thresholds=[-0.5, 0.5],
        labels=[],
        colors=PMV_COLORS,
    )
    .plot(ax=ax_sum, legend=False, vertical=True)
)

fig2.savefig(os.path.join(OUTDIR, "pmv_psychrometric_comfort.pdf"))
plt.show()
