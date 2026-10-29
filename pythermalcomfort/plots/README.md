# Plotting Overview

`pythermalcomfort.plots.matplotlib` provides class-based plotting utilities for threshold-region and summary visualizations.

The API is intentionally compact:
- `ThresholdPlot`: build threshold-region charts by configuring model, axes, fixed parameters, and regions.
- `SummaryPlot`: build compact horizontal/vertical summaries from DataFrame outputs.

Both APIs return Matplotlib handles so users can customize visuals with standard Matplotlib code.

## Quick Start

```python
import matplotlib.pyplot as plt

from pythermalcomfort.models import pmv_ppd_iso
from pythermalcomfort.plots.matplotlib import SummaryPlot, ThresholdPlot

threshold = (
    ThresholdPlot(pmv_ppd_iso)
    .set_x_axis("tdb", 18.0, 34.0, resolution=0.2)
    .set_y_axis("rh", 20.0, 100.0, resolution=0.5)
    .set_params(vr=0.10, met=1.2, clo=0.5, wme=0.0)
    .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    .plot(title="PMV Threshold Regions")
)

# Full matplotlib customization remains available
threshold.ax.set_xlabel("Air temperature [degC]")
threshold.ax.set_ylabel("Relative humidity [%]")

df = ...
summary = (
    SummaryPlot(df)
    .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    .plot(title="Measured PMV Summary")
)

plt.show()
```

## How `ThresholdPlot` draws its boundaries

Every boundary is solved by bisection, one row of the plot at a time: where the
output crosses a threshold, and where the model leaves its applicability
limits. The regions are then filled between those curves. Three consequences:

- `resolution` is optional on both axes. It does not set the precision of a
  boundary, so it only matters for a model that turns sharply enough to step
  over a feature between samples.
- The boundaries are available as coordinate arrays, not just drawn paths.
- A threshold crossed more than once per row gets one curve per crossing.

```python
lower, upper = threshold.boundaries  # one BoundaryCurve per threshold branch
print(lower.threshold, lower.branch)  # -0.5 0
print(lower.x, lower.y)  # NaN where the threshold is not crossed
```

`ppd` is the case that needs branches: it falls to a minimum at neutrality and
rises again, so each threshold is crossed on both sides of the comfort dip and
"PPD below 10" is a strip with "above 10" on either side — five bands drawn
from three regions.

The layout has to be the same in every row: the same thresholds crossed in the
same order, running the same way. An internal hole in the valid area, or a pair
of crossings that appears only partway up the chart, raises `ValueError`.
Narrowing the axis ranges to where the model is well behaved usually fixes it.

## Chart styling

`ThresholdPlot` and `PsychrometricPlot` set their own look on the axis, so a
chart drawn on an `ax` you created matches one where `plot()` made the axis:

- no grid — call `result.ax.grid(True)` to put it back
- no top spine; `ThresholdPlot` also hides the right spine
- no boundary lines — pass `show_lines=True` to draw them
- `ThresholdPlot` out-of-model-limits areas in neutral gray `#C4C9CC`,
  overridable via `invalid_color`
- `PsychrometricPlot` places its humidity-ratio axis on the right and uses the
  lighter neutral gray `#DADCDD` for out-of-model-limits areas, also overridable
  via `invalid_color`

`AdaptivePlot` keeps its grid, which its bands are read against.

## Design Notes

- Keep configuration explicit and ordered: set axes and params first, then regions, then render.
- Keep implementation simple and maintainable.
- Keep output open for customization by returning artist handles (`ax`, fills, lines, legend, processed data).
