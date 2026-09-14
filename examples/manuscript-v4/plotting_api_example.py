from matplotlib import pyplot as plt

from pythermalcomfort.models import pmv_ppd_iso
from pythermalcomfort.plots.matplotlib import ThresholdPlot

plot = (
    ThresholdPlot(pmv_ppd_iso)
    .set_x_axis("tdb", 19.5, 30.5, resolution=1)
    .set_y_axis("rh", 20.0, 80.0, resolution=20)
    .set_params(tr=26, vr=0.1, met=1, clo=0.75, wme=0.0)
    .set_regions(output="pmv", thresholds=[-0.5, 0.5])
    .plot()
)
plot.ax.set_xlabel(r"Dry-bulb Air Temperature ($^\circ$C)")
plot.ax.set_ylabel("Relative Humidity (%)")
plt.tight_layout()
plot.fig.show()
