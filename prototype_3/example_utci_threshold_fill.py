"""Simple UTCI threshold-fill example using prototype_3 RangeScene."""

from __future__ import annotations

import matplotlib.pyplot as plt
from range_scene import Output, RangeScene

from pythermalcomfort.models import utci

scene = (
    RangeScene(model_func=utci)
    .axes(tdb=(-10.0, 40.0), rh=(20.0, 100.0))
    .fixed(
        v=1.0,
        units="SI",
        limit_inputs=True,
        round_output=False,
    )
)



ax = scene.plot(
    output=Output.UTCI,
    levels=[26.0, 32.0],
    colors=["#4c78a8", "#f2d7b6", "#e15759"],
    x_step=1.0,
    y_step=2.0,
    linewidth=2.0,
    linestyle="--",
)


ax.set_title("UTCI Threshold-Filled Regions")
ax.set_xlabel("Air temperature [degC]")
ax.set_ylabel("Relative humidity [%]")

# Extra: Native Matplotlib styling through scene helpers
scene.adjust_lines(color="black", linewidth=1.5)
scene.adjust_fills(alpha=0.65)


plt.tight_layout()
plt.show()