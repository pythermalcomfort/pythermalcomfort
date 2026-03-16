"""Simple UTCI threshold-fill example using prototype_3 RangeScene."""

from __future__ import annotations

from range_scene import RangeScene

from pythermalcomfort.models import utci

scene = (
    RangeScene(model_func=utci)
    .axes(tdb=(-10.0, 40.0), rh=(20.0, 100.0))
    .fixed(
        vr=1.0,
        units="SI",
        limit_inputs=True,
        round_output=False,
    )
)

scene.allowed_parameters()
