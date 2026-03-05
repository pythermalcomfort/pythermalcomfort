import inspect

import matplotlib.pyplot as plt
import numpy as np

from pythermalcomfort.models import pmv_ppd_iso, utci

# ----------------------------
# Helper classes and constants
# ----------------------------


class Parameter:
    def __init__(self, name):
        self.name = name


# Map model function to the attribute to plot from the result
MODEL_METRIC = {
    utci: "utci",
    pmv_ppd_iso: "pmv"
}

# ----------------------------
# Model-specific parameter classes
# ----------------------------


class UTCIParameters:
    TDB = Parameter("tdb")
    RH = Parameter("rh")
    V = Parameter("v")
    TR = Parameter("tr")


class PMVParameters:
    TDB = Parameter("tdb")
    RH = Parameter("rh")
    VR = Parameter("vr")
    MET = Parameter("met")
    CLO = Parameter("clo")
    WME = Parameter("wme")
    TR = Parameter("tr")


# ----------------------------
# Main plotting class
# ----------------------------


class RangeScene:
    def __init__(self, model_func, resolution=50):
        self.model_func = model_func
        self.x_var = None
        self.y_var = None
        self.x_range = None
        self.y_range = None
        self.constant_params = {}
        self.resolution = resolution

    def axes(self, axes: dict):
        """
        Dictionary with exactly 2 entries, e.g.,
        .axes({
            UTCIParameters.TDB: (0,30),
            UTCIParameters.RH: (20,80)
        })
        """
        if len(axes) != 2:
            raise ValueError("axes must have exactly two entries (x and y).")

        items = list(axes.items())
        self.x_var, self.x_range = items[0]
        self.y_var, self.y_range = items[1]

        return self

    def parameters(self, params: dict):
        """
        Dictionary mapping Parameter objects to values, e.g.,
        .axes({
            UTCIParameters.V: 0.5,
            UTCIParameters.TR: 25
        })
        """
        sig = inspect.signature(self.model_func)
        valid_params = set(sig.parameters.keys())

        converted = {}

        for param_obj, value in params.items():
            name = param_obj.name

            if name not in valid_params:
                raise ValueError(
                    f"Invalid parameter '{name}'. Valid parameters: {valid_params}"
                )

            converted[name] = value

        self.constant_params = converted

        return self

    def plot(self, cmap="viridis"):
        x = np.linspace(*self.x_range, self.resolution)
        y = np.linspace(*self.y_range, self.resolution)

        X, Y = np.meshgrid(x, y)

        model_inputs = {self.x_var.name: X, self.y_var.name: Y, **self.constant_params}

        result = self.model_func(**model_inputs)

        attr = MODEL_METRIC.get(self.model_func, None)
        Z = getattr(result, attr) if attr else result

        fig, ax = plt.subplots()

        im = ax.imshow(
            Z,
            origin="lower",
            extent=[*self.x_range, *self.y_range],
            aspect="auto",
            cmap=cmap,
        )

        fig.colorbar(im, ax=ax)

        ax.set_xlabel(self.x_var.name)
        ax.set_ylabel(self.y_var.name)

        return ax


# ----------------------------
# Example usage
# ----------------------------

ax1 = (
    RangeScene(utci)
    .axes({
        UTCIParameters.TDB: (10, 40),
        UTCIParameters.RH: (20, 80)
    })
    .parameters({
        UTCIParameters.V: 0.5,
        UTCIParameters.TR: 25
    })
    .plot(cmap="coolwarm")
)
ax1.set_title("UTCI")
plt.show()

ax2 = (
    RangeScene(pmv_ppd_iso)
    .axes({
        PMVParameters.TDB: (18, 30),
        PMVParameters.RH: (30, 70)
    })
    .parameters({
        PMVParameters.MET: 1.2,
        PMVParameters.CLO: 0.5,
        PMVParameters.VR: 0.1,
        PMVParameters.WME: 0.0,
        PMVParameters.TR: 25,
    })
    .plot(cmap="coolwarm")
)
ax2.set_title("PMV")
plt.show()
