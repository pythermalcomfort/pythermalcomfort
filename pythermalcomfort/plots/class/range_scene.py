import inspect
import matplotlib.pyplot as plt
import numpy as np
import warnings
from copy import deepcopy

class RangeScene:
    """
    A class for visualizing 2D model outputs over a parameter range.

    Provides a chainable interface to define x and y axes, fix other model parameters,
    adjust plotting styles, and plot results as fields or contour lines.

    Parameters
    ----------
    model_func : callable
        A model function that accepts named parameters and returns an object with attributes
        representing different metrics.

    Attributes
    ----------
    x_var : str
        The name of the model parameter used for the x-axis.
    y_var : str
        The name of the model parameter used for the y-axis.
    x_range : tuple
        The min and max values for the x-axis.
    y_range : tuple
        The min and max values for the y-axis.
    constant_params : dict
        Fixed model parameters not used as axes.
    _field_style : dict
        Styling options for contourf or imshow plots.
    _line_style : dict
        Styling options for contour lines.
    """
    def __init__(self, model_func):
        self.model_func = model_func

        self.x_var = None
        self.y_var = None
        self.x_range = None
        self.y_range = None
        self.constant_params = {}

        sig = inspect.signature(model_func)
        self.params = sig.parameters
        self.param_names = set(self.params.keys())

        self._field_style = {}
        self._line_style = {}

    def _validate_axis_input(self, kwargs, axis_name: str):
        """
        Validate input for x or y axis and return parameter name and numeric range.

        Parameters
        ----------
        kwargs : dict
            Single key-value pair mapping a model parameter name to a 2-element list or tuple
            representing the axis range (min, max).
        axis_name : str
            Name of the axis ("x" or "y") used in error messages.

        Returns
        -------
        name : str
            The validated parameter name to be used as axis.
        rng : tuple of float
            The validated (min, max) numeric range.

        Raises
        ------
        ValueError
            If more than one parameter is provided, parameter name is invalid,
            range is not length 2, contains non-numeric values, or min >= max.
        """
        if len(kwargs) != 1:
            raise ValueError(f"{axis_name}() must have exactly 1 parameter")

        name, rng = next(iter(kwargs.items()))
        if name not in self.param_names:
            raise ValueError(f"Invalid parameter '{name}'")

        if not (isinstance(rng, (tuple, list)) and len(rng) == 2):
            raise ValueError(f"{axis_name} range must be a 2-element tuple/list")

        try:
            min_val, max_val = float(rng[0]), float(rng[1])
        except (TypeError, ValueError):
            raise ValueError(f"{axis_name} range values must be numeric")

        if min_val >= max_val:
            raise ValueError(f"{axis_name} range requires min < max (got {min_val} >= {max_val})")

        return name, (min_val, max_val)

    def x(self, **kwargs):
        """
        Set the x-axis variable and its range.

        Parameters
        ----------
        kwargs : dict
            Single key-value pair where key is a parameter name from the model,
            and value is a 2-element tuple/list specifying (min, max).

        Returns
        -------
        self : RangeScene
            Returns self to allow method chaining.

        Raises
        ------
        ValueError
            If more than one parameter is provided, or the parameter name is invalid,
            or the range is not valid (non-numeric or min >= max).
        """
        name, rng = self._validate_axis_input(kwargs, "x")
        self.x_var = name
        self.x_range = rng
        return self

    def y(self, **kwargs):
        """
        Set the y-axis variable and its range.

        Parameters
        ----------
        kwargs : dict
            Single key-value pair where key is a parameter name from the model,
            and value is a 2-element tuple/list specifying (min, max).

        Returns
        -------
        self : RangeScene
            Returns self to allow method chaining.

        Raises
        ------
        ValueError
            If more than one parameter is provided, or the parameter name is invalid,
            or the range is not valid (non-numeric or min >= max).
        """
        name, rng = self._validate_axis_input(kwargs, "y")
        self.y_var = name
        self.y_range = rng
        return self

    def _handle_wind_alias(self, kwargs):
        """
        Handle wind speed alias parameters for the model.

        Converts between 'v' and 'vr' if the model expects one but the other is provided.
        Issues a warning when automatic conversion occurs.

        Parameters
        ----------
        kwargs : dict
            Dictionary of parameter names and values, potentially including 'v' or 'vr'.

        Returns
        -------
        dict
            A copy of kwargs with 'v' or 'vr' renamed according to the model signature.

        Raises
        ------
        ValueError
            If both 'v' and 'vr' are provided simultaneously.
        """
        kwargs = kwargs.copy()

        has_v = "v" in kwargs
        has_vr = "vr" in kwargs

        if has_v and has_vr:
            raise ValueError("Provide either 'v' or 'vr', not both.")

        if "vr" in self.param_names:
            if has_v:
                warnings.warn(
                    "Parameter 'v' is provided but the model expects 'vr'. "
                    "Automatically converting 'v' -> 'vr'.",
                    stacklevel=2
                )
                kwargs["vr"] = kwargs.pop("v")

        elif "v" in self.param_names:
            if has_vr:
                warnings.warn(
                    "Parameter 'vr' is provided but the model expects 'v'. "
                    "Automatically converting 'vr' -> 'v'.",
                    stacklevel=2
                )
                kwargs["v"] = kwargs.pop("vr")

        return kwargs

    def _validate_fixed_params(self, kwargs):
        """
        Validate that fixed model parameters are allowed and scalar.

        Ensures that:
        - Each parameter exists in the model signature.
        - Parameter is not already used as an x or y axis.
        - Parameter value is a scalar (not list, tuple, or ndarray).

        Parameters
        ----------
        kwargs : dict
            Dictionary of parameter names and their fixed values.

        Raises
        ------
        ValueError
            If a parameter name is invalid or already used as an axis.
        TypeError
            If a parameter value is not a scalar.
        """
        for name, value in kwargs.items():
            if name not in self.param_names:
                raise ValueError(f"Invalid parameter '{name}'")
            if name in (self.x_var, self.y_var):
                raise ValueError(f"Parameter '{name}' is already used as an axis")
            if isinstance(value, (list, tuple, np.ndarray)):
                raise TypeError(f"{name} must be a fixed scalar value")

    def fixed(self, **kwargs):
        """
        Set fixed parameters for the model that are not used as axes.

        Handles wind speed alias ('v' <-> 'vr') automatically if needed.

        Parameters
        ----------
        kwargs : dict
            Key-value pairs of parameter names and fixed scalar values.

        Returns
        -------
        self : RangeScene
            Returns self to allow method chaining.

        Raises
        ------
        ValueError
            If a parameter is invalid or already used as an axis.
        TypeError
            If a parameter value is not scalar.
        """
        kwargs = self._handle_wind_alias(kwargs)
        self._validate_fixed_params(kwargs)
        self.constant_params.update(kwargs)
        return self

    def _copy(self):
        """
        Create a deep copy of the RangeScene instance.

        This method duplicates all axis settings, fixed parameters, and plotting styles,
        returning a new RangeScene object that can be modified independently.
        Useful for chainable method calls such as `with_field` or `with_lines`.

        Returns
        -------
        RangeScene
            A new RangeScene instance with the same configuration as the original.
        """
        new = RangeScene(self.model_func)

        new.x_var = self.x_var
        new.y_var = self.y_var
        new.x_range = deepcopy(self.x_range)
        new.y_range = deepcopy(self.y_range)
        new.constant_params = deepcopy(self.constant_params)

        new.params = self.params
        new.param_names = self.param_names.copy()

        new._field_style = deepcopy(getattr(self, "_field_style", {}))
        new._line_style = deepcopy(getattr(self, "_line_style", {}))

        return new

    def with_field(self, **kwargs):
        """
        Return a copy of the RangeScene instance with updated field (contour/heatmap) style.

        This method allows customizing plotting options for the field, such as
        color map (`cmap`), value limits (`vmin`, `vmax`), and other matplotlib kwargs.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments specifying field style options for matplotlib plotting.

        Returns
        -------
        RangeScene
            A new RangeScene instance with updated field style, leaving the original unchanged.
            Supports chainable calls.
        """
        new = self._copy()
        new._field_style.update(kwargs)
        return new

    def with_lines(self, **kwargs):
        """
        Return a copy of the RangeScene instance with updated line (contour) style.

        This method allows customizing plotting options for contour lines, such as
        color (`colors`), line width (`linewidths`), and other matplotlib kwargs.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments specifying line style options for matplotlib contour plotting.

        Returns
        -------
        RangeScene
            A new RangeScene instance with updated line style, leaving the original unchanged.
            Supports chainable calls.
        """
        new = self._copy()
        new._line_style.update(kwargs)
        return new

    def plot(self, metric=None, resolution=1000, levels=None, fill=True):
        """
        Plot the model output over the defined x and y ranges.

        Parameters
        ----------
        metric : str, optional
            Name of the metric to plot. Defaults to the first attribute of the model result.
        resolution : int, optional
            Number of points along each axis. Defaults to 1000.
        levels : list or int, optional
            Contour levels for plotting. If provided, contour lines will be drawn.
        fill : bool, default=True
            Whether to fill the contour (True) or use imshow (False).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure object.
        ax : matplotlib.axes.Axes
            The axes object containing the plot.

        Raises
        ------
        ValueError
            If x or y axes are not defined.
        """
        if self.x_var is None or self.y_var is None:
            raise ValueError("Both x and y axes must be defined")

        x = np.linspace(*self.x_range, resolution)
        y = np.linspace(*self.y_range, resolution)
        X, Y = np.meshgrid(x, y)

        model_inputs = {
            self.x_var: X,
            self.y_var: Y,
            **self.constant_params
        }

        result = self.model_func(**model_inputs)

        if metric is None:
            metric = list(vars(result).keys())[0]

        Z = getattr(result, metric)

        fig, ax = plt.subplots()

        if fill and levels is not None:
            cf = ax.contourf(
                X, Y, Z,
                levels=levels,
                **self._field_style
            )
            plt.colorbar(cf, ax=ax, label=metric)
        else:
            im = ax.imshow(
                Z,
                extent=[*self.x_range, *self.y_range],
                origin="lower",
                aspect="auto",
                **self._field_style
            )
            plt.colorbar(im, ax=ax, label=metric)

        if levels is not None:
            cs = ax.contour(
                X, Y, Z,
                levels=levels,
                **self._line_style
            )
            ax.clabel(cs, inline=True, fontsize=8)

        ax.set_xlabel(self.x_var)
        ax.set_ylabel(self.y_var)

        return fig, ax
