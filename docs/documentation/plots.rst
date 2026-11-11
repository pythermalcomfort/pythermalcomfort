Plots
=====

This section provides an overview of the plotting capabilities available in pythermalcomfort.
The library offers class-based utilities to visualize model results with Matplotlib.
We also provide examples of how to create visually compelling and informative plots using popular Python libraries such as Matplotlib and Seaborn.

To use the plotting functions we included in pythermalcomfort, please install the optional dependencies using:

.. code-block:: bash

    pip install pythermalcomfort[plots]

Typical Use Cases
-----------------

- Comparing the results of different thermal comfort models.
- Visualizing the impact of various parameters (e.g., temperature, humidity, clothing insulation) on model output.
- Creating custom plots to communicate findings effectively.
- Analyse the comfort levels in different environments.
- Process, analyze, and visualise:
    - climate data in different formats (e.g., CSV, Excel, JSON, EPW)
    - results from building simulation software.
    - results from sensor data.

Plotting Examples
-----------------

The `pythermalcomfort.plots.matplotlib` module includes configurable plotting classes.
These classes can be used to compare model results and to visualize how variables affect outcomes.

The `plot` module has a sub-module for different backends, currently supporting Matplotlib only, but we plan to add support for other libraries in the future (e.g., Plotly, Bokeh, Seaborn).

All plotting APIs return Matplotlib handles that can be further customized.
This allows users to create richer visualizations by adding overlays or modifying existing artists.
The full list of customizable parameters is available in the class docstrings.
Use the search functionality in the top-right corner to find specific functions.

The charts apply a few style defaults on top of Matplotlib's, such as the Arial font
(where installed), bold titles, grey axes and no top or right frame lines. They only
style pythermalcomfort charts, and your own Matplotlib settings take precedence: anything
set with ``plt.rcParams``, ``plt.style.use``, a seaborn theme or a ``matplotlibrc`` file
wins. To change or remove a default, edit ``plot_style``, for example to turn bold titles
off:

.. code-block:: python

    from pythermalcomfort.plots.matplotlib import plot_style

    plot_style["axes.titleweight"] = "normal"  # or plot_style.clear() for plain Matplotlib

The style applies while ``plot()`` runs, so a legend, title or text you add to the
returned axes afterwards gets plain Matplotlib styling. Wrap the plot and your additions
in ``plot_style_context()`` to keep the package style:

.. code-block:: python

    from pythermalcomfort.plots.matplotlib import plot_style_context

    with plot_style_context():
        result = AdaptivePlot(model_func=adaptive_ashrae).plot(title="ASHRAE 55")
        result.ax.scatter([25], [29], label="Measured")
        result.ax.legend()

Click on the links below to show some examples of the plotting functions available in pythermalcomfort.
The full list of functions is available in the "Plotting Functions Reference" section at the end of this page.

.. toctree::

    plots/matplotlib/threshold_plot.ipynb
    plots/matplotlib/summary_plot.ipynb
    plots/matplotlib/psychrometric_plot.ipynb
    plots/matplotlib/adaptive_plot.ipynb

Practical Recipes
-----------------

These examples combine different chart types with standard Matplotlib workflows
to produce time series, overlays, and multi-panel figures.

.. toctree::

    plots/matplotlib/example_plots.ipynb

Threshold Plot
--------------

.. autoclass:: pythermalcomfort.plots.matplotlib.ThresholdPlot
    :members: __init__, set_x_axis, set_y_axis, set_params, set_regions, plot
    :member-order: bysource

.. autoclass:: pythermalcomfort.plots.matplotlib.ThresholdPlotResult
    :members:

Summary Plot
------------

.. autoclass:: pythermalcomfort.plots.matplotlib.SummaryPlot
    :members: __init__, set_regions, plot
    :member-order: bysource

.. autoclass:: pythermalcomfort.plots.matplotlib.SummaryPlotResult
    :members:

Psychrometric Plot
------------------

.. autoclass:: pythermalcomfort.plots.matplotlib.PsychrometricPlot
    :members: __init__, set_x_axis, set_y_axis, set_params, set_regions, plot
    :member-order: bysource
