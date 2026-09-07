Changelog
=========

Unreleased
----------

* Sped up ``two_nodes_gagge_sleep`` by compiling its stateful simulation loop
  with Numba while preserving its public output values and shapes. Empty
  ``tdb``/``tr``/``v``/``rh``/``clo``/``thickness_quilt`` inputs now raise a
  clear ``ValueError`` instead of failing with an unrelated ``TypeError``.
* Pinned ``tests/conftest.py``'s ``validation-data-comfort-models`` fixture URL to the
  ``v1.0.0`` tag instead of ``main``, so upstream fixture changes can't silently affect
  CI before the pin is deliberately bumped and reviewed. See ``CONTRIBUTING.rst``'s
  "Keeping the validation-data-comfort-models pin current" section.
* Addressed Copilot review feedback on the 4.4.1 ``phs`` fix: pass ``param_name``
  explicitly to ``valid_range()`` for the ``(tr - tdb)`` check, and added regression
  tests for the applicability-limit and minute-1 skin-temperature behavior.
* Fixed the saturation vapour pressure calculation in ``utci``
  (`#372 <https://github.com/pythermalcomfort/pythermalcomfort/issues/372>`_): the
  Hardy/Wexler equation's ``ln(T)`` term used ``np.log1p`` (which computes
  ``ln(1 + T)``) instead of ``np.log``, inflating the saturation vapour pressure by
  ~1%. The resulting UTCI error is negligible in mild conditions (~0.03 °C at 25 °C,
  50% RH) but grows to ~0.7 °C at 40 °C, 80% RH, where UTCI matters most for heat
  stress assessment. Updated the affected hard-coded test expectations and added a
  regression test cross-checking ``utci``'s vapour pressure against ``p_sat``.

4.4.1 (2026-08-18)
------------------

* Fixed the ``phs`` applicability limits (`#225 <https://github.com/pythermalcomfort/pythermalcomfort/issues/225>`_):
    - the ``tr`` limit is now checked against ``ISO 7933 Annex A, Table A.1``'s actual
      ``0 < (tr - tdb) < 60`` range, instead of checking raw ``tr`` against ``(0, 60)``.
    - the metabolic rate limit is now standard-specific: ``100-450 W/m2`` (1.7-7.5 met)
      for the 2004 standard, ``56-250 W/m2`` (0.96-4.3 met) for the 2023 standard,
      instead of always using the 2004 range.
* Added the ISO 7933:2023 Annex E minute-1 skin temperature special case to the
  ``phs`` 2023 model (``t_sk`` is forced to its equilibrium value on the first
  minute), matching the standard's own reference program. This special case is
  not present in the 2004 edition.

4.4.0 (2026-07-25)
------------------

* Added ``ireq`` model to calculate Required Clothing Insulation (IREQ) and
  Duration Limited Exposure (DLE) based on ISO 11079.

4.3.0 (2026-07-24)
------------------

* Added ``SummaryPlot.set_categories()`` for summarizing an already-classified
  per-row category array — e.g. adaptive comfort's per-row acceptability
  bands, which can't be reduced to one continuous column plus fixed
  thresholds the way ``set_regions()`` handles PMV/UTCI-style outputs. See
  the ``set_categories()`` docstring for an ``np.select``-based recipe.

4.2.0 (2026-07-24)
------------------

* Added a ``pa`` (water vapour partial pressure) applicability check to
  ``pmv_ppd_iso``, per the ISO 7730 Clause 4 limit of 0 Pa to 2 700 Pa. Inputs
  outside this range (e.g. ``tdb=30``, ``rh=100`` gives ``pa`` ~4 243 Pa) now
  return ``nan`` instead of a value outside the standard's applicability.
* Fixed ``clo_dynamic_iso`` to estimate walking speed using the ISO 7730
  Annex C / ISO 9920 formula for undefined walking speed
  (``v_walk = 0.0052 * (met * 58.15 - 58)``, clipped to 0-0.7 m/s) instead of
  reusing ``v_relative``'s activity-generated-air-speed formula, which is a
  distinct formula intended for the whole-body PMV heat balance rather than
  the clothing dynamic insulation correction.
* Corrected the initial guess for clothing surface temperature in
  ``pmv_ppd_iso`` to match the corrected Annex D formula in ISO 7730:2025
  (``3.5 * (6.45 * icl + 0.1)``, missing the ``6.45 *`` factor present in the
  ISO 7730:2005 Annex D listing). This only affects the starting point of the
  iterative solver and does not change any output value.
* Added ``"7730-2025"`` as a supported ``model`` value for ``pmv_ppd_iso``
  and made it the default, since ISO 7730:2025 is now the current edition of
  the standard. ``"7730-2005"`` remains supported for backwards
  compatibility; both currently return identical results since the PMV/PPD
  formulae are unchanged between editions.

4.1.1 (2026-07-20)
------------------

* Sped up ``cooling_effect`` (~10x) by calling the already numba-jitted Gagge
  two-node kernel directly instead of the full ``set_tmp()`` public API on
  every root-finding iteration.
* Sped up ``solar_gain`` (~100x+) with numba: table-based interpolation and
  posture handling converted to JIT-compiled code.
* Pinned ``pillow>=10.3.0`` in docs requirements to resolve a transitive
  Snyk-flagged vulnerability.
* Reduced the ``build-test-publish-testPyPI.yml`` CI matrix to speed up
  TestPyPI release checks.

4.1.0 (2026-07-20)
------------------

* Added ``heat_index_schoen``, the Temperature-Humidity Index (THI) heat
  index model in accordance with Schoen (2005).
* Sped up ``heat_index_rothfusz``, ``heat_index_schoen``, and
  ``heat_index_lu`` with numba JIT compilation. ``heat_index_lu`` (an
  iterative root-solver) sees the largest gain, roughly 29x faster.

4.0.3 (2026-07-20)
------------------

* Added a ``Sports.CROQUET`` preset to ``sports_heat_stress_risk``
  (``clo=0.7, met=4.5, vr=0.5, duration=90``).
* Fixed the extreme-risk interpolation segment so it reaches a risk level of
  4.9 exactly 5 °C above the extreme threshold, instead of reaching it early
  at +4.5 °C and leaving the last 0.5 °C of the range dead.
* Fixed an inconsistency in ``sports_heat_stress_risk``'s extreme-risk
  interpolation: the upper anchor temperature used internally to scale the
  risk level was the raw, unrounded solver output, while the ``t_extreme``
  value returned to callers is rounded to one decimal. This could produce a
  risk level inconsistent with the documented/returned thresholds. The
  interpolation now uses the same rounded ``t_extreme`` that is returned.

4.0.2 (2026-06-23)
------------------

* Improved sports heat stress risk interpolation within the extreme range: the
  upper anchor temperature (where risk reaches 4.9) is now computed dynamically
  as ``t_extreme + 5 °C``. This makes the extreme-range scale consistent across
  humidity conditions — the risk always spans exactly 5 °C above the
  humidity-dependent extreme threshold regardless of ambient conditions.

4.0.1 (2026-06-17)
------------------

* Added Python 3.14 support. Removed ``pytest-travis-fold`` (unmaintained,
  incompatible with Python 3.14) and lifted the ``pytest<7`` cap.
* Fixed duplicate parametrize IDs in the ridge-regression test suite,
  which pytest 9 now rejects (``ast.Str`` was removed in Python 3.14).

4.0.0 (2026-06-16)
------------------

.. note::
    Version 4.0.0 introduces a new **Matplotlib plotting API** under
    ``pythermalcomfort.plots.matplotlib``. All four plot classes follow a
    fluent builder pattern — chain setter calls and finish with ``.plot()``
    to receive standard Matplotlib ``Figure`` / ``Axes`` handles for full
    customisation.

**New plotting module** (``pythermalcomfort.plots.matplotlib``)

* ``ThresholdPlot`` — shade comfort/stress regions on any two-axis chart
  (e.g. operative temperature vs. relative humidity, temperature vs. air
  speed). Configure regions via ``set_regions(thresholds, labels, colors)``.
* ``SummaryPlot`` — horizontal or vertical bar-summary chart built from a
  ``pandas.DataFrame``; useful for comparing multiple spaces or scenarios at
  a glance.
* ``AdaptivePlot`` — ready-made adaptive comfort chart for ASHRAE 55 and
  EN 16798, with configurable comfort bands.
* ``PsychrometricPlot`` — psychrometric chart with overlaid comfort regions.

All classes share a common ``BasePlot`` base and centralised visual defaults
(``_shared.py``), making it easy to apply a consistent house style.

**Other changes**

* Refactored type hints across model function signatures to use a
  ``NumericInput`` alias (``float | int | np.floating | np.integer``),
  improving IDE auto-complete and static-analysis accuracy.
* Added ``hr_to_rh`` utility for humidity-ratio → relative-humidity
  conversion.
* Minor cooling-effect calculation streamlining and constant centralisation.

3.9.8 (2026-05-25)
------------------

* Added optional ``round_output`` parameter to ``adaptive_ashrae`` and ``adaptive_en``
  to control rounding of output values.
* Added ``limit_inputs`` parameter to ``ankle_draft`` and ``vertical_tmp_grad_ppd``,
  consistent with other model functions.
* ``ankle_draft`` and ``vertical_tmp_grad_ppd`` now raise ``UserWarning`` when inputs
  exceed model applicability limits.
* Fixed ``compliance`` attribute being included in non-ASHRAE PMV model outputs;
  it is now only returned by ``pmv_ppd_ashrae``.
* Fixed UTCI stress category mapping when ``units="IP"``; categories were
  incorrectly mapped before IP unit conversion.

3.9.3 (2026-05-01)
------------------

* Maintenance release: internal CI pipeline improvements and dependency updates.
  No user-facing changes.

3.9.2 (2026-04-14)
------------------

* Updated `sports_heat_stress_risk` so `risk_level_interpolated` now uses `1.0-4.0` instead of `0.0-3.0`.
* Updated `sports_heat_stress_risk` to enforce the sport-specific minimum air speed (`sport.vr`).

3.9.1 (2026-02-25)
------------------

* Improved speed of PHS model.

3.9.0 (2026-02-03)
------------------

* Added `sports_heat_stress_risk` function to assess heat stress risk for athletes during outdoor sports activities based on environmental conditions. Addresses issue #237.

3.7.0 (2025-10-28)
------------------

* Added machine learning model to predict skin and rectal temperature `ridge_regression_predict_t_re_t_sk`.

3.6.1 (2025-10-07)
------------------

* Fix issue with `disc` calculation in the two_nodes_gagge model and limiting its value to 6. Close #251
* Improve documentation for the `disc` function.
* PMV ASHRAE model returns the `compliance` boolean value with the ASHRAE 55:2023 standard. Close #253
* Improve formatting of models outputs to the console.

3.6.0 (2025-09-22)
------------------

.. warning::
    breaking change for the `phs` function:
    - the `phs` function now returns all the outputs needed to use the outputs of a previous calculation as inputs for a new calculation.
    - removed outputs: ``water_loss``, ``water_loss_watt``
    - added outputs (with units):

        * ``sweat_loss_g`` [g] — cumulative sweat mass per person (not area‑normalised)
        * ``sweat_rate_watt`` [W·m⁻²] — instantaneous evaporative heat flux at skin
        * ``evap_load_wm2_min`` [W·min·m⁻²] — accumulated evaporative load for chaining

    Migration:

    .. code-block:: python

        # <= 3.5.x
        grams = res.water_loss
        w_m2 = res.water_loss_watt
        # >= 3.6.0
        grams = res.sweat_loss_g
        w_m2 = res.sweat_rate_watt
        carry = res.evap_load_wm2_min  # for multi‑segment runs


3.5.1 (2025-09-15)
------------------

* Improved documentation on how to contribute to the project

3.5.0 (2025-09-10)
------------------

* Added the `scale_winds_speed_log` function to scale wind speed.

3.4.3 (2025-07-31)
------------------

* fix: wind chill temperature was not imported in pythermalcomfort.models

3.4.2 (2025-07-22)
------------------

* fixed unit of `sweat_rate` in the PHS model

3.4.1 (2025-07-14)
------------------

.. warning::
    removed support for Python 3.8 and 3.9
    pythermalcomfort now requires Python 3.10 or higher.
    Added support for Python 3.13


* fixed some typo in the documentation
* better formatted the code

3.4.0 (2025-06-08)
------------------

* Added the `work_capacity_dunne`.
* Added the `work_capacity_hothaps`.
* Added the `work_capacity_iso`.
* Added the `work_capacity_niosh`.

3.3.0 (2025-06-05)
------------------

* Added the `two_nodes_gagge_ji` function to calculate the two-node model for older individuals
* Added the `Temperature-Humidity Index (THI)`.

3.2.0 (2025-05-20)
------------------

* Added the `two_nodes_gagge_sleep` function to calculate the two-node model for sleeping individuals
* Added the `ESI` function to calculate the Environmental Stress Index.

3.1.0 (2025-04-28)
-------------------
* Updated the PHS model in compliance with the ISO 7933:2023 standard
    - Added default‑kwarg overrides for 2023 mode (f_r, t_re, t_cr_eq)
    - removed unused variable `round` from `default_kwargs`
* Included test cases according to the ISO 7933:2023 standard
* Added `AutoStrMixin` to provide a formatted `__str__` representation for result classes

.. note::
    By default the ISO 7933:2023 standard will now be used for the PHS model.
    The 2004 standard can still be specified as an optional parameter

.. warning::
    The third ISO 7933:2023 test case is currently failing and has been marked xfail while the discrepancy is investigated.

3.0.1 (2025-04-14)
-------------------

* allow np.float and np.int as inputs to all functions
* fixed documentation for phs - met units

3.0.0 (2025-02-03)
-------------------

.. warning::
    pythermalcomfort version 3.0.0 introduces some breaking changes.

    **How functions return results:**
    as the functions now return dataclass instances with the calculation results.
    This change enhances the structure and accessibility of the results.
    For example:

    .. code-block:: python

        from pythermalcomfort.models import pmv_ppd_iso

        result = pmv_ppd_iso(
            tdb=[22, 25], tr=25, vr=0.1, rh=50, met=1.4, clo=0.5, model="7730-2005"
        )
        print(result.pmv)  # [-0.  0.41]

    This update aims to make the package more user-friendly and to provide a more organized way to access all calculation results.

    **Moved functions**
    Moved all the functions that were in the `psychrometrics.py` file to the `utilities.py` file.

    **Changed function names**
    All the PMV functions have been renamed using the following format: `pmv_XXX` where XXX is the standard or the model name.

    **PMV function**
    The pmv_ppd function now has been split into two functions: pmv_ppd_iso and pmv_ppd_ashrae.

.. note::
    We have updated all functions to accept Numpy arrays as inputs, allowing you to pass multiple values at once for faster results.
    Single values are still accepted, and the functions will return results as before.
    Additionally, we have synchronized the tests with the R comf package to ensure consistent calculation results across both packages.

    Other improvements include:

    * Enhanced documentation with more examples.
    * Better described the models.
    * Added more tests to ensure calculation accuracy.
    * Implemented input validation to ensure inputs are within model applicability limits.
    * Harmonized input names across all functions.
    * Added surveys to assess thermal comfort to the documentation.
    * Added a detailed section about clothing insulation.

2.10.0 (2024-03-18)
-------------------

* allow n-dimensional arrays for ``pet_steady`` and speedup ``p_sat`` calculation


2.9.1 (2024-01-19)
-------------------

* Fixed error calculation of mass sweating in PET mode, the unit was incorrect

2.9.0 (2024-01-15)
-------------------

.. warning::
    pythermalcomfort 2.9.0 is no longer compatible with Python 3.8

* The PHS model accepts arrays as inputs

2.8.11 (2023-10-26)
-------------------

* wrote more test and improved code

2.8.11 (2023-10-26)
-------------------

* fixed issues with the documentation and sorted the models in alphabetical order

2.8.7 (2023-10-23)
-------------------

* Adaptive ASHRAE now returns a dataclass

2.8.6 (2023-10-09)
-------------------

* re-structured and linted the code

2.8.4 (2023-09-20)
-------------------

* calculation of cooling effect in pmv (standard='ashrae') triggered only when v>0.1 m/s

2.8.3 (2023-09-14)
-------------------

* general improvements in the JOS3 model

2.8.2 (2023-09-04)
-------------------

* general improvements in the JOS3 model
* fixed error when e_max == 0

2.8.1 (2023-07-05)
-------------------

* pythermalcomfort needs Python version > 3.8
* fixed issue in Cooling Effect calculation

2.8.0 (2023-07-03)
-------------------

* allowing the cooling effect to range from 0 to 40
* fixed PHS documentation
* improved JOS3 documentation

2.7.0 (2023-02-16)
-------------------

* changed coefficient of vasodilation in set_tmp() to 120 to match ASHRAE 55 2020 code
* slightly modified value in validation tables

2.6.0 (2023-01-17)
-------------------

* max sweating rate can be passed to two node model
* max skin wettedness can be passed to two node model
* rounding w to two decimals
* use_fans_heatwave function accepts arrays
* fixed typos unit documentation

2.5.4 (2022-10-12)
-------------------

* PHS model accepts all required inputs to be run on a minute by minute basis
* fix error check compliance PHS model

2.5.0 (2022-06-13)
-------------------

* Added the adaptive thermal heat balance (ATHB) model

2.4.0 (2022-06-10)
-------------------

* Added e_pmv model - Adjusted Predicted Mean Votes with Expectancy Factor
* Added a_pmv model - Adaptive Predicted Mean Vote

2.3.0 (2022-06-01)
-------------------

* Added discomfort index

2.2.0 (2022-05-17)
-------------------

* Implemented a better equation to calculate the mean radiant temperature

2.1.1 (2022-05-17)
-------------------

* Fixed how DISC is calculated

2.1.0 (2022-04-20)
-------------------

* Added Physiological Equivalent Temperature (PET) model
* In PMV and PPD function you can specify if occupants has control over airspeed

2.0.2 (2022-04-12)
-------------------

* UTCI accepts lists as inputs

2.0.0 (2022-04-07)
-------------------

.. warning::
    Version 2.0.0 introduces some breaking changes. Now the default behaviour of most of the function is that they return a ``np.nan`` if the inputs are outside the model applicability limits.

    For most functions we are no longer printing ``Warnings``. If you want the function to return a value even if your inputs are outside the model applicability limits then you can set the variable ``limit_input = False``. Please note that you should refrain from doing this.


.. note::
    Starting from Version 2.0.0 of pythermalcomfort now most of the functions (see detailed list below) accept Numpy arrays or lists as inputs. This allows you to write more concise and faster code since we optimized vectorization, where possible using Numba.

* Allowing users to pass Numpy arrays or lists as input to the pmv_ppd, pmv, clo_tout, both adaptive models, utci, set_tmp, two_nodes
* Changed the input variable from return_invalid to limit_input
* Increased speed by using Numba @vectorize decorator
* Changed ASHRAE 55 2020 limits to match new addenda
* Improved documentation

1.11.0 (2022-03-16)
-------------------

* Allowing users to pass a Numpy array as input into the UTCI function
* Numpy is now a requirement of pythermalcomfort
* Improved PMV, JOS-3, and UTCI documentation
* Testing PMV, SET, and solar gains models using online reference tables

1.10.0 (2021-11-15)
-------------------

* Added JOS-3 model

1.9.0 (2021-10-07)
------------------

* Added Normal Effective Temperature (NET)
* Added Apparent Temperature (AT)
* Added Wind Chill Index (WCI)

1.8.0 (2021-09-28)
------------------

* Gagge's two-node model
* Added WBGT equation
* Added Heat index (HI)
* Added humidex index

1.7.1 (2021-09-08)
------------------

* Added ASHRAE equation to calculate the operative temperature

1.7.0 (2021-07-29)
------------------

* Implemented function to calculate the if fans are beneficial during heatwaves
* Fixed error in the SET equation to calculated radiative heat transfer coefficient
* Fixed error in SET definition
* Moved functions optimized with Numba to new file

1.6.2 (2021-07-08)
------------------

* Updated equation clo_dynamic based on ANSI/ASHRAE Addendum f to ANSI/ASHRAE Standard 55-2020
* Fixed import errors in examples

1.6.1 (2021-07-05)
------------------

* optimized UTCI function with Numba

1.6.0 (2021-05-21)
------------------

* (BREAKING CHANGE) moved some of the functions from psychrometrics to utilities
* added equation to calculate body surface area

1.5.2 (2021-05-05)
------------------

* return stress category UTCI

1.5.1 (2021-04-29)
------------------

* optimized phs with Numba

1.5.0 (2021-04-21)
------------------

* added Predicted Heat Strain (PHS) index from ISO 7933:2004

1.4.6 (2021-03-30)
------------------

* changed equation to calculate convective heat transfer coefficient in set_tmp() as per Gagge's 1986
* fixed vasodilation coefficient in set_tmp()
* docs changed term air velocity with air speed and improved documentation
* added new tests for comfort functions

1.3.6 (2021-02-04)
------------------

* fixed error calculation solar_altitude and sharp for supine person in solar_gain

1.3.5 (2021-02-02)
------------------

* not rounding SET temperature when calculating cooling effect

1.3.3 (2020-12-14)
------------------

* added function to calculate sky-vault view fraction

1.3.2 (2020-12-14)
------------------

* replaced input solar_azimuth with sharp in the solar_gain() function
* fixed small error in example pmv calculation

1.3.1 (2020-10-30)
------------------

* Fixed error calculation of cooling effect with elevated air temperatures

1.3.0 (2020-10-19)
------------------

* Changed PMV elevated air speed limit from 0.2 to 0.1 m/s

1.2.3 (2020-09-09)
------------------

* Fixed error in the calculation of erf
* Updated validation table erf

1.2.2 (2020-08-21)
------------------

* Changed default diameter in mean_radiant_tmp
* Improved documentation


1.2.0 (2020-07-29)
------------------

* Significantly improved calculation speed using numba. Wrapped set and pmv functions

1.0.6 (2020-07-24)
------------------

* Minor speed improvement changed math.pow with **
* Added validation PMV validation table from ISO 7730

1.0.4 (2020-07-20)
------------------

* Improved speed calculation of the Cooling Effect
* Bisection has been replaced with Brentq function from scipy

1.0.3 (2020-07-01)
------------------

* Annotated variables in the SET code.

1.0.2 (2020-06-11)
------------------

* Fixed an error in the bisection equation used to calculated Cooling Effect.


1.0.0 (2020-06-09)
------------------

* Major stable release.

0.7.0 (2020-06-09)
------------------

* Added equation to calculate the dynamic clothing insulation

0.6.3 (2020-04-11)
------------------

* Fixed error in calculation adaptive ASHRAE
* Added some examples

0.6.3 (2020-03-17)
------------------

* Renamed function to_calc to t_o
* Fixed error calculation of relative air speed
* renamed input parameter ta to tdb
* Added function to calculate mean radiant temperature from black globe temperature
* Added function to calculate solar gain on people
* Added functions to calculate vapour pressure, wet-bulb temperature, dew point temperature, and psychrometric data from dry bulb temperature and RH
* Added authors
* Added dictionaries with reference clo and met values
* Added function to calculate enthalpy_air

0.5.2 (2020-03-11)
------------------

* Added function to calculate the running mean outdoor temperature

0.5.1 (2020-03-06)
------------------

* There was an error in version 0.4.2 in the calculation of PMV and PPD with elevated air speed, i.e. vr > 0.2 which has been fixed in this version
* Added function to calculate the cooling effect in accordance with ASHRAE

0.4.1 (2020-02-17)
------------------

* Removed compatibility with python 2.7 and 3.5

0.4.0 (2020-02-17)
------------------

* Created adaptive_EN, v_relative, t_clo, vertical_tmp_gradient, ankle_draft functions and wrote tests.
* Added possibility to decide with measuring system to use SI or IP.

0.3.0 (2020-02-13)
------------------

* Created set_tmp, adaptive_ashrae, UTCI functions and wrote tests.
* Added warning to let the user know if inputs entered do not comply with Standards applicability limits.

0.1.0 (2020-02-11)
------------------

* Created pmv, pmv_ppd functions and wrote tests.
* Documented code.

0.0.0 (2020-02-11)
------------------

* First release on PyPI.
