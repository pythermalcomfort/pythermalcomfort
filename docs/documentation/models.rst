Models
======

This section of the documentation provides detailed documentation on the various thermal comfort models, heat and cold indexes implemented in the `pythermalcomfort` package.
These models are used to assess and predict thermal comfort, heat stress, physiological variables and more as a function of both environmental and personal parameters.
Each model is accompanied by detailed descriptions, usage examples, and references to relevant standards and research.
This documentation aims to provide a comprehensive understanding of how to use these models to evaluate thermal comfort in various scenarios.
Please note that models are shown in alphabetical order and are not sorted based on their accuracy.

Adaptive Model
--------------

The adaptive thermal comfort model is a method that relates indoor design temperatures or acceptable temperature ranges to outdoor meteorological or climatological parameters.
It's specifically intended for **occupant-controlled naturally conditioned spaces**, where the thermal conditions are primarily regulated by occupants through the use of openings in the building envelope, such as windows.
The adaptive model is based on the idea that people in naturally ventilated spaces adjust to their environment through a variety of behavioural and physiological adaptations.

Below are some key characteristics and criteria of the adaptive model, according to the ASHRAE 55-2023 standard [55ASHRAE2023]_:

**Applicability**: The adaptive model can only be applied in spaces that meet specific criteria:
  *   There is no mechanical cooling or heating system in operation.
  *   Occupants have metabolic rates ranging from 1.0 to 1.5 met.
  *   Occupants are able to adjust their clothing to indoor or outdoor thermal conditions within a range of 0.5 to 1.0 clo.
  *   The prevailing mean outdoor temperature is between 10°C (50°F) and 33.5°C (92.3°F).
  *   The space has operable fenestration that can be readily opened and adjusted by the occupants.

**Methodology**: The model uses the prevailing mean outdoor air temperature to determine acceptable indoor operative temperatures. The prevailing mean outdoor temperature is calculated as a running average of the mean daily outdoor temperatures over a period of days.
  *   The calculation of the prevailing mean outdoor temperature gives more weight to recent days, since these have a greater influence on occupants' comfort temperatures.

**Comfort Zone**: The adaptive model defines comfort zones differently from the PMV model. Instead of using a heat balance approach to determine an ideal temperature, it relies on an empirical model that links satisfaction with the prevailing mean outdoor temperature.
  *   The comfort zone is defined by the 80% or 90% acceptability limits.
  *   The model does not require estimation of clothing values, since it accounts for clothing adaptation by relating the range of satisfactory indoor temperatures to the outdoor climate.

**Air Speed**: When using the adaptive model, no humidity or air speed limits are required. However, if elevated air movement is present, air movement extensions to the comfort zone's lower and upper operative temperature limits can be used.

**Underlying Basis**: The adaptive model is derived from a global database of measurements taken primarily in office buildings. It is based on the observation that people adapt to their environment over time.

**Purpose**: The adaptive model is specifically intended for use in naturally conditioned spaces, where occupants have a degree of control over their environment. It is not appropriate for other types of spaces.

The adaptive model, in contrast to the PMV model, does not rely on a heat-balance approach. Instead, it uses an empirical model that relates satisfaction to the prevailing mean outdoor air temperature. The comfort zone is defined by the percentage of people who find the environment acceptable. The adaptive model is based on the idea that people adjust to their environment over time.

The use of the adaptive model requires documentation of the compliance time periods, the prevailing mean outdoor design temperature, and any increased air speed adjustments.


ASHRAE 55
~~~~~~~~~

.. autofunction:: pythermalcomfort.models.adaptive_ashrae.adaptive_ashrae

.. autoclass:: pythermalcomfort.classes_return.AdaptiveASHRAE
    :members:

EN 16798-1 2019
~~~~~~~~~~~~~~~

.. autofunction:: pythermalcomfort.models.adaptive_en.adaptive_en

.. autoclass:: pythermalcomfort.classes_return.AdaptiveEN
    :members:

Apparent Temperature (AT)
-------------------------

.. autofunction:: pythermalcomfort.models.at.at

.. autoclass:: pythermalcomfort.classes_return.AT
    :members:

Ankle draft
-----------

.. autofunction:: pythermalcomfort.models.ankle_draft.ankle_draft

.. autoclass:: pythermalcomfort.classes_return.AnkleDraft
    :members:

Clothing prediction
-------------------

.. autofunction:: pythermalcomfort.models.clo_tout.clo_tout

.. autoclass:: pythermalcomfort.classes_return.CloTOut
    :members:

Cooling Effect (CE)
-------------------

.. autofunction:: pythermalcomfort.models.cooling_effect.cooling_effect

.. autoclass:: pythermalcomfort.classes_return.CE
    :members:

Discomfort Index (DI)
---------------------

.. autofunction:: pythermalcomfort.models.discomfort_index.discomfort_index

.. autoclass:: pythermalcomfort.classes_return.DI
    :members:

Environmental Stress Index (ESI)
--------------------------------
.. autofunction:: pythermalcomfort.models.esi.esi

.. autoclass:: pythermalcomfort.classes_return.ESI
    :members:

Gagge two-node model
--------------------

.. autofunction:: pythermalcomfort.models.two_nodes_gagge.two_nodes_gagge

.. autoclass:: pythermalcomfort.classes_return.GaggeTwoNodes
    :members:

Two-node model sleep
--------------------

.. autofunction:: pythermalcomfort.models.two_nodes_gagge_sleep.two_nodes_gagge_sleep

.. autoclass:: pythermalcomfort.classes_return.GaggeTwoNodesSleep
    :members:

Gagge-Ji two-node model
-----------------------

.. autofunction:: pythermalcomfort.models.two_nodes_gagge_ji.two_nodes_gagge_ji

.. autoclass:: pythermalcomfort.classes_return.GaggeTwoNodesJi
    :members:

Heat Index (HI)
---------------

The Heat Index (HI) is a commonly used metric to estimate apparent temperature (AT) incorporating the effects of humidity based on Steadman’s model [Steadman1979]_ of human thermoregulation.

Lu and Romps (2022) [lu]_ found that Steadman’s model produces unrealistic results under extreme conditions, such as excessively hot and humid or cold and dry environments, rendering the heat index undefined.
For instance, at 80% relative humidity, the heat index is only valid within a temperature range of 288–304 K.
To address this issue, Lu and Romps (2022) [lu]_ developed a new model that extends the range of validity of the heat index.

pythermalcomfort therefore includes two equations to calculate the Heat Index. One in accordance with the new Lu and Romps (2022) model which is an extension of the first version of Steadman’s (1979) apparent temperature :py:class:`~pythermalcomfort.models.heat_index_lu.heat_index_lu`.
The other is developed by Rothfusz (1990) and it is a simplified model derived by multiple regression analysis in temperature and relative humidity from the first version of Steadman’s (1979) apparent temperature (AT) [Rothfusz1990]_ :py:class:`~pythermalcomfort.models.heat_index_rothfusz.heat_index_rothfusz`.

Heat Index (HI) Lu and Romps (2022)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pythermalcomfort.models.heat_index_lu.heat_index_lu

.. autoclass:: pythermalcomfort.classes_return.HI
    :members:

Heat Index (HI) Rothfusz (1990)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pythermalcomfort.models.heat_index_rothfusz.heat_index_rothfusz

.. autoclass:: pythermalcomfort.classes_return.HI
    :noindex:
    :members:

Heat Index (HI) Schoen (2005)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pythermalcomfort.models.heat_index_schoen.heat_index_schoen

.. autoclass:: pythermalcomfort.classes_return.HI
    :noindex:
    :members:

Humidex
-------

.. autofunction:: pythermalcomfort.models.humidex.humidex

.. autoclass:: pythermalcomfort.classes_return.Humidex
    :members:

Required Clothing Insulation (IREQ) and Duration Limited Exposure (DLE)
-----------------------------------------------------------------------

.. autofunction:: pythermalcomfort.models.ireq.ireq

.. autoclass:: pythermalcomfort.classes_return.IREQ
    :members:

Joint system thermoregulation model (JOS-3)
-------------------------------------------

.. autoclass:: pythermalcomfort.models.jos3.JOS3
    :members:
    :undoc-members:
    :special-members: __init__
    :exclude-members: tdb, tr, to, rh, v, posture, clo, par, t_body, bsa, r_t, r_et, w, w_mean, t_skin_mean, t_skin, t_core, t_cb, t_artery, t_vein, t_superficial_vein, t_muscle, t_fat, body_names, bmr

Normal Effective Temperature (NET)
----------------------------------

.. autofunction:: pythermalcomfort.models.net.net

.. autoclass:: pythermalcomfort.classes_return.NET
    :members:

Predicted Heat Strain (PHS) Index
---------------------------------

.. autofunction:: pythermalcomfort.models.phs.phs

.. autoclass:: pythermalcomfort.classes_return.PHS
    :members:

Physiological Equivalent Temperature (PET)
------------------------------------------

.. autofunction:: pythermalcomfort.models.pet_steady.pet_steady

.. autoclass:: pythermalcomfort.classes_return.PETSteady
    :members:

Predicted Mean Vote (PMV) and Predicted Percentage of Dissatisfied (PPD)
------------------------------------------------------------------------

The Predicted Mean Vote (PMV) is an index that aims to predict the mean value of thermal sensation votes from a large group of people, based on a seven-point scale ranging from "cold" (-3) to "hot" (+3).
It was developed by Fanger [Fanger1970]_.

The PMV is designed to predict the average thermal sensation of a large group of people exposed to the same environment [7730ISO2005]_ [7730ISO2025]_.
It calculates the heat balance of a typical occupant and relates their thermal gains or losses to their predicted mean thermal sensation [55ASHRAE2023]_.

The PMV can be used to check if a thermal environment meets comfort criteria and to establish requirements for different levels of acceptability.
The PMV model is applicable to healthy men and women exposed to indoor environments where thermal comfort is desirable, but moderate deviations from thermal comfort occur, in the design of new environments or the assessment of existing ones.
The PMV is intended to be used for moderate thermal environments.

The PMV calculation considers several factors:
  *   **Dry-bulb air temperature** (`T` :sub:`db`).
  *   **Mean radiant temperature** (`T` :sub:`r`).
  *   **Relative humidity** (RH).
  *   **Relative air velocity** (`V` :sub:`r`), the relative air speed caused by body movement and not the air speed measured by the air speed sensor
  *   **Metabolic rate** (M), which is the rate of heat production in the body. Metabolic rate data is available in the ASHRAE Handbook—Fundamentals and users should use their judgement to match activities to comparable activities in the table.
  *   **Clothing insulation** (`I` :sub:`cl,r`), dynamic intrinsic insulation, this is the thermal insulation from the skin surface to the outer clothing surface, including enclosed air layers, under the environmental conditions.

The PMV model is applicable when the six main parameters are within specific intervals.
These values are specified by the ASHRAE 55 [55ASHRAE2023]_ and ISO 7730 standards [7730ISO2005]_ [7730ISO2025]_.
The ISO also states that the PMV model is only applicable for PMV between -2 and +2 [7730ISO2005]_ [7730ISO2025]_.

There are several formulations of the PMV model that have been developed over the years.
The two most commonly used are the original PMV model [Fanger1970]_ and the ASHRAE 55 PMV model [55ASHRAE2023]_.
Tartarini and Schiavon (2025) [Tartarini2025PMV]_ compared the accuracy of the PMV models implemented in the ISO 7730:2005 and ASHRAE 55:2023 standards and found that the ISO 7730:2005 model has a higher accuracy than the ASHRAE 55:2023 model.
However, it should be noted that both PMV models have low accuracy in predicting thermal sensation votes, especially outside thermal neutrality [Tartarini2025PMV]_.
For this reason, it is recommended that the use of the PMV be restricted to values between -0.5 and +0.5 [Tartarini2025PMV]_, where an environment may be deemed thermally neutral by a large group of occupants.

The Predicted Percentage Dissatisfied (PPD) is an index that provides a **quantitative prediction of the percentage of people likely to feel too warm or too cool in a given environment**.
The PPD is derived from the PMV.
Specifically, **the PPD predicts the number of thermally dissatisfied persons among a large group of people,** where thermally dissatisfied people are those who would vote hot, warm, cool, or cold on a seven-point thermal sensation scale.


PMV formulations
~~~~~~~~~~~~~~~~
After Fanger developed the original PMV model which it is still included in the ISO 7330 in its original form, several other PMV formulations have been proposed.
These include but are not limited to:

  *  the aPMV model [Yao2009]_ :py:meth:`pythermalcomfort.models.pmv_a.pmv_a`,
  *  the ASHRAE 55 PMV model [55ASHRAE2023]_ :py:meth:`pythermalcomfort.models.pmv_ppd_ashrae.pmv_ppd_ashrae`,
  *  the ATHB model [Schweiker2022]_ :py:meth:`pythermalcomfort.models.pmv_athb.pmv_athb`,
  *  the ePMV model [Fanger2002]_ :py:meth:`pythermalcomfort.models.pmv_e.pmv_e`.


ISO 7730 - PMV and PPD
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pythermalcomfort.models.pmv_ppd_iso.pmv_ppd_iso

.. autoclass:: pythermalcomfort.classes_return.PMVPPD
    :members:


ASHRAE 55 - PMV and PPD
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pythermalcomfort.models.pmv_ppd_ashrae.pmv_ppd_ashrae

.. autoclass:: pythermalcomfort.classes_return.PMVPPDAshrae
    :members:

Adaptive Predicted Mean Vote (aPMV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pythermalcomfort.models.pmv_a.pmv_a

.. autoclass:: pythermalcomfort.classes_return.APMV
    :members:

Adaptive Thermal Heat Balance (ATHB)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pythermalcomfort.models.pmv_athb.pmv_athb

.. autoclass:: pythermalcomfort.classes_return.ATHB
    :members:

Adjusted Predicted Mean Votes with Expectancy Factor (ePMV)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: pythermalcomfort.models.pmv_e.pmv_e

.. autoclass:: pythermalcomfort.classes_return.EPMV
    :members:

Ridge Regression to predict Rectal and Skin Temperatures
--------------------------------------------------------

.. autofunction:: pythermalcomfort.models.ridge_regression_predict_t_re_t_sk.ridge_regression_predict_t_re_t_sk

.. autoclass:: pythermalcomfort.classes_return.PredictedBodyTemperatures
    :members:

Solar gain on people
--------------------

.. autofunction:: pythermalcomfort.models.solar_gain.solar_gain

.. autoclass:: pythermalcomfort.classes_return.SolarGain
    :members:

Sports Heat Stress Risk
------------------------

.. autofunction:: pythermalcomfort.models.sports_heat_stress_risk.sports_heat_stress_risk

.. autoclass:: pythermalcomfort.models.sports_heat_stress_risk.Sports
    :members:

.. autoclass:: pythermalcomfort.classes_return.SportsHeatStressRisk
    :members:

Standard Effective Temperature (SET)
------------------------------------

.. autofunction:: pythermalcomfort.models.set_tmp.set_tmp

.. autoclass:: pythermalcomfort.classes_return.SET
    :members:

Universal Thermal Climate Index (UTCI)
--------------------------------------

.. autofunction:: pythermalcomfort.models.utci.utci

.. autoclass:: pythermalcomfort.classes_return.UTCI
    :members:

Use Fans During Heatwaves
-------------------------

.. autofunction:: pythermalcomfort.models.use_fans_heatwaves.use_fans_heatwaves

.. autoclass:: pythermalcomfort.classes_return.UseFansHeatwaves
    :members:

Vertical air temperature gradient
---------------------------------

.. autofunction:: pythermalcomfort.models.vertical_tmp_grad_ppd.vertical_tmp_grad_ppd

.. autoclass:: pythermalcomfort.classes_return.VerticalTGradPPD
    :members:

Wet Bulb Globe Temperature Index (WBGT)
---------------------------------------

.. autofunction:: pythermalcomfort.models.wbgt.wbgt

.. autoclass:: pythermalcomfort.classes_return.WBGT
    :members:

Liljegren outdoor Wet Bulb Globe Temperature (WBGT)
------------------------------------------------------------

.. autofunction:: pythermalcomfort.models.wbgt_liljegren.wbgt_liljegren

.. autoclass:: pythermalcomfort.classes_return.WBGTLiljegren
    :members:

Wind chill index
----------------

.. autofunction:: pythermalcomfort.models.wci.wci

.. autoclass:: pythermalcomfort.classes_return.WCI
    :members:

Wind chill temperature
----------------------

.. autofunction:: pythermalcomfort.models.wind_chill_temperature.wind_chill_temperature

.. autoclass:: pythermalcomfort.classes_return.WCT
    :members:

Work capacity (Dunne et al.)
-----------------------------

.. autofunction:: pythermalcomfort.models.work_capacity_dunne.work_capacity_dunne

.. autoclass:: pythermalcomfort.classes_return.WorkCapacity
    :members:

Work capacity (Hothaps)
-----------------------

.. autofunction:: pythermalcomfort.models.work_capacity_hothaps.work_capacity_hothaps

.. autoclass:: pythermalcomfort.classes_return.WorkCapacity
    :noindex:
    :members:

Work capacity (ISO)
-------------------

.. autofunction:: pythermalcomfort.models.work_capacity_iso.work_capacity_iso

.. autoclass:: pythermalcomfort.classes_return.WorkCapacity
    :noindex:
    :members:

Work capacity (NIOSH)
---------------------

.. autofunction:: pythermalcomfort.models.work_capacity_niosh.work_capacity_niosh

.. autoclass:: pythermalcomfort.classes_return.WorkCapacity
    :noindex:
    :members:

Temperature Humidity Index (THI)
--------------------------------

.. autofunction:: pythermalcomfort.models.thi.thi

.. autoclass:: pythermalcomfort.classes_return.THI
    :members:
