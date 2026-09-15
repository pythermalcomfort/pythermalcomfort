Clothing
========

Introduction
------------

There are several key variables related to the thermal insulation and water vapour resistance of clothing ensembles.
The standard, BS EN ISO 9920:2009, defines them.
These variables are crucial for assessing thermal stress and comfort in different environments.

**Thermal Insulation (I)**
    This represents the resistance to dry heat loss between two surfaces and is expressed in square metres Kelvin per watt (**m²⋅K⋅W⁻¹**).
    It's the temperature gradient divided by the heat loss per unit of body surface area. Thermal insulation is often expressed in 'clo' units, where 1 clo = 0.155 m²⋅K⋅W⁻¹.

    *   **Total Insulation** (I\ :sub:`T`): This is the **thermal insulation from the body surface to the environment**, encompassing all clothing layers, enclosed air layers, and the boundary air layer, under static reference conditions.
    *   **Basic Insulation** (I\ :sub:`cl`): Also known as intrinsic insulation, this is the **thermal insulation from the skin surface to the outer clothing surface**, including enclosed air layers, under static reference conditions. **This is the value used as input for example in the PMV model**.
    *   **Air Insulation** (I\ :sub:`a`): This is the **thermal insulation of the boundary air layer** around the outer clothing or, when nude, around the skin surface. This can be influenced by air and body movement, and also can be expressed as a combination of convective and radiative heat transfer coefficients. It can be calculated using :py:meth:`pythermalcomfort.clothing.clo_insulation_air_layer`
    *   **Resultant Total Insulation** (I\ :sub:`T,r`): This is the **actual thermal insulation from the body surface to the environment**, considering all clothing, enclosed air layers, and boundary air layers under given environmental conditions and activities. It accounts for the effects of movements and wind.
    *   **Effective Thermal Insulation** (I\ :sub:`clu`): This term is used for individual garments. It's determined on a manikin wearing only a single garment.

**Water Vapour Resistance** (R\ :sub:`e`)
    This represents the resistance to water vapour transfer between two surfaces and is expressed in square metres kilopascal per watt (**m²⋅kPa⋅W⁻¹**). It's the vapour pressure gradient divided by the evaporative heat loss per unit of body surface area.

    *   **Basic Water Vapour Resistance** (R\ :sub:`e,cl`): This is the **water vapour resistance from the skin surface to the outer clothing surface** under reference conditions. The standard also defines resultant or dynamic basic water vapour resistance to account for the impact of body and air movement on this variable.
    *   **Total Water Vapour Resistance** (R\ :sub:`e,T`): The **water vapour resistance of a clothing ensemble**.

**Clothing Area Factor** (f\ :sub:`cl`)
    This factor accounts for the increase in surface area due to clothing.
    It is used in calculations of the heat transfer between the body and the environment and is related to the thermal insulation of the clothing.
    It can be calculated by using :py:meth:`pythermalcomfort.clothing.clo_area_factor`

**Permeability Index** (i\ :sub:`m`)
    This index is related to the permeability of fabric layers and is used in estimating water vapour resistance. It is not directly related to insulation, but to the fabric's ability to allow vapour to pass through. For an air layer, im is around 0.5. For many types of permeable clothing, it may be set to 0.38.

These variables are essential for estimating the thermal characteristics of clothing ensembles and for evaluating thermal stress using standardized methods.

Determining Estimated Clothing Insulation
-----------------------------------------

The EN ISO 9920:2009 standard provides several methods for estimating clothing insulation when direct measurement isn't feasible.
These methods include using pre-measured tables, summing individual garment values, or applying empirical equations.
It's important to note that these are estimates, and measurements from manikins and human subjects are more accurate.

Methods for Estimating Clothing Insulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using Tables of Complete Ensembles:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  *   The standard provides tables (Annex A) with insulation values (I\ :sub:`T` and I\ :sub:`cl`) for various clothing ensembles.
  *   These values are measured using a **standing thermal manikin in low air movement** (< 0.2 m/s) conditions.
  *   The tables also list the **clothing area factor** (f\ :sub:`cl`) for each ensemble.
  *   To use this method, **match your ensemble to the closest one in the table** for the most accurate estimate.
  *   **Interpolation** between table values can provide estimates for ensembles not exactly listed.
  *   Small corrections can be made by adding or subtracting insulation values for individual garments to the values for a closely matching ensemble.
  *   **Remember to correct these values for movement and air velocity** when applying to real-world situations.

Summation of Individual Garment Insulations:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  *   When a complete ensemble isn't available, you can estimate the ensemble insulation (I\ :sub:`cl`) by summing the effective thermal insulation (I\ :sub:`clu`) of each garment by using :py:meth:`pythermalcomfort.clothing.clo_intrinsic_insulation_ensemble`
  *   This method assumes **uniform insulation distribution** and may be inaccurate for unevenly layered clothing. **Using full ensembles from the tables is preferable**.

The standard also provides other methods which are not listed here.

Influence of Body and Air Movement on Thermal Insulation and Vapour Resistance
------------------------------------------------------------------------------

**Body movement and air movement significantly reduce both the thermal insulation and water vapour resistance** of clothing ensembles.
This reduction is primarily due to a "pumping effect" where air is exchanged with the environment via openings, and also by compression of clothing and air penetration through fabrics.

Correction of clothing insulation for body movement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To correct static clothing insulation values for the effects of air and body movement, the ISO 9920 provides correction equations based on the total static insulation value, (I\ :sub:`T`), to obtain the resultant total clothing insulation, (I\ :sub:`T,r`).
These equations take into account **air velocity relative to the person** (v\ :sub:`r`, from 0.15 to 3.5 m/s) and **walking speed** (v\ :sub:`w`, from 0 to 1.2 m/s).

The correction equations are comprised within this function :py:meth:`pythermalcomfort.clothing.clo_total_insulation`

Other Factors Influencing Clothing Insulation
---------------------------------------------

Posture
~~~~~~~

*   Changes in posture, like sitting, impact the body's heat exchange surface.
*   **Sitting generally increases air layer insulation** (\ :sub:`Ia`) due to air pockets forming around the knees and hips.
*   However, sitting **reduces clothing insulation** (\ :sub:`Icl`) because clothing on the back, thighs, and buttocks is compressed.
*   Typically, \ :sub:`Icl` **decreases by 6% to 18%** when a person sits, while \ :sub:`Ia` **increases by 10% to 25%**.
*   The overall effect depends on the balance between clothing and air insulation. For example: for a nude person, the total insulation typically increases by about 10% when sitting. For a person wearing thick clothing, the total insulation typically decreases by about 10% when sitting.
*   These effects don't account for the influence of the seat itself.

Effect of Seats
~~~~~~~~~~~~~~~

*   The type of seat can either add to or reduce a person's insulation.
*   A standard car seat can increase insulation by approximately 0.25 clo; however, ventilation within the car seat can decrease insulation.
*   Office chairs can increase insulation by 0.04 clo to 0.17 clo, depending on the backrest and seat thickness.
*   A sofa adds around 0.21 clo.
*   **Net chairs and wooden stools** can **decrease insulation by about 0.03 clo**.

Effect of Pressure
~~~~~~~~~~~~~~~~~~

*   Changes in air pressure, such as a decrease when at altitude, affect both dry and evaporative heat transfer.
*   **Lower air pressure reduces convective heat transfer**, leading to an increase in dry insulation.
*   **Low pressure enhances evaporative heat transfer**, which causes greater heat loss.
*   The net effect depends on the balance between these processes and the level of sweating.

Wetting
~~~~~~~

*   When clothing gets wet, it **loses part of its insulation**.
*   Moisture increases the material’s conductivity, thus decreasing insulation.
*   Moisture in clothing also results in **additional evaporation and increased heat loss**.

Washing
~~~~~~~

*   **Washing can alter the thermal insulation properties** of clothing.
*   The extent of the effect varies with the textile type but is usually within the limits of measurement accuracy. Insulation may increase due to fibre contraction in woven or knitted garments. However, insulation mostly decreases because of reduced thickness.
*   **Cold protective clothing with polyester batting tends to decrease in thickness and insulation** after washing.

Functions
=========

Air insulation layer (I\ :sub:`a`)
----------------------------------

.. autofunction:: pythermalcomfort.clothing.clo_insulation_air_layer

Correction factor for (I\ :sub:`T`)
-----------------------------------

.. autofunction:: pythermalcomfort.clothing.clo_correction_factor_environment

Clothing area factor (f\ :sub:`cl`)
-----------------------------------

.. autofunction:: pythermalcomfort.clothing.clo_area_factor

Dynamic clothing
----------------

Below are the two functions to calculate the dynamic clothing in accordance with ISO and ASHRAE.

.. autofunction:: pythermalcomfort.clothing.clo_dynamic_iso

.. autofunction:: pythermalcomfort.clothing.clo_dynamic_ashrae

Intrinsic clothing insulation ensemble (I\ :sub:`cl`)
-----------------------------------------------------

.. autofunction:: pythermalcomfort.clothing.clo_intrinsic_insulation_ensemble

Total insulation of the clothing ensemble (I\ :sub:`T`)
-------------------------------------------------------

.. autofunction:: pythermalcomfort.clothing.clo_total_insulation

Clothing insulation of typical ensembles, [clo]
-----------------------------------------------

.. autodata:: pythermalcomfort.utilities.clo_typical_ensembles

**Example**

.. code-block:: python

    from pythermalcomfort.utilities import clo_typical_ensembles

    print(clo_typical_ensembles["Typical summer indoor clothing"])
    # Output: 0.5

Insulation of individual garments, [clo]
----------------------------------------

.. autodata:: pythermalcomfort.utilities.clo_individual_garments

**Example**

.. code-block:: python

    from pythermalcomfort.utilities import clo_individual_garments

    print(clo_individual_garments["T-shirt"])
    # 0.08

    # calculate total clothing insulation
    i_cl = (
        clo_individual_garments["T-shirt"]
        + clo_individual_garments["Men's underwear"]
        + clo_individual_garments["Thin trousers"]
        + clo_individual_garments["Shoes or sandals"]
    )
    print(i_cl)
    # 0.29
