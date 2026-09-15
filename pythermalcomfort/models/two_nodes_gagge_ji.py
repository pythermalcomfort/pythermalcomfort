from __future__ import annotations

import math

import numpy as np

from pythermalcomfort.classes_input import GaggeTwoNodesJiInputs, NumericInput
from pythermalcomfort.classes_return import GaggeTwoNodesJi
from pythermalcomfort.utilities import Postures


def two_nodes_gagge_ji(
    tdb: NumericInput,
    tr: NumericInput,
    v: NumericInput,
    met: NumericInput,
    clo: NumericInput,
    vapor_pressure: NumericInput,
    wme: NumericInput = 0,
    body_surface_area: NumericInput = 1.8258,
    p_atm: NumericInput = 101325,
    position: str = Postures.sitting.value,
    **kwargs,
) -> GaggeTwoNodesJi:
    """Two-node model for older people under hot and cold exposures proposed by Ji et
    al. [Ji2022]_.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    tr : float or list of floats
        Mean radiant temperature, [°C].
    v : float or list of floats
        Air speed, [m/s].
    met : float or list of floats
        Metabolic rate, [met].
    clo : float or list of floats
        Clothing insulation, [clo].
    vapor_pressure : float or list of floats
        Vapor pressure, [torr].

        .. note::
            Vapor pressure can be calculated using the relative humidity and the saturation
            vapor pressure, which can be computed using the function
            :py:meth:`pythermalcomfort.psychrometrics.p_sat_torr`.

    wme : float or list of floats
        External work, [met]. Defaults to 0.
    body_surface_area : float or list of floats
        Body surface area, [m2]. Defaults to 1.8258.

        .. note::
            Body surface area can be calculated with weight and height using the function
            :py:meth:`pythermalcomfort.utilities.body_surface_area`.

    p_atm : float or list of floats
        Atmospheric pressure, [Pa]. Defaults to 101325.

    position : str, optional
        Select either "sitting" or "standing". Defaults to "sitting".

    Other Parameters
    ----------------
    body_weight : float, optional
        Body weight, [kg]. Defaults to 70 kg.
    length_time_simulation : int, optional
        Length of the simulation, [minutes]. Defaults to 120 minutes.
    initial_skin_temp : float, optional
        Initial skin temperature, [°C]. Defaults to 36.8 °C.
    initial_core_temp : float, optional
        Initial core temperature, [°C]. Defaults to 36.49 °C.
    acclimatized : bool, optional
        If True, the model assumes the person is acclimatized to heat. Defaults to True.

    Returns
    -------
    GaggeTwoNodesJi
        A dataclass containing the simulated core and skin temperatures over time. See :py:class:`~pythermalcomfort.classes_return.GaggeTwoNodesJi` for more details.
        To access the individual attributes, use the corresonding attrubute of the returned
        instance, e.g. `result.t_core` or `result.t_skin`.

    Example
    -------
    .. code-block:: python

        from pythermalcomfort.models import two_nodes_gagge_ji
        from pythermalcomfort.utilities import body_surface_area
        from pythermalcomfort.psychrometrics import p_sat_torr

        rh = 20
        vapor_pressure = rh * p_sat_torr(tdb=36.5) / 100

        result = two_nodes_gagge_ji(
            tdb=36.5,
            tr=36.5,
            v=0.25,
            met=0.95,
            clo=0.1,
            vapor_pressure=vapor_pressure,
            wme=0,
            body_surface_area=body_surface_area(weight=80.1, height=1.8),
            p_atm=101325,
            position="sitting",
            acclimatized=True,
            body_weight=80.1,
            length_time_simulation=120,
        )
    """
    body_weight = kwargs.pop("body_weight", 70)
    length_time_simulation = kwargs.pop("length_time_simulation", 120)
    initial_skin_temp = kwargs.pop("initial_skin_temp", 36.8)
    initial_core_temp = kwargs.pop("initial_core_temp", 36.49)
    acclimatized = kwargs.pop("acclimatized", True)

    if kwargs:
        error_msg = f"Unexpected keyword arguments: {list(kwargs.keys())}"
        raise TypeError(error_msg)

    tdb = np.asarray(tdb)
    tr = np.asarray(tr)
    v = np.asarray(v)
    met = np.asarray(met)
    clo = np.asarray(clo)
    vapor_pressure = np.asarray(vapor_pressure)
    wme = np.asarray(wme)
    body_surface_area = np.asarray(body_surface_area)
    p_atm = np.asarray(p_atm)

    # Validate inputs
    GaggeTwoNodesJiInputs(
        tdb=tdb,
        tr=tr,
        v=v,
        met=met,
        clo=clo,
        vapor_pressure=vapor_pressure,
        wme=wme,
        body_surface_area=body_surface_area,
        p_atm=p_atm,
        position=position,
    )

    excluded_params = {
        "position",
        "acclimatized",
        "body_weight",
        "length_time_simulation",
        "initial_skin_temp",
        "initial_core_temp",
    }

    results_array_of_dicts = np.vectorize(
        _two_nodes_ji_optimized,
        excluded=excluded_params,
        otypes=[object],
    )(
        tdb=tdb,
        tr=tr,
        v=v,
        met=met,
        clo=clo,
        vapor_pressure=vapor_pressure,
        wme=wme,
        body_surface_area=body_surface_area,
        p_atm=p_atm,
        position=position,
        acclimatized=acclimatized,
        body_weight=body_weight,
        length_time_simulation=length_time_simulation,
        initial_skin_temp=initial_skin_temp,
        initial_core_temp=initial_core_temp,
    )

    output_data = {}
    if results_array_of_dicts.ndim == 0:
        # scalar inputs
        output_data = results_array_of_dicts.item()
    elif results_array_of_dicts.size == 0:
        # empty input arrays
        output_data = {"t_core": [], "t_skin": []}
    else:
        # multiple simulations were run with array inputs
        # t_core and t_skin are arrays of arrays, each array corresponds to a simulation
        first_result_dict = results_array_of_dicts[0]
        for key in first_result_dict.keys():
            output_data[key] = []

        for res_dict in results_array_of_dicts:
            for key, value in res_dict.items():
                # creates a list of NumPy arrays for each key
                output_data[key].append(value)

    return GaggeTwoNodesJi(**output_data)


def _two_nodes_ji_optimized(
    tdb,
    tr,
    v,
    met,
    clo,
    vapor_pressure,
    wme,
    body_surface_area,
    p_atm,
    position,
    acclimatized,
    body_weight,
    length_time_simulation,
    initial_skin_temp,
    initial_core_temp,
):
    # Initial variables as defined in the ASHRAE 55-2020
    air_speed = max(v, 0.1)
    met_factor = 58.2  # met conversion factor
    sbc = 0.000000056697  # Stefan-Boltzmann constant (W/m2K4)

    # These values are taken from Ji, 2022
    c_sw = 170  # driving coefficient for regulatory sweating
    c_dil = 50  # driving coefficient for vasodilation
    c_str = 0.75  # driving coefficient for vasoconstriction
    a_cof = 0.2  # coefficient in sweat rate

    # Max and min values defined by Ji, 2022
    min_skin_blood_flow = 0.75  # min SBF for older people
    max_skin_blood_flow = 63  # max SBF for older people
    max_sweating_rate_factor = 0.9  # 90% sweating efficiency
    evap_sweating_reg_max = 400  # W/m2

    # Other coefficients defined by Ji, 2022
    c_de = 0.6  # attenuation coefficient of vasodilation
    c_ce = 0.5  # attenuation coefficient of vasoconstriction
    c_swe = 1  # sweat attenuation coefficient
    c_she = 1  # shivering attenuation coefficient
    cof_scs = 19.4  # added shivering coefficient
    cof_sc = 50  # added shivering coefficient
    cof_ss = 0.5  # added shivering coefficient

    # Neutral/trigger values for different systems (in older people)
    # Taken from Ji, 2022
    t_cr0_dil = 37.3  # vasodilation threshold
    t_sk0_cons = 33.25  # vasoconstriction threshold
    t_cr0_sw = 37  # core sweating threshold
    t_sk0_sw = 34.3  # skin sweating threshold
    t_cr0_sh = 36.7  # shivering threshold

    alfa = 0.1
    skin_blood_flow_neutral = 6.3  # as defined by Ji, 2021

    t_skin = initial_skin_temp
    t_core = initial_core_temp
    m_bl = skin_blood_flow_neutral

    # initialize some variables
    e_skin = 0.1 * met  # total evaporative heat loss, W
    w = 0.06  # skin wettedness

    pressure_in_atmospheres = p_atm / 101325
    n_simulation = 0

    r_clo = 0.155 * clo  # thermal resistance of clothing, K*m2/W
    # increase in body surface area due to clothing
    f_a_cl = 1.0 + 0.2 * clo if clo < 0.5 else 1.05 + 0.1 * clo
    lr = 2.2 / pressure_in_atmospheres  # Lewis relation
    m = met * met_factor  # metabolic rate in W/m2

    i_cl = 1.0  # permeation efficiency of water vapor naked skin
    if clo > 0:
        i_cl = 0.45  # permeation efficiency of water vapor through the clothing layer

    # Set w_max and the max sweating rate, taken from emails with Dr Laouadi
    w_max = 0.85
    if acclimatized:
        evap_sweating_reg_max = 1.25 * evap_sweating_reg_max
        w_max = 1
    m_rsw_max = evap_sweating_reg_max / 0.68
    m_rsw_max = m_rsw_max * max_sweating_rate_factor

    if not w_max:  # if the user did not pass a value of w_max
        # W_max should be fixed to either 0.85 or 1 (for acclimatized people);  See ISO PHS model and other related standard
        w_max = 1

    # Heat transfer coefficients
    h_cc = 3  # initial value - convective heat transfer coefficient
    h_r = 4.7  # initial value - linearized radiative heat transfer coefficient

    skin_temp_hist = []
    core_temp_hist = []

    while n_simulation < length_time_simulation:
        n_simulation += 1

        iteration_limit = 150  # for following while loop

        h_t = (
            h_r + h_cc
        )  # sum of convective and radiant heat transfer coefficient W/(m2*K)
        r_a = 1.0 / (f_a_cl * h_t)  # resistance of air layer to dry heat
        t_op = (h_r * tr + h_cc * tdb) / h_t  # operative temperature

        # t_cl temperature of the outer surface of clothing
        t_cl = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo)  # initial guess
        n_iterations = 0
        tc_converged = False

        while not tc_converged:
            if position == Postures.sitting.value:
                # 0.7 ratio between radiation area of the body and the body area
                h_r = 4.0 * 0.97 * sbc * ((t_cl + tr) / 2.0 + 273.15) ** 3.0 * 0.7
            else:  # if standing
                # 0.77 ratio between radiation area of the body and the body area
                h_r = 4.0 * 0.97 * sbc * ((t_cl + tr) / 2.0 + 273.15) ** 3.0 * 0.77
            h_t = h_r + h_cc
            r_a = 1.0 / (f_a_cl * h_t)
            t_op = (h_r * tr + h_cc * tdb) / h_t
            t_cl_new = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo)
            if abs(t_cl_new - t_cl) < 0.01:
                tc_converged = True
            t_cl = t_cl_new
            n_iterations += 1

            if n_iterations > iteration_limit:
                raise StopIteration("Max iterations exceeded")

        # Convective heat transfer coefficient based on clothing
        d_tcl_air = t_cl - tdb  # difference between clothing and air temperature
        if air_speed < 0.2:
            # Free convection. Gao et al 2019: Formulation of human body heat transfer coefficient...
            if d_tcl_air > 0:
                # Upward flow
                h_cc = 2.5 * d_tcl_air**0.16
            else:
                # Downward flow
                h_cc = 2.5 * math.fabs(d_tcl_air) ** 0.41
        else:
            # Forced convection
            # Original formula from SET code
            h_cc = 8.6 * air_speed**0.53

        q_sensible = (t_skin - t_op) / (r_a + r_clo)  # total sensible heat loss, W

        # respiration
        q_res = (
            0.0023 * m * (44.0 - vapor_pressure)
        )  # latent heat loss due to respiration
        c_res = 0.0014 * m * (34.0 - tdb)  # sensible convective heat loss respiration

        # hf_cs rate of energy transport between core and skin, W
        # 5.28 is the average body tissue conductance in W/(m2 C)
        # 1.163 is the thermal capacity of blood in Wh/(L C)
        hf_cs = (t_core - t_skin) * (5.28 + 1.163 * m_bl)
        s_core = m - hf_cs - q_res - c_res - wme  # rate of energy storage in the core
        s_skin = hf_cs - q_sensible - e_skin  # rate of energy storage in the skin
        tc_sk = 0.97 * alfa * body_weight  # thermal capacity skin
        tc_cr = 0.97 * (1 - alfa) * body_weight  # thermal capacity core
        d_t_sk = (s_skin * body_surface_area) / (
            tc_sk * 60.0
        )  # rate of change skin temperature °C per minute
        d_t_cr = (s_core * body_surface_area) / (
            tc_cr * 60.0
        )  # rate of change core temperature °C per minute
        t_skin = t_skin + d_t_sk
        t_core = t_core + d_t_cr

        #
        # Main equations used in Ji, 2022
        #

        # Calculate skin blood flow (SBF)
        t_cr_dil = max(0, t_core - t_cr0_dil)  # dilation trigger
        t_sk_cons = max(0, t_sk0_cons - t_skin)  # constriction trigger
        m_bl = (skin_blood_flow_neutral + c_de * c_dil * t_cr_dil) / (
            1 + c_ce * c_str * t_sk_cons
        )  # sbf
        # Set min/max skin blood flow values
        m_bl = min(max_skin_blood_flow, m_bl)
        m_bl = max(min_skin_blood_flow, m_bl)

        # Calculate sweat triggers
        t_sk_sw = max(0, t_skin - t_sk0_sw)
        t_cr_sw = max(0, t_core - t_cr0_sw)

        # Update alfa (for next cycle)
        alfa = 0.0417737 + 0.7451832 / (m_bl + 0.5854417)

        # Sweat rate (SWR) - accounting for the metabolic rate
        m_rsw = (
            c_swe
            * c_sw
            * ((1 - alfa) * t_cr_sw + (alfa + a_cof) * t_sk_sw)
            * math.exp(t_sk_sw / 10.7)
        )
        # Apply max sweating rate
        m_rsw = min(m_rsw, m_rsw_max)
        # Energy lost via evaporation
        e_rsw = 0.68 * m_rsw  # heat lost by vaporization sweat

        r_e_cl = r_clo / (lr * i_cl)  # evaporative resistance of clothing in K*m2*atm/W
        r_e_a = 1 / (
            lr * f_a_cl * h_cc
        )  # evaporative resistance of air layer in K*m2*atm/W
        r_total = r_e_cl + r_e_a  # total evaporative resistance in K*m2*atm/W

        e_max = (
            math.exp(18.6686 - 4030.183 / (t_skin + 235.0)) - vapor_pressure
        ) / r_total
        p_rsw = e_rsw / e_max  # ratio heat loss sweating to max heat loss sweating

        # calculate first the sweat efficiency using the skin WETNESS (assuming all sweat is converted to evaporative heat).
        # Note skin wetness is different than skin WETTEDNESS which assumes the correct evaporation efficiency
        he_n = (
            1 / r_total
        )  # coefficient of the evaporative heat transfer (inverse of the total evaporative heat resistance; W/(K*m2*Pa))
        wettedness_dif = 1 / (2 + 2.46 * he_n)
        wp = (
            wettedness_dif + (1 - wettedness_dif) * p_rsw
        )  # skin wetness (between 0.06 and 1)

        # Calculate skin wettedness, by taking the ISO PHS evaporation efficiency formula: eff = 1 - 0.5 * w**2
        w = (math.sqrt(2 * wp**2 + 1) - 1) / wp

        # Limit w
        # print("Skin wetness exceeds max value ${:.2f}".format(w_max))
        w = min(w, w_max)

        # Recalculate evaporative sweat heat
        p_rsw = (w - wettedness_dif) / (1.0 - wettedness_dif)
        e_rsw = max(0, p_rsw * e_max)
        e_diff = max(0, w * e_max - e_rsw)

        # check condensation on skin surface due to high RH > 100%; body emerged in water;  the model is not valid for this case
        if e_max < 0:
            # sweating is suppressed, and condensation latent heat is not taken into account
            w = 0.0
            e_diff = 0.0
            e_rsw = 0.0

        # Total evaporative heat loss by sweating and vapor diffusion
        e_skin = e_rsw + e_diff

        # Calculate Shivering (SHIV)
        t_cr_sh = max(0, t_cr0_sh - t_core)
        met_shivering = c_she * (
            cof_scs * t_cr_sh * t_sk_cons + cof_sc * t_cr_sh + cof_ss * t_sk_cons
        )
        m = met * met_factor + met_shivering

        # Append skin and core temp for time point
        skin_temp_hist.append(t_skin)
        core_temp_hist.append(t_core)

    return {"t_core": np.asarray(core_temp_hist), "t_skin": np.asarray(skin_temp_hist)}
