from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy import optimize

from pythermalcomfort.classes_input import NumericInput, PETSteadyInputs
from pythermalcomfort.classes_return import PETSteady
from pythermalcomfort.psychrometrics import p_sat
from pythermalcomfort.utilities import Postures, Sex, body_surface_area


def pet_steady(
    tdb: NumericInput,
    tr: NumericInput,
    v: NumericInput,
    rh: NumericInput,
    met: NumericInput,
    clo: NumericInput,
    p_atm: NumericInput = 1013.25,
    position: str | list[str] = Postures.sitting.value,
    age: NumericInput = 23,
    sex: str | np.ndarray | list = Sex.male.value,
    weight: NumericInput = 75,
    height: NumericInput = 1.8,
    wme: NumericInput = 0,
) -> PETSteady:
    """Calculate the steady physiological equivalent temperature (PET) using the Munich
    Energy-balance Model for Individuals (MEMI) to simulate the human body's thermal
    state in a medically realistic manner.

    PET is defined as the air temperature
    at which, in a typical indoor setting the heat budget of the human body is balanced
    with the same core and skin temperature as under the complex outdoor conditions to be
    assessed [Hoppe1999]_.
    The following assumptions are made for the indoor reference climate: tdb = tr, v = 0.1
    m/s, water vapour pressure = 12 hPa, clo = 0.9 clo, and met = 1.37 met + basic
    metabolism.
    PET allows a layperson to compare the total effects of complex thermal circumstances
    outside with his or her own personal experience indoors in this way. This function
    solves the heat balances without accounting for heat storage in the human body.

    The PET was originally proposed by Hoppe [Hoppe1999]_. In 2018, Walther and Goestchel [Walther2018]_
    proposed a correction of the original model, purging the errors in the
    PET calculation routine, and implementing a state-of-the-art vapour diffusion model.
    Walther and Goestchel (2018) model is therefore used to calculate the PET.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    tr : float or list of floats
        Mean radiant temperature, [°C].
    v : float or list of floats
        Wind speed, [m/s].
    rh : float or list of floats
        Relative humidity, [%].
    met : float or list of floats
        Metabolic rate, [met].
    clo : float or list of floats
        Clothing insulation, [clo].
    p_atm : float or list of floats, optional
        Atmospheric pressure, [hPa]. Defaults to 1013.25.
    position : str or list of str, optional
        Position of the person "sitting", "standing", "standing, forced convection". Defaults to "sitting".
    age : int or list of ints, optional
        Age of the person. Defaults to 23.
    sex : str or list of str, optional
        Sex of the person "male" or "female". Defaults to "male"
    weight : float or list of floats, optional
        Weight of the person, [kg]. Defaults to 75.
    height : float or list of floats, optional
        Height of the person, [m]. Defaults to 1.8.
    wme : float or list of floats, optional
        External work, [met]. Defaults to 0.

    Returns
    -------
    PETSteady
        A dataclass containing the Physiological Equivalent Temperature. See :py:class:`~pythermalcomfort.classes_return.PETSteady` for more details.
        To access the `pet` value, use the `pet` attribute of the returned `PETSteady` instance, e.g., `result.pet`.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models import pet_steady

        result = pet_steady(tdb=25, tr=25, v=0.1, rh=50, met=1.2, clo=0.5)
        print(result.pet)  # 24.67

        result = pet_steady(
            tdb=[25, 30],
            tr=[25, 30],
            v=[0.1, 0.2],
            rh=[50, 60],
            met=[1.2, 1.4],
            clo=[0.5, 0.6],
        )
        print(result.pet)  # [24.67 31.46]
    """
    # Validate inputs using the PETSteadyInputs class
    PETSteadyInputs(
        tdb=tdb,
        tr=tr,
        v=v,
        rh=rh,
        met=met,
        clo=clo,
        p_atm=p_atm,
        position=position,
        age=age,
        sex=sex,
        weight=weight,
        height=height,
        wme=wme,
    )

    return PETSteady(
        pet=_pet_steady_vectorised(
            tdb=tdb,
            tr=tr,
            v=v,
            rh=rh,
            met=met,
            clo=clo,
            p_atm=p_atm,
            position=position,
            age=age,
            sex=sex,
            weight=weight,
            height=height,
            wme=wme,
        ),
    )


@np.vectorize
def _pet_steady_vectorised(
    tdb,
    tr,
    v,
    rh,
    met,
    clo,
    p_atm,
    position,
    age,
    sex,
    weight,
    height,
    wme,
) -> npt.ArrayLike:
    met_factor = 58.2  # met conversion factor
    met = met * met_factor  # metabolic rate

    def vasomotricity(t_cr, t_sk):
        """Vasomotricity (blood flow) in function of the core and skin temperatures.

        Parameters
        ----------
        t_cr : float
            The body core temperature, [°C]
        t_sk : float
            The body skin temperature, [°C]

        Returns
        -------
        dict
            "m_blood": Blood flow rate, [kg/m2/h] and "alpha": repartition of body
            mass
            between core and skin [].
        """
        # skin and core temperatures set values
        tc_set = 36.6  # 36.8
        tsk_set = 34  # 33.7
        # Set value signals
        sig_skin = tsk_set - t_sk
        sig_core = t_cr - tc_set
        if sig_core < 0:
            # In this case, T_core<Tc_set --> the blood flow is reduced
            sig_core = 0.0
        if sig_skin < 0:
            # In this case, Tsk>Tsk_set --> the blood flow is increased
            sig_skin = 0.0
        # 6.3 L/m^2/h is the set value of the blood flow
        m_blood = (6.3 + 75.0 * sig_core) / (1.0 + 0.5 * sig_skin)
        # 90 L/m^2/h is the blood flow upper limit
        if m_blood > 90:
            m_blood = 90.0
        # in other models, alpha is used to update tbody
        alpha = 0.0417737 + 0.7451833 / (m_blood + 0.585417)

        return {"m_blood": m_blood, "alpha": alpha}

    def sweat_rate(t_body):
        """Sweating mechanism depending on the body and core temperatures.

        Parameters
        ----------
        t_body : float
            weighted average between skin and core temperatures, [°C]

        Returns
        -------
        m_rsw : float
            The sweating flow rate, [g/m2/h].
        """
        tc_set = 36.6  # 36.8
        tsk_set = 34  # 33.7
        tbody_set = 0.1 * tsk_set + 0.9 * tc_set  # Calculation of the body
        # temperature
        # through a weighted average
        sig_body = t_body - tbody_set
        if sig_body < 0:
            # In this case, Tbody<Tbody_set --> The sweat flow is 0
            sig_body = 0.0
        # from Gagge's model
        m_rsw = 304.94 * sig_body
        # 500 g/m^2/h is the upper sweat rate limit
        m_rsw = min(m_rsw, 500)

        return m_rsw

    def solve_pet(
        t_arr,
        _tdb,
        _tr,
        _v=0.1,
        _rh=50,
        _met=80,
        _clo=0.9,
        actual_environment=False,
    ):
        """Solve PET by either computing the vectorial balance of the three unknown
        temperatures (T_core, T_sk, T_clo) or by finding the environment operative
        temperature that yields the same energy balance as the actual environment.

        Parameters
        ----------
        t_arr : list or list of floats
            [T_core, T_sk, T_clo], [°C]
        _tdb : float
            dry bulb air temperature, [°C]
        _tr : float
            mean radiant temperature, [°C]
        _v : float, default 0.1 m/s for the reference environment
            air speed, [m/s]
        _rh : float, default 50 % for the reference environment
            relative humidity, [%]
        _met : float, default 80 W for the reference environment
            metabolic rate, [W/m2]
        _clo : float, default 0.9 clo for the reference environment
            clothing insulation, [clo]
        actual_environment : boolean
            True=solve 3eqs/3unknowns, False=solve for PET

        Returns
        -------
        float or list
            PET or energy balance.
        """
        e_skin = 0.99  # Skin emissivity
        e_clo = 0.95  # Clothing emissivity
        h_vap = 2.42 * 10**6  # Latent heat of evaporation [J/Kg]
        sbc = 5.67 * 10**-8  # Stefan-Boltzmann constant [W/(m2*K^(-4))]
        cb = 3640  # Blood specific heat [J/kg/k]

        t_cr, t_sk, t_clo = t_arr
        e_bal_vec = [
            0.0,
            0.0,
            0.0,
        ]  # required for the vectorial expression of the balance
        # Area parameters of the body:
        a_dubois = body_surface_area(weight, height)
        # Base metabolism for men and women in [W]
        met_female = (
            3.19
            * weight**0.75
            * (
                1.0
                + 0.004 * (30.0 - age)
                + 0.018 * (height * 100.0 / weight ** (1.0 / 3.0) - 42.1)
            )
        )
        met_male = (
            3.45
            * weight**0.75
            * (
                1.0
                + 0.004 * (30.0 - age)
                + 0.01 * (height * 100.0 / weight ** (1.0 / 3.0) - 43.4)
            )
        )
        # Attribution of internal energy depending on the sex of the subject
        met_correction = met_male if sex == Sex.male.value else met_female

        # Source term : metabolic activity
        he = (_met + met_correction) / a_dubois
        # impact of efficiency
        h = he * (1.0 - wme)  # [W/m2]

        # correction for wind
        i_m = 0.38  # Woodcock ratio for vapour transfer through clothing [-]

        # Calculation of the Burton surface increase coefficient, k = 0.31 for Hoeppe:
        fcl = (
            1 + 0.31 * _clo
        )  # Increase heat exchange surface depending on clothing level
        f_a_cl = (173.51 * _clo - 2.36 - 100.76 * _clo * _clo + 19.28 * _clo**3.0) / 100
        a_clo = a_dubois * f_a_cl + a_dubois * (fcl - 1.0)  # clothed body surface area

        f_eff = (
            0.696 if position == Postures.standing.value else 0.725
        )  # effective radiation factor

        a_r_eff = (
            a_dubois * f_eff
        )  # Effective radiative area depending on the position of the subject

        # Partial pressure of water in the air
        vpa = _rh / 100.0 * p_sat(_tdb) / 100  # [hPa]
        if not actual_environment:  # mode=False means we are calculating the PET
            vpa = 12  # [hPa] vapour pressure of the standard environment

        # Convection coefficient depending on wind velocity and subject position
        hc = 2.67 + 6.5 * _v**0.67  # sitting
        if position == Postures.standing.value:  # standing
            hc = 2.26 + 7.42 * _v**0.67
        if position == "standing, forced convection":  # standing, forced convection
            hc = 8.6 * _v**0.513
        # h_cc corrected convective heat transfer coefficient
        h_cc = 3.0 * pow(p_atm / 1013.25, 0.53)
        hc = max(h_cc, hc)
        # modification of hc with the total pressure
        hc = hc * (p_atm / 1013.25) ** 0.55

        # Respiratory energy losses
        t_exp = 0.47 * _tdb + 21.0  # Expired air temperature calculation [degC]
        d_vent_pulm = he * 1.44 * 10.0 ** (-6.0)  # breathing flow rate
        c_res = 1010 * (_tdb - t_exp) * d_vent_pulm  # Sensible heat energy loss [W/m2]
        vpexp = p_sat(t_exp) / 100  # Latent heat energy loss [hPa]
        q_res = 0.623 * h_vap / p_atm * (vpa - vpexp) * d_vent_pulm  # [W/m2]
        ere = c_res + q_res  # [W/m2]

        # Calculation of the equivalent thermal resistance of body tissues
        alpha = vasomotricity(t_cr, t_sk)["alpha"]
        tbody = alpha * t_sk + (1 - alpha) * t_cr

        # Clothed fraction of the body approximation
        r_cl = _clo / 6.45  # Conversion in [m2.K/W]
        y = 0
        f_a_cl = min(f_a_cl, 1.0)
        if _clo >= 2.0:
            y = 1.0
        if 0.6 < _clo < 2.0:
            y = (height - 0.2) / height
        if 0.6 >= _clo > 0.3:
            y = 0.5
        if 0.3 >= _clo > 0.0:
            y = 0.1
        # calculation of the clothing radius depending on the clothing level (6.28 = 2*
        # pi !)
        r2 = a_dubois * (fcl - 1.0 + f_a_cl) / (6.28 * height * y)  # External radius
        r1 = f_a_cl * a_dubois / (6.28 * height * y)  # Internal radius
        di = r2 - r1
        # Calculation of the equivalent thermal resistance of body tissues
        htcl = 6.28 * height * y * di / (r_cl * np.log(r2 / r1) * a_clo)  # [W/(m2.K)]

        # Calculation of sweat losses
        qmsw = sweat_rate(tbody)
        # h_vap/1000 = 2400 000[J/kg] divided by 1000 = [J/g] // qwsw/3600 for [g/m2/h]
        # to [
        # g/m2/s]
        esw = h_vap / 1000 * qmsw / 3600  # [W/m2]
        # Saturation vapor pressure at temperature Tsk
        p_v_sk = p_sat(t_sk) / 100  # hPa
        # Calculation of vapour transfer
        lr = 16.7 * 10 ** (-1)  # [K/hPa] Lewis ratio
        he_diff = hc * lr  # diffusion coefficient of air layer
        fecl = 1 / (1 + 0.92 * hc * r_cl)  # Burton efficiency factor
        e_max = he_diff * fecl * (p_v_sk - vpa)  # maximum diffusion at skin surface
        if e_max == 0:  # added this otherwise e_req / e_max cannot be calculated
            e_max = 0.001
        w = esw / e_max  # skin wettedness
        if w > 1:
            w = 1
            delta = esw - e_max
            if delta < 0:
                esw = e_max
        esw = max(esw, 0)
        # i_m= Woodcock's ratio (see above)
        r_ecl = (1 / (fcl * hc) + r_cl) / (
            lr * i_m
        )  # clothing vapour transfer resistance after Woodcock's method
        ediff = (1 - w) * (p_v_sk - vpa) / r_ecl  # diffusion heat transfer
        evap = -(ediff + esw)  # [W/m2]

        # Radiation losses bare skin
        r_bare = (
            a_r_eff
            * (1.0 - f_a_cl)
            * e_skin
            * sbc
            * ((_tr + 273.15) ** 4.0 - (t_sk + 273.15) ** 4.0)
            / a_dubois
        )
        # ... for clothed area
        r_clo = (
            f_eff
            * a_clo
            * e_clo
            * sbc
            * ((_tr + 273.15) ** 4.0 - (t_clo + 273.15) ** 4.0)
            / a_dubois
        )
        r_sum = r_clo + r_bare  # radiation total

        # Convection losses for bare skin
        c_bare = hc * (_tdb - t_sk) * a_dubois * (1.0 - f_a_cl) / a_dubois  # [W/m^2]
        # ... for clothed area
        c_clo = hc * (_tdb - t_clo) * a_clo / a_dubois  # [W/m^2]
        csum = c_clo + c_bare  # convection total

        # Balance equations of the 3-nodes model
        e_bal_vec[0] = (
            h
            + ere
            - (vasomotricity(t_cr, t_sk)["m_blood"] / 3600 * cb + 5.28) * (t_cr - t_sk)
        )  # Core balance [W/m^2]
        e_bal_vec[1] = (
            r_bare
            + c_bare
            + evap
            + (vasomotricity(t_cr, t_sk)["m_blood"] / 3600 * cb + 5.28) * (t_cr - t_sk)
            - htcl * (t_sk - t_clo)
        )  # Skin balance [W/m^2]
        e_bal_vec[2] = c_clo + r_clo + htcl * (t_sk - t_clo)  # Clothes balance [W/m^2]
        e_bal_scal = h + ere + r_sum + csum + evap

        # returning either the calculated core, skin, clo temperatures or the PET
        if actual_environment:
            # if we solve for the system we need to return 3 temperatures
            return list(e_bal_vec)
        # solving for the PET requires the scalar balance only
        return e_bal_scal

    def pet_fc(_t_stable):
        """Find the solution.

        Parameters
        ----------
        _t_stable : list or list of floats
            3 temperatures obtained from the actual environment (T_core,T_skin,T_clo).

        Returns
        -------
        float
            The PET comfort index.
        """

        # Definition of a function with the input variables of the PET reference situation
        def f(tx):
            return solve_pet(
                _t_stable,
                _tdb=tx,
                _tr=tx,
                actual_environment=False,
            )

        # solving for PET
        pet_guess = _t_stable[2]  # start with the clothing temperature

        return round(optimize.fsolve(f, pet_guess)[0], 2)

    # initial guess
    t_guess = np.asarray([36.7, 34, 0.5 * (tdb + tr)])
    # solve for Tc, Tsk, Tcl temperatures
    t_stable = optimize.fsolve(
        solve_pet,
        t_guess,
        args=(
            tdb,
            tr,
            v,
            rh,
            met,
            clo,
            True,
        ),
    )
    # compute PET
    return pet_fc(t_stable)
