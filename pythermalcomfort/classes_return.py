from __future__ import annotations

import datetime as dt
import textwrap
from dataclasses import dataclass, fields, is_dataclass

import numpy as np
import numpy.typing as npt
from numpy._typing import NDArray


class AutoStrMixin:
    def __str__(self) -> str:
        """Pretty-print dataclass fields as aligned single-line 'name : value'.

        - One line per field, no header/title.
        - NumPy arrays with >10 elements show first 5 and last 5.
        - Other long strings are shortened to 80 chars with an ellipsis.
        """
        if not is_dataclass(self):
            return super().__str__()

        MAX_STR_WIDTH = 80
        names = [f.name for f in fields(type(self))]
        values = [getattr(self, n) for n in names]

        # Format values as strings, using comma separation for arrays/lists
        def value_to_str(val):
            if isinstance(val, np.ndarray):
                v = np.array2string(
                    val,
                    separator=", ",
                    max_line_width=MAX_STR_WIDTH,
                    precision=2,
                    threshold=10,
                    edgeitems=5,
                    formatter={"float_kind": lambda x: f"{x:.2f}"},
                )
            elif isinstance(val, (list | tuple)):
                if len(val) > 10:
                    head = ", ".join(str(x) for x in val[:5])
                    tail = ", ".join(str(x) for x in val[-5:])
                    v = f"[{head}, ..., {tail}]"
                else:
                    v = f"[{', '.join(str(x) for x in val)}]"
            else:
                v = str(val)
            # Shorten if needed
            if isinstance(val, str) and len(v) > MAX_STR_WIDTH:
                v = textwrap.shorten(v, width=MAX_STR_WIDTH, placeholder="...")
            return v

        value_strs = [value_to_str(v) for v in values]
        # Find max name length for alignment
        max_name_len = max(len(n) for n in names) if names else 0
        # Compose lines with aligned colons
        lines = [
            f"{n.ljust(max_name_len)} : {v}"
            for n, v in zip(names, value_strs, strict=True)
        ]
        # Shorten each line to MAX_STR_WIDTH
        lines = [
            textwrap.shorten(line, width=MAX_STR_WIDTH, placeholder="...")
            if len(line) > MAX_STR_WIDTH
            else line
            for line in lines
        ]

        # Compute header length
        header_base = f"{self.__class__.__name__}"
        max_line_len = max([len(header_base)] + [len(line) for line in lines])
        header = f"{'-' * (max_line_len)}"
        title = f"{header_base}".center(max_line_len)
        result = [header, title, header] + lines
        return "\n".join(result)

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, item: str):
        try:
            return getattr(self, item)
        except AttributeError as exc:
            error_msg = (
                f"{self.__class__.__name__} has no field '{item}'. "
                f"Available fields: {[f.name for f in fields(type(self))]}"
            )
            raise KeyError(error_msg) from exc


@dataclass(frozen=True, repr=False)
class APMV(AutoStrMixin):
    """A dataclass to store the results of the adaptive Predicted Mean Vote (aPMV)
    model.

    Attributes
    ----------
    a_pmv : float or list of floats
        Predicted Mean Vote.
    """

    a_pmv: float | list[float]


@dataclass(frozen=True, repr=False)
class AdaptiveASHRAE(AutoStrMixin):
    """A dataclass to store the results of the adaptive thermal comfort model based on
    ASHRAE 55.

    Attributes
    ----------
    tmp_cmf : float or list of floats
        Comfort temperature at a specific running mean temperature, default in [°C] or [°F].
    tmp_cmf_80_low : float or list of floats
        Lower acceptable comfort temperature for 80% occupants, default in [°C] or [°F].
    tmp_cmf_80_up : float or list of floats
        Upper acceptable comfort temperature for 80% occupants, default in [°C] or [°F].
    tmp_cmf_90_low : float or list of floats
        Lower acceptable comfort temperature for 90% occupants, default in [°C] or [°F].
    tmp_cmf_90_up : float or list of floats
        Upper acceptable comfort temperature for 90% occupants, default in [°C] or [°F].
    acceptability_80 : bool or list of bools
        Acceptability for 80% occupants.
    acceptability_90 : bool or list of bools
        Acceptability for 90% occupants.
    """

    tmp_cmf: float | list[float]
    tmp_cmf_80_low: float | list[float]
    tmp_cmf_80_up: float | list[float]
    tmp_cmf_90_low: float | list[float]
    tmp_cmf_90_up: float | list[float]
    acceptability_80: bool | list[bool]
    acceptability_90: bool | list[bool]


@dataclass
class AdaptiveEN(AutoStrMixin):
    """Dataclass to store the results of the adaptive thermal comfort calculation based
    on EN 16798-1 2019.

    Attributes
    ----------
    tmp_cmf : float or list of floats
        Comfort temperature at that specific running mean temperature, default in [°C] or in [°F].
    acceptability_cat_i : bool or list of bools
        If the indoor conditions comply with comfort category I.
    acceptability_cat_ii : bool or list of bools
        If the indoor conditions comply with comfort category II.
    acceptability_cat_iii : bool or list of bools
        If the indoor conditions comply with comfort category III.
    tmp_cmf_cat_i_up : float or list of floats
        Upper acceptable comfort temperature for category I, default in [°C] or in [°F].
    tmp_cmf_cat_ii_up : float or list of floats
        Upper acceptable comfort temperature for category II, default in [°C] or in [°F].
    tmp_cmf_cat_iii_up : float or list of floats
        Upper acceptable comfort temperature for category III, default in [°C] or in [°F].
    tmp_cmf_cat_i_low : float or list of floats
        Lower acceptable comfort temperature for category I, default in [°C] or in [°F].
    tmp_cmf_cat_ii_low : float or list of floats
        Lower acceptable comfort temperature for category II, default in [°C] or in [°F].
    tmp_cmf_cat_iii_low : float or list of floats
        Lower acceptable comfort temperature for category III, default in [°C] or in [°F].
    """

    tmp_cmf: float | list[float]
    acceptability_cat_i: bool | list[bool]
    acceptability_cat_ii: bool | list[bool]
    acceptability_cat_iii: bool | list[bool]
    tmp_cmf_cat_i_up: float | list[float]
    tmp_cmf_cat_ii_up: float | list[float]
    tmp_cmf_cat_iii_up: float | list[float]
    tmp_cmf_cat_i_low: float | list[float]
    tmp_cmf_cat_ii_low: float | list[float]
    tmp_cmf_cat_iii_low: float | list[float]


@dataclass(frozen=True, repr=False)
class AnkleDraft(AutoStrMixin):
    """Dataclass to store the results of the ankle draft calculation.

    Attributes
    ----------
    ppd_ad : float or list of floats
        Predicted Percentage of Dissatisfied occupants with ankle draft, [%].
    acceptability : bool or list of bools
        Indicates if the air speed at the ankle level is acceptable according to ASHRAE 55 2020 standard.
    """

    ppd_ad: float | list[float]
    acceptability: bool | list[bool]


@dataclass(frozen=True, repr=False)
class AT(AutoStrMixin):
    """Dataclass to store the results of the Apparent Temperature (AT) calculation.

    Attributes
    ----------
    at : float or list of floats
        Apparent temperature, [°C]
    """

    at: float


@dataclass(frozen=True, repr=False)
class ATHB(AutoStrMixin):
    """Dataclass to store the results of the Adaptive Thermal Heat Balance (ATHB)
    calculation.

    Attributes
    ----------
    athb_pmv : float or list of floats
        Predicted Mean Vote calculated with the Adaptive Thermal Heat Balance framework.
    """

    athb_pmv: float | list[float]


@dataclass(frozen=True, repr=False)
class CloTOut(AutoStrMixin):
    """Dataclass to represent the clothing insulation Icl as a function of outdoor air
    temperature.

    Attributes
    ----------
    clo_tout : float or list of floats
        Representative clothing insulation Icl.
    """

    clo_tout: float | list[float]


@dataclass(frozen=True, repr=False)
class CE(AutoStrMixin):
    """Dataclass to represent the Cooling Effect (CE).

    Attributes
    ----------
    ce : float or list of floats
        Cooling Effect value.
    """

    ce: float | list[float]


@dataclass(frozen=True, repr=False)
class DI(AutoStrMixin):
    """Dataclass to represent the Discomfort Index (DI) and its classification.

    Attributes
    ----------
    di : float or list of floats
        Discomfort Index, [°C].
    discomfort_condition : str or list of str
        Classification of the thermal comfort conditions according to the discomfort index.
    """

    di: float | list[float]
    discomfort_condition: str | list[str]


@dataclass(frozen=True, repr=False)
class EPMV(AutoStrMixin):
    """Dataclass to represent the Adjusted Predicted Mean Votes with Expectancy Factor
    (ePMV).

    Attributes
    ----------
    e_pmv : float or list of floats
        Adjusted Predicted Mean Votes with Expectancy Factor.
    """

    e_pmv: float | list[float]


@dataclass(frozen=True, repr=False)
class ESI(AutoStrMixin):
    """Dataclass to represent the Environmental Stress Index (ESI).

    Attributes
    ----------
    esi : float or list of floats
        Environmental Stress Index.
    """

    esi: float | list[float]


@dataclass(frozen=True, repr=False)
class HI(AutoStrMixin):
    """Dataclass to represent the Heat Index (HI).

    Attributes
    ----------
    hi : float or list of floats
        Heat Index, [°C] or [°F] depending on the units.
    """

    hi: npt.ArrayLike
    stress_category: str | list[str] | None = None


@dataclass(frozen=True, repr=False)
class Humidex(AutoStrMixin):
    """Dataclass to represent the Humidex and its discomfort category.

    Attributes
    ----------
    humidex : float or list of floats
        Humidex value, [°C].
    discomfort : str or list of str
        Degree of comfort or discomfort as defined in Havenith and Fiala (2016).
    """

    humidex: float | list[float]
    discomfort: str | list[str]

@dataclass(frozen=True, repr=False)
class MRT(AutoStrMixin):
    """Dataclass to represent the Mean Radiant Temperature (MRT).

    Attributes
    ----------
    mrt : float or list of floats
        Mean Radiant Temperature, [°C] or [°F].
    """

    mrt: float | list[float]

@dataclass(frozen=True, repr=False)
class SkyTemperature(AutoStrMixin):
    """Dataclass to represent the Mean Radiant Temperature (MRT).

    Attributes
    ----------
    t_sky : float or list of floats
        Temperature sky, [°C] or [°F].
    """

    t_sky: float | list[float]

@dataclass(frozen=True, repr=False)
class SkyEmissivity(AutoStrMixin):
    """Dataclass to represent the Mean Radiant Temperature (MRT).

    Attributes
    ----------
    eps_sky : xxx
    """

    eps_sky: float | list[float]


@dataclass(frozen=True, repr=False)
class NET(AutoStrMixin):
    """Dataclass to represent the Normal Effective Temperature (NET).

    Attributes
    ----------
    net : float or list of floats
        Normal Effective Temperature, [°C].
    """

    net: float | list[float]


@dataclass(frozen=True, repr=False)
class PETSteady(AutoStrMixin):
    """Dataclass to represent the Physiological Equivalent Temperature (PET).

    Attributes
    ----------
    pet : float or list of floats
        Physiological Equivalent Temperature.
    """

    pet: float | list[float]


@dataclass(frozen=True, repr=False)
class PHS(AutoStrMixin):
    """Dataclass to represent the Predicted Heat Strain (PHS).

    Attributes
    ----------
    t_re : float or list of floats
        Rectal temperature, [°C].
    t_sk : float or list of floats
        Skin temperature, [°C].
    t_cr : float or list of floats
        Core temperature, [°C].
    t_cr_eq : float or list of floats
        Core temperature as a function of the metabolic rate, [°C].
    t_sk_t_cr_wg : float or list of floats
        Fraction of the body mass at the skin temperature.
    d_lim_loss_50 : float or list of floats
        Maximum allowable exposure time limited by cumulative sweat mass loss
        for a mean worker (dehydration constraint), [minutes].
    d_lim_loss_95 : float or list of floats
        Maximum allowable exposure time limited by cumulative sweat mass loss
        for 95% of the working population (dehydration constraint), [minutes].
    d_lim_t_re : float or list of floats
        Maximum allowable exposure time for heat storage, [minutes].
    sweat_loss_g : float or list of floats
        Cumulative evaporated sweat mass for the whole person over the simulated duration, [grams (g)].
        This is a total mass per person (not per unit area).
    sweat_rate_watt : float or list of floats
        Instantaneous evaporative heat flux due to regulatory sweating at the skin,
        per unit area, [W·m⁻²]. Used at each simulation time step.
    evap_load_wm2_min : float or list of floats
        Accumulated evaporative load per unit area over the simulated duration, [W·min·m⁻²].
        Computed as the running sum of the instantaneous evaporative heat flux (W·m⁻²)
        at each 1 minute step. Intended for chaining simulation segments.
    """

    t_re: float | list[float]
    t_sk: float | list[float]
    t_cr: float | list[float]
    t_cr_eq: float | list[float]
    t_sk_t_cr_wg: float | list[float]
    d_lim_loss_50: float | list[float]
    d_lim_loss_95: float | list[float]
    d_lim_t_re: float | list[float]
    sweat_loss_g: float | list[float]
    sweat_rate_watt: float | list[float]
    evap_load_wm2_min: float | list[float]


@dataclass(frozen=True, repr=False)
class PMV(AutoStrMixin):
    """Dataclass to represent the Predicted Mean Vote (PMV).

    Attributes
    ----------
    pmv : float or list of floats
        Predicted Mean Vote.
    """

    pmv: float | list[float]


@dataclass(frozen=True, repr=False)
class Eps_Sky(AutoStrMixin):
    """Dataclass to represent the Sky Emissivity value.

    Attributes
    ----------
    eps_sky : float or list of floats
        Sky Emissivity

    """

    eps_sky: float | list[float]


@dataclass(frozen=True, repr=False)
class PMVPPD(AutoStrMixin):
    """Dataclass to represent the Predicted Mean Vote (PMV) and Predicted Percentage of
    Dissatisfied (PPD).

    Attributes
    ----------
    pmv : float or list of floats
        Predicted Mean Vote.
    ppd : float or list of floats
        Predicted Percentage of Dissatisfied.
    tsv : str or list of strings
        Predicted thermal sensation vote.
    compliance : bool or list of bools, optional
        True if PMV is within the acceptable range (-0.5 < PMV < 0.5) according to
        ASHRAE Standard 55-2023. Only returned by pmv_ppd_ashrae function.
    """

    pmv: float | list[float]
    ppd: float | list[float]
    tsv: float | list[float]
    compliance: bool | list[bool] | None = None


@dataclass(frozen=True, repr=False)
class PsychrometricValues(AutoStrMixin):
    p_sat: NDArray[np.float64]
    p_vap: NDArray[np.float64]
    hr: NDArray[np.float64]
    wet_bulb_tmp: NDArray[np.float64]
    dew_point_tmp: NDArray[np.float64]
    h: NDArray[np.float64]


@dataclass(frozen=True, repr=False)
class SET(AutoStrMixin):
    """Dataclass to represent the Standard Effective Temperature (SET).

    Attributes
    ----------
    set : float or list of floats
        Standard effective temperature, [°C].
    """

    set: float | list[float]


@dataclass(frozen=True, repr=False)
class SolarGain(AutoStrMixin):
    """Dataclass to represent the solar gain to the human body.

    Attributes
    ----------
    erf : float or list of floats
        Solar gain to the human body using the Effective Radiant Field [W/m2].
    delta_mrt : float or list of floats
        Delta mean radiant temperature. The amount by which the mean radiant
        temperature of the space should be increased if no solar radiation is present.
    """

    erf: float | list[float]
    delta_mrt: float | list[float]


@dataclass(frozen=True, repr=False)
class GaggeTwoNodes(AutoStrMixin):
    """Dataclass to represent the results of the two-node model of human temperature
    regulation.

    Attributes
    ----------
    e_skin : float or list of floats
        Total rate of evaporative heat loss from skin, [W/m2]. Equal to e_rsw + e_diff.
    e_rsw : float or list of floats
        Rate of evaporative heat loss from sweat evaporation, [W/m2].
    e_max : float or list of floats
        Maximum rate of evaporative heat loss from skin, [W/m2].
    q_sensible : float or list of floats
        Sensible heat loss from skin, [W/m2].
    q_skin : float or list of floats
        Total rate of heat loss from skin, [W/m2]. Equal to q_sensible + e_skin.
    q_res : float or list of floats
        Total rate of heat loss through respiration, [W/m2].
    t_core : float or list of floats
        Core temperature, [°C].
    t_skin : float or list of floats
        Skin temperature, [°C].
    m_bl : float or list of floats
        Skin blood flow, [kg/h/m2].
    m_rsw : float or list of floats
        Rate at which regulatory sweat is generated, [mL/h/m2].
    w : float or list of floats
        Skin wettedness, adimensional. Ranges from 0 to 1.
    w_max : float or list of floats
        Skin wettedness (w) practical upper limit, adimensional. Ranges from 0 to 1.
    set : float or list of floats
        Standard Effective Temperature (SET).
    et : float or list of floats
        New Effective Temperature (ET).
    pmv_gagge : float or list of floats
        PMV Gagge.
    pmv_set : float or list of floats
        PMV SET.
    disc : float or list of floats
        Thermal discomfort.
    t_sens : float or list of floats
        Predicted Thermal Sensation.
    """

    e_skin: float | list[float]
    e_rsw: float | list[float]
    e_max: float | list[float]
    q_sensible: float | list[float]
    q_skin: float | list[float]
    q_res: float | list[float]
    t_core: float | list[float]
    t_skin: float | list[float]
    m_bl: float | list[float]
    m_rsw: float | list[float]
    w: float | list[float]
    w_max: float | list[float]
    set: float | list[float]
    et: float | list[float]
    pmv_gagge: float | list[float]
    pmv_set: float | list[float]
    disc: float | list[float]
    t_sens: float | list[float]


@dataclass(frozen=True, repr=False)
class GaggeTwoNodesJi(AutoStrMixin):
    """Dataclass to represent the results of the Gagge-Ji model of human temperature.

    Attributes
    ----------
    t_core : float or list of floats
        Core temperature, [°C].
    t_skin : float or list of floats
        Skin temperature, [°C].
    """

    t_core: float | list[float]
    t_skin: float | list[float]


@dataclass(frozen=True, repr=False)
class THI(AutoStrMixin):
    """Dataclass to represent the Temperature-Humidity Index (THI).

    Attributes
    ----------
    thi : float or list of floats
        Temperature-Humidity Index (THI).
    """

    thi: float | list[float]


@dataclass(frozen=True, repr=False)
class GaggeTwoNodesSleep(AutoStrMixin):
    """Dataclass to represent the results of the two-node sleep model.

    Attributes
    ----------
    set : float or list of floats
        Standard Effective Temperature (SET).
    t_core : float or list of floats
        Core temperature, [°C].
    t_skin : float or list of floats
        Skin temperature, [°C].
    wet : float or list of floats
        Skin wettedness, adimensional. Ranges from 0 to 1.
    t_sens : float or list of floats
        Predicted Thermal Sensation.
    disc : float or list of floats
        Thermal discomfort.
    e_skin : float or list of floats
        Total rate of evaporative heat loss from skin, [W/m2]. Equal to e_rsw + e_diff.
    met_shivering : float or list of floats
        Metabolic rate due to shivering, [W/m2].
    alfa : float or list of floats
        Dynamic fraction of total body mass assigned to the skin node (dimensionless).
    skin_blood_flow: float or list of floats
        Skin-blood-flow rate per unit surface area, [kg/h/m2].
    """

    set: float | list[float]
    t_core: float | list[float]
    t_skin: float | list[float]
    wet: float | list[float]
    t_sens: float | list[float]
    disc: float | list[float]
    e_skin: float | list[float]
    met_shivering: float | list[float]
    alfa: float | list[float]
    skin_blood_flow: float | list[float]


@dataclass(frozen=True)
class UseFansHeatwaves(AutoStrMixin):
    """Dataclass to represent the results of using fans during heatwaves.

    Attributes
    ----------
    e_skin : float or list of floats
        Total rate of evaporative heat loss from skin, [W/m2]. Equal to e_rsw + e_diff.
    e_rsw : float or list of floats
        Rate of evaporative heat loss from sweat evaporation, [W/m2].
    e_max : float or list of floats
        Maximum rate of evaporative heat loss from skin, [W/m2].
    q_sensible : float or list of floats
        Sensible heat loss from skin, [W/m2].
    q_skin : float or list of floats
        Total rate of heat loss from skin, [W/m2]. Equal to q_sensible + e_skin.
    q_res : float or list of floats
        Total rate of heat loss through respiration, [W/m2].
    t_core : float or list of floats
        Core temperature, [°C].
    t_skin : float or list of floats
        Skin temperature, [°C].
    m_bl : float or list of floats
        Skin blood flow, [kg/h/m2].
    m_rsw : float or list of floats
        Rate at which regulatory sweat is generated, [mL/h/m2].
    w : float or list of floats
        Skin wettedness, adimensional. Ranges from 0 to 1.
    w_max : float or list of floats
        Skin wettedness (w) practical upper limit, adimensional. Ranges from 0 to 1.
    heat_strain : bool or list of bools
        True if the model predicts that the person may be experiencing heat strain.
    heat_strain_blood_flow : bool or list of bools
        True if heat strain is caused by skin blood flow (m_bl) reaching its maximum value.
    heat_strain_w : bool or list of bools
        True if heat strain is caused by skin wettedness (w) reaching its maximum value.
    heat_strain_sweating : bool or list of bools
        True if heat strain is caused by regulatory sweating (m_rsw) reaching its maximum value.
    """

    e_skin: float | list[float]
    e_rsw: float | list[float]
    e_max: float | list[float]
    q_sensible: float | list[float]
    q_skin: float | list[float]
    q_res: float | list[float]
    t_core: float | list[float]
    t_skin: float | list[float]
    m_bl: float | list[float]
    m_rsw: float | list[float]
    w: float | list[float]
    w_max: float | list[float]
    heat_strain: bool | list[bool]
    heat_strain_blood_flow: bool | list[bool]
    heat_strain_w: bool | list[bool]
    heat_strain_sweating: bool | list[bool]


@dataclass(frozen=True, repr=False)
class UTCI(AutoStrMixin):
    """Dataclass to represent the Universal Thermal Climate Index (UTCI).

    Attributes
    ----------
    utci : float or list of floats
        Universal Thermal Climate Index, [°C] or in [°F].
    stress_category : str or list of strs
        UTCI categorized in terms of thermal stress [Blazejczyk2013]_.
    """

    utci: float | list[float]
    stress_category: str | list[str]


@dataclass(frozen=True, repr=False)
class VerticalTGradPPD(AutoStrMixin):
    """Dataclass to represent the Predicted Percentage of Dissatisfied (PPD) with
    vertical temperature gradient.

    Attributes
    ----------
    ppd_vg : float or list of floats
        Predicted Percentage of Dissatisfied occupants with vertical temperature gradient.
    acceptability : bool or list of bools
        True if the value of air speed at the ankle level is acceptable (PPD_vg <= 5%).
    """

    ppd_vg: float | list[float]
    acceptability: bool | list[bool]


@dataclass(frozen=True, repr=False)
class WBGT(AutoStrMixin):
    """Dataclass to represent the Wet Bulb Globe Temperature (WBGT) index.

    Attributes
    ----------
    wbgt : float or list of floats
        Wet Bulb Globe Temperature Index.
    """

    wbgt: float | list[float]


@dataclass(frozen=True, repr=False)
class WCI(AutoStrMixin):
    """Dataclass to represent the Wind Chill Index (WCI).

    Attributes
    ----------
    wci : float or list of floats
        Wind Chill Index, [W/m^2].
    """

    wci: float | list[float]


@dataclass(frozen=True, repr=False)
class WCT(AutoStrMixin):
    """Dataclass to represent the Wind Chill Temperature (WCT).

    Attributes
    ----------
    wct : float or list of floats
        Wind Chill Temperature, [°C].
    """

    wct: float | list[float]


@dataclass(frozen=True)
class WorkCapacity(AutoStrMixin):
    """Dataclass to represent work loss.

    Attributes
    ----------
    capacity : float or list of floats
        Work capacity affected by heat.
    """

    capacity: float | list[float]


@dataclass(frozen=True, repr=False)
class JOS3BodyParts(AutoStrMixin):
    """Dataclass to represent the body parts in the JOS3 model.

    It is very important to
    keep the order of the attributes as they are defined in the dataclass ['head',
    'neck', 'chest', 'back', 'pelvis', 'left_shoulder', 'left_arm', 'left_hand',
    'right_shoulder', 'right_arm', 'right_hand', 'left_thigh', 'left_leg', 'left_foot',
    'right_thigh', 'right_leg', 'right_foot']

    Attributes
    ----------
    head : float
        Index of the head.
    neck : float
        Index of the neck.
    chest : float
        Index of the chest.
    back : float
        Index of the back.
    pelvis : float
        Index of the pelvis.
    left_shoulder : float
        Index of the left shoulder.
    left_arm : float
        Index of the left arm.
    left_hand : float
        Index of the left hand.
    right_shoulder : float
        Index of the right shoulder.
    right_arm : float
        Index of the right arm.
    right_hand : float
        Index of the right hand.
    left_thigh : float
        Index of the left thigh.
    left_leg : float
        Index of the left leg.
    left_foot : float
        Index of the left foot.
    right_thigh : float
        Index of the right thigh.
    right_leg : float
        Index of the right leg.
    right_foot : float
        Index of the right hand.
    """

    head: float | None = None
    neck: float | None = None
    chest: float | None = None
    back: float | None = None
    pelvis: float | None = None
    left_shoulder: float | None = None
    left_arm: float | None = None
    left_hand: float | None = None
    right_shoulder: float | None = None
    right_arm: float | None = None
    right_hand: float | None = None
    left_thigh: float | None = None
    left_leg: float | None = None
    left_foot: float | None = None
    right_thigh: float | None = None
    right_leg: float | None = None
    right_foot: float | None = None

    @classmethod
    def get_attribute_names(cls):
        return [field.name for field in fields(cls)]


def get_attribute_values(cls):
    return np.asarray([getattr(cls, field.name) for field in fields(cls)])


@dataclass(frozen=True, repr=False)
class JOS3Output(AutoStrMixin):
    """Dataclass to represent the output of the JOS3 model simulation.

    Attributes
    ----------
    simulation_time : datetime.timedelta
        The elapsed simulation time.
    dt : int or float
        The time step in seconds.
    t_skin_mean : float
        Mean skin temperature of the whole body [°C].
    t_skin : np.ndarray
        Skin temperatures by the local body segments [°C].
    t_core : np.ndarray
        Core temperatures by the local body segments [°C].
    w_mean : float
        Mean skin wettedness of the whole body [-].
    w : np.ndarray
        Skin wettedness on local body segments [-].
    weight_loss_by_evap_and_res : float
        Weight loss by evaporation and respiration [g/sec].
    cardiac_output : float
        Cardiac output [L/h].
    q_thermogenesis_total : float
        Total thermogenesis [W].
    q_res : float
        Heat loss by respiration [W].
    q_skin2env : np.ndarray
        Total heat loss from the skin to the environment [W].
    height : float
        Body height [m].
    weight : float
        Body weight [kg].
    bsa : np.ndarray
        Body surface area.
    fat : float
        Body fat rate [%].
    sex : str
        Sex.
    age : int
        Age [years].
    t_core_set : np.ndarray
        Core set point temperature [°C].
    t_skin_set : np.ndarray
        Skin set point temperature [°C].
    t_cb : float
        Central blood temperature [°C].
    t_artery : np.ndarray
        Arterial blood temperature [°C].
    t_vein : np.ndarray
        Venous blood temperature [°C].
    t_superficial_vein : np.ndarray
        Superficial venous blood temperature [°C].
    t_muscle : np.ndarray
        Muscle temperature [°C].
    t_fat : np.ndarray
        Fat temperature [°C].
    to : float
        Operative temperature [°C].
    r_t : np.ndarray
        Radiative heat transfer coefficient.
    r_et : np.ndarray
        Evaporative heat transfer coefficient.
    tdb : np.ndarray
        Dry bulb air temperature [°C].
    tr : np.ndarray
        Mean radiant temperature [°C].
    rh : np.ndarray
        Relative humidity [%].
    v : np.ndarray
        Air velocity [m/s].
    par : float
        Physical activity ratio.
    clo : np.ndarray
        Clothing insulation.
    e_skin : np.ndarray
        Evaporative heat loss from the skin [W].
    e_max : np.ndarray
        Maximum evaporative heat loss from the skin [W].
    e_sweat : np.ndarray
        Evaporative heat loss from the skin by only sweating [W].
    bf_core : np.ndarray
        Core blood flow rate [L/h].
    bf_muscle : np.ndarray
        Muscle blood flow rate [L/h].
    bf_fat : np.ndarray
        Fat blood flow rate [L/h].
    bf_skin : np.ndarray
        Skin blood flow rate [L/h].
    bf_ava_hand : np.ndarray
        AVA blood flow rate of one hand [L/h].
    bf_ava_foot : np.ndarray
        AVA blood flow rate of one foot [L/h].
    q_bmr_core : np.ndarray
        Core thermogenesis by basal metabolism [W].
    q_bmr_muscle : np.ndarray
        Muscle thermogenesis by basal metabolism [W].
    q_bmr_fat : np.ndarray
        Fat thermogenesis by basal metabolism [W].
    q_bmr_skin : np.ndarray
        Skin thermogenesis by basal metabolism [W].
    q_work : np.ndarray
        Thermogenesis by work [W].
    q_shiv : np.ndarray
        Thermogenesis by shivering [W].
    q_nst : np.ndarray
        Thermogenesis by non-shivering [W].
    q_thermogenesis_core : np.ndarray
        Core total thermogenesis [W].
    q_thermogenesis_muscle : np.ndarray
        Muscle total thermogenesis [W].
    q_thermogenesis_fat : np.ndarray
        Fat total thermogenesis [W].
    q_thermogenesis_skin : np.ndarray
        Skin total thermogenesis [W].
    q_skin2env_sensible : np.ndarray
        Sensible heat loss from the skin to the environment [W].
    q_skin2env_latent : np.ndarray
        Latent heat loss from the skin to the environment [W].
    q_res_sensible : np.ndarray
        Sensible heat loss by respiration [W].
    q_res_latent : np.ndarray
        Latent heat loss by respiration [W].
    """

    simulation_time: dt.timedelta | None = None
    dt: float | None = None
    t_skin_mean: float | None = None
    t_skin: JOS3BodyParts | None = None
    t_core: JOS3BodyParts | None = None
    w_mean: float | None = None
    w: JOS3BodyParts | None = None
    weight_loss_by_evap_and_res: float | None = None
    cardiac_output: float | None = None
    q_thermogenesis_total: float | None = None
    q_res: float | None = None
    q_skin2env: JOS3BodyParts | None = None
    height: float | None = None
    weight: float | None = None
    bsa: JOS3BodyParts | None = None
    fat: float | None = None
    sex: str | None = None
    age: int | None = None
    t_core_set: JOS3BodyParts | None = None
    t_skin_set: JOS3BodyParts | None = None
    t_cb: float | None = None
    t_artery: JOS3BodyParts | None = None
    t_vein: JOS3BodyParts | None = None
    t_superficial_vein: JOS3BodyParts | None = None
    t_muscle: JOS3BodyParts | None = None
    t_fat: JOS3BodyParts | None = None
    to: JOS3BodyParts | None = None
    r_t: JOS3BodyParts | None = None
    r_et: JOS3BodyParts | None = None
    tdb: JOS3BodyParts | None = None
    tr: JOS3BodyParts | None = None
    rh: JOS3BodyParts | None = None
    v: JOS3BodyParts | None = None
    par: float | None = None
    clo: JOS3BodyParts | None = None
    e_skin: JOS3BodyParts | None = None
    e_max: JOS3BodyParts | None = None
    e_sweat: JOS3BodyParts | None = None
    bf_core: JOS3BodyParts | None = None
    bf_muscle: JOS3BodyParts | None = None
    bf_fat: JOS3BodyParts | None = None
    bf_skin: JOS3BodyParts | None = None
    bf_ava_hand: float | None = None
    bf_ava_foot: float | None = None
    q_bmr_core: JOS3BodyParts | None = None
    q_bmr_muscle: JOS3BodyParts | None = None
    q_bmr_fat: JOS3BodyParts | None = None
    q_bmr_skin: JOS3BodyParts | None = None
    q_work: JOS3BodyParts | None = None
    q_shiv: JOS3BodyParts | None = None
    q_nst: JOS3BodyParts | None = None
    q_thermogenesis_core: JOS3BodyParts | None = None
    q_thermogenesis_muscle: JOS3BodyParts | None = None
    q_thermogenesis_fat: JOS3BodyParts | None = None
    q_thermogenesis_skin: JOS3BodyParts | None = None
    q_skin2env_sensible: JOS3BodyParts | None = None
    q_skin2env_latent: JOS3BodyParts | None = None
    q_res_sensible: float | None = None
    q_res_latent: float | None = None


@dataclass(frozen=True, repr=False)
class PredictedBodyTemperatures(AutoStrMixin):
    """Dataclass for returning predicted temperature history.

    The `duration` mentioned in the shapes below is the `duration` parameter
    passed to the prediction function.

    Attributes
    ----------
    t_re : numpy.ndarray of float, in °C
        Predicted rectal temperature history.
        - If scalar inputs are provided, this is a 1D array of shape (`duration`,).
        - If vector inputs are provided, this is a 2D array of shape (`n_inputs`, `duration`).
    t_sk : numpy.ndarray of float, in °C
        Predicted mean skin temperature history.
        - If scalar inputs are provided, this is a 1D array of shape (`duration`,).
        - If vector inputs are provided, this is a 2D array of shape (`n_inputs`, `duration`).
    """

    t_re: np.ndarray
    t_sk: np.ndarray


@dataclass(frozen=True, repr=False)
class ScaleWindSpeedLog(AutoStrMixin):
    """Dataclass to represent the output of scale_wind_speed_log.

    Attributes
    ----------
    v_z2 : numpy.ndarray or float or list of floats
        Wind speed at the target height z2, [m/s]. The type can be a scalar,
        list or numpy array depending on the inputs provided to the function.
    """

    v_z2: npt.ArrayLike
