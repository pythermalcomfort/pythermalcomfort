from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pythermalcomfort._internal.validation import _mapping, validate_type
from pythermalcomfort.classes_input import NumericInput
from pythermalcomfort.classes_return import HeatWatch
from pythermalcomfort.models.two_nodes_gagge import two_nodes_gagge
from pythermalcomfort.utilities import Postures

# DISC equals 4.7 when skin wettedness reaches w_max, so D = 1 at the person's limit.
_DISC_AT_W_MAX = 4.7
# Core temperature is normalised between the person's core temperature at the same
# activity in cool air and the rectal temperature limit used by the Predicted Heat
# Strain model (ISO 7933), both in °C. The baseline is never below the Gagge neutral
# value.
_T_CORE_NEUTRAL = 36.8
_T_CORE_LIMIT = 38.0
# Cool reference environment for the baseline. At a given metabolic rate, core
# temperature does not depend on the environment over a wide range of cool climates
# (the "prescriptive zone") [Lind1963]. In the Gagge model the baseline changes by
# less than 0.05 °C between 10 and 15 °C for metabolic rates up to 3 met.
_REFERENCE_TMP = 15.0
_REFERENCE_V = 0.2
_REFERENCE_RH = 50.0
# The Gagge model's largest core temperature errors occur above 3.75 met in hot
# environments [Joshi2025], and at higher rates the baseline approaches the core
# temperature limit even in cool air.
_MET_MAX = 3.75
_HHRS_MAX = 6.0

# Each category covers one unit of the score, including its upper edge.
_RISK_CATEGORIES = {
    1: "Minimal",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Severe",
    6: "Extreme",
}

# Children aged 10-16 years have a similar hyperthermia risk to adults [Smallcombe2025],
# so they use the adult w_max. There is no comparable evidence for younger children.
_MIN_AGE = 10
# Age bands (upper edge, exclusive) with the maximum skin wettedness.
_AGE_BANDS = (
    (40, 0.85),
    (60, 0.8),
    (70, 0.75),
    (80, 0.7),
    (np.inf, 0.65),
)
_W_MAX_FACTOR_HEALTH_CONDITION = 0.95
# two_nodes_gagge accepts only these two positions.
_POSTURES = (Postures.sitting.value, Postures.standing.value)


@dataclass(frozen=True)
class HeatWatchPersona:
    """Personal characteristics used by :py:func:`heatwatch`.

    Parameters
    ----------
    age : int
        Age, [years]. Must be 10 or older. Children aged 10 to 16 years have a
        similar risk of hyperthermia to adults during exercise in the heat
        [Smallcombe2025]_, so they use the adult maximum skin wettedness. There is
        no comparable evidence for younger children.
    met : float
        Metabolic rate, [met]. Must be greater than 0 and no more than 3.75, because
        the Gagge two-node model has its largest core temperature errors above
        3.75 met in hot environments [Joshi2025]_.
    clo : float
        Clothing insulation, [clo].
    posture : str, optional
        Posture passed to the Gagge two-node model, either "sitting" or "standing".
        Defaults to "standing".
    obesity : bool, optional
        True if the person has obesity. Defaults to False.
    medication : bool, optional
        True if the person takes medication that impairs thermoregulation. Defaults to
        False.
    pregnancy : bool, optional
        True if the person is pregnant. Defaults to False.
    mobility_impairment : bool, optional
        True if the person has a mobility impairment. Defaults to False.
    illness : bool, optional
        True if the person has a chronic illness, for example cardiovascular disease.
        Defaults to False.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models.heatwatch import HeatWatchPersona

        older_adult = HeatWatchPersona(
            age=75, met=1.2, clo=0.45, posture="sitting", illness=True, medication=True
        )
        print(round(older_adult.w_max, 3))  # 0.632
    """

    age: int
    met: float
    clo: float
    posture: str = Postures.standing.value
    obesity: bool = False
    medication: bool = False
    pregnancy: bool = False
    mobility_impairment: bool = False
    illness: bool = False

    def __post_init__(self):
        validate_type(value=self.age, name="age", allowed_types=(int,))
        validate_type(value=self.met, name="met", allowed_types=(int, float))
        validate_type(value=self.clo, name="clo", allowed_types=(int, float))
        validate_type(value=self.posture, name="posture", allowed_types=(str,))
        for name in (
            "obesity",
            "medication",
            "pregnancy",
            "mobility_impairment",
            "illness",
        ):
            validate_type(value=getattr(self, name), name=name, allowed_types=(bool,))

        if self.age < _MIN_AGE:
            msg = f"age must be an integer >= {_MIN_AGE}, got {self.age}"
            raise ValueError(msg)
        if not (np.isfinite(self.met) and 0 < self.met <= _MET_MAX):
            msg = f"met must be a number > 0 and <= {_MET_MAX}, got {self.met}"
            raise ValueError(msg)
        if not (np.isfinite(self.clo) and self.clo >= 0):
            msg = f"clo must be a finite number >= 0, got {self.clo}"
            raise ValueError(msg)
        if self.posture not in _POSTURES:
            msg = f"posture must be one of {list(_POSTURES)}, got {self.posture!r}"
            raise ValueError(msg)

    @property
    def w_max(self) -> float:
        """Maximum skin wettedness, [dimensionless].

        Depends on age and is multiplied by 0.95 for each of obesity, medication,
        pregnancy, mobility impairment and illness.
        """
        w_max = next(w for upper_age, w in _AGE_BANDS if self.age < upper_age)
        health_conditions = sum(
            (
                self.obesity,
                self.medication,
                self.pregnancy,
                self.mobility_impairment,
                self.illness,
            )
        )
        return w_max * _W_MAX_FACTOR_HEALTH_CONDITION**health_conditions

    @property
    def t_core_baseline(self) -> float:
        """Core temperature at the same activity in cool air, [°C].

        Calculated with the Gagge two-node model at 15 °C air and mean radiant
        temperature, 50 % relative humidity and 0.2 m/s, with the person's metabolic
        rate, clothing, posture and ``w_max``. It is never below 36.8 °C, the Gagge
        neutral core temperature. The rise in core temperature above this value is the
        part caused by heat rather than by the activity itself [Lind1963]_.
        """
        t_core = two_nodes_gagge(
            tdb=_REFERENCE_TMP,
            tr=_REFERENCE_TMP,
            v=_REFERENCE_V,
            rh=_REFERENCE_RH,
            met=self.met,
            clo=self.clo,
            position=self.posture,
            w_max=self.w_max,
            round_output=False,
        ).t_core
        return max(_T_CORE_NEUTRAL, float(t_core))


def heatwatch(
    tdb: NumericInput,
    tr: NumericInput,
    v: NumericInput,
    rh: NumericInput,
    persona: HeatWatchPersona,
    round_output: bool = True,
) -> HeatWatch:
    """Calculate the HeatWatch heat-health risk score (HHRS) for a person.

    The score combines two outputs of the Gagge two-node model [Gagge1986]_, run with
    the person's metabolic rate, clothing, posture and maximum skin wettedness
    (``w_max``):

    * relative skin wettedness, ``D = disc / 4.7``, which is 0 at thermal comfort and
      1 when skin wettedness reaches ``w_max``;
    * relative core temperature, ``C = (t_core - t_base) / (38.0 - t_base)``, which
      is 0 at ``t_base`` and 1 at 38.0 °C, the rectal temperature limit used by the
      Predicted Heat Strain model (ISO 7933). ``t_base`` is
      :py:attr:`HeatWatchPersona.t_core_baseline`, the person's core temperature at
      the same activity in cool air, so the normal rise in core temperature during
      work does not count as heat strain [Lind1963]_.

    The score follows whichever limit the person is closer to:

    .. math::

        HHRS = 6 \\, \\mathrm{clip}(\\max(D, C), 0, 1)

    Skin blood flow is not used, because in the Gagge model it follows core
    temperature and saturates at a model constant (90 kg/h/m2), not at a
    physiological limit. The score therefore does not represent cardiovascular
    strain.

    The score has no separate air temperature cut-off, so radiant heat and metabolic
    rate raise it through the Gagge model even in cool air. Outdoors, use the air
    speed at body height and a mean radiant temperature that includes the solar load,
    because the score is sensitive to both.

    The score is assigned to six categories: Minimal (0 to 1), Low (above 1 to 2),
    Moderate, High, Severe and Extreme (above 5 to 6). With ``round_output=True`` the
    category is assigned from the rounded score, so the label always matches the
    displayed value.

    Each condition is simulated as a separate 60-minute exposure. The score is a
    relative indicator of physiological strain, not a probability of a health event.

    Parameters
    ----------
    tdb : float or list of floats
        Dry bulb air temperature, [°C].
    tr : float or list of floats
        Mean radiant temperature, [°C].
    v : float or list of floats
        Air speed, [m/s].
    rh : float or list of floats
        Relative humidity, [%].
    persona : HeatWatchPersona
        Personal characteristics, including ``met``, ``clo`` and ``posture``.
    round_output : bool, optional
        If True, rounds the score to one decimal place. Defaults to True.

    Returns
    -------
    HeatWatch
        A dataclass containing the score, the risk category, and the DISC and core
        temperature values from the Gagge model. See
        :py:class:`~pythermalcomfort.classes_return.HeatWatch` for more details.

    Raises
    ------
    TypeError
        If ``persona`` is not a :py:class:`HeatWatchPersona` instance.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models.heatwatch import HeatWatchPersona, heatwatch

        older_adult = HeatWatchPersona(
            age=75, met=1.2, clo=0.45, posture="sitting", illness=True, medication=True
        )
        result = heatwatch(tdb=[30, 38], tr=[30, 38], v=0.2, rh=50, persona=older_adult)
        print(result.hhrs)  # [2.2 6. ]
        print(result.risk_category)  # ['Moderate' 'Extreme']
    """
    if not isinstance(persona, HeatWatchPersona):
        raise TypeError("persona must be a HeatWatchPersona instance.")

    gagge = two_nodes_gagge(
        tdb=tdb,
        tr=tr,
        v=v,
        rh=rh,
        met=persona.met,
        clo=persona.clo,
        position=persona.posture,
        w_max=persona.w_max,
        round_output=False,
    )
    disc = np.asarray(gagge.disc, dtype=float)
    t_core = np.asarray(gagge.t_core, dtype=float)

    relative_wettedness = disc / _DISC_AT_W_MAX
    t_core_baseline = persona.t_core_baseline
    relative_core = (t_core - t_core_baseline) / (_T_CORE_LIMIT - t_core_baseline)
    combined = np.maximum(relative_wettedness, relative_core)
    hhrs = _HHRS_MAX * np.clip(combined, 0, 1)

    if round_output:
        hhrs = np.round(hhrs, 1)

    # [()] turns a 0-d array from scalar input into a scalar and leaves arrays as-is
    return HeatWatch(
        hhrs=hhrs,
        risk_category=_mapping(hhrs, _RISK_CATEGORIES),
        disc=disc[()],
        t_core=t_core[()],
    )
