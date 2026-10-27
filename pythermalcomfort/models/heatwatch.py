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
# Skin blood flow is normalised between the Gagge neutral value and the maximum
# passed to the model, both in kg/h/m2.
_M_BL_NEUTRAL = 6.3
_M_BL_MAX = 90.0
# The score is scaled down linearly from full value at 25 °C to zero at 20 °C.
_TDB_CORRECTION_ZERO = 20.0
_TDB_CORRECTION_FULL = 25.0
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

# Age bands (upper edge, exclusive) with the maximum skin wettedness and the weight
# given to relative skin wettedness over skin blood flow.
_AGE_BANDS = (
    (18, 0.85, 0.6),
    (40, 0.85, 0.7),
    (60, 0.8, 0.55),
    (70, 0.75, 0.5),
    (80, 0.7, 0.4),
    (np.inf, 0.65, 0.3),
)
_W_MAX_FACTOR_HEALTH_CONDITION = 0.95
_W_MAX_FACTOR_ACCLIMATISED = 1 / 0.9


@dataclass(frozen=True)
class Persona:
    """Personal characteristics used by :py:func:`heatwatch`.

    Parameters
    ----------
    age : int
        Age, [years].
    met : float
        Metabolic rate, [met].
    clo : float
        Clothing insulation, [clo].
    posture : str, optional
        Posture passed to the Gagge two-node model. One of the values of
        :py:class:`~pythermalcomfort.utilities.Postures`. Defaults to "standing".
    acclimatised : bool, optional
        True if the person is acclimatised to heat. Raises w_max. Defaults to False.
    obesity : bool, optional
        True if the person has obesity. Defaults to False.
    medication : bool, optional
        True if the person takes medication that impairs thermoregulation. Defaults to
        False.
    pregnancy : bool, optional
        True if the person is pregnant. Defaults to False.
    mobility_impairment : bool, optional
        True if the person has a mobility impairment. Defaults to False.
    homelessness : bool, optional
        True if the person is experiencing homelessness. Defaults to False.
    illness : bool, optional
        True if the person has a chronic illness, for example cardiovascular disease.
        Defaults to False.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models.heatwatch import Persona

        older_adult = Persona(
            age=75, met=1.2, clo=0.45, posture="sitting", illness=True, medication=True
        )
        print(round(older_adult.w_max, 3))  # 0.632
        print(older_adult.alpha)  # 0.4
    """

    age: int
    met: float
    clo: float
    posture: str = Postures.standing.value
    acclimatised: bool = False
    obesity: bool = False
    medication: bool = False
    pregnancy: bool = False
    mobility_impairment: bool = False
    homelessness: bool = False
    illness: bool = False

    def __post_init__(self):
        validate_type(value=self.age, name="age", allowed_types=(int,))
        validate_type(value=self.met, name="met", allowed_types=(int, float))
        validate_type(value=self.clo, name="clo", allowed_types=(int, float))
        validate_type(value=self.posture, name="posture", allowed_types=(str,))
        for name in (
            "acclimatised",
            "obesity",
            "medication",
            "pregnancy",
            "mobility_impairment",
            "homelessness",
            "illness",
        ):
            validate_type(value=getattr(self, name), name=name, allowed_types=(bool,))

        if self.age < 0:
            msg = f"age must be a non-negative integer >= 0, got {self.age}"
            raise ValueError(msg)
        if self.met <= 0:
            msg = f"met must be a positive number > 0, got {self.met}"
            raise ValueError(msg)
        if self.clo < 0:
            msg = f"clo must be a non-negative number >= 0, got {self.clo}"
            raise ValueError(msg)
        allowed_postures = [posture.value for posture in Postures]
        if self.posture not in allowed_postures:
            msg = f"posture must be one of {allowed_postures}, got {self.posture!r}"
            raise ValueError(msg)

    def _age_band(self) -> tuple[float, float]:
        for upper_age, w_max, alpha in _AGE_BANDS:
            if self.age < upper_age:
                return w_max, alpha
        raise AssertionError("the last age band has no upper limit")

    @property
    def w_max(self) -> float:
        """Maximum skin wettedness, [dimensionless].

        Depends on age, is divided by 0.9 for acclimatised people, and is multiplied by
        0.95 for each of obesity, medication, pregnancy, mobility impairment and
        illness.
        """
        w_max, _ = self._age_band()
        if self.acclimatised:
            w_max *= _W_MAX_FACTOR_ACCLIMATISED
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
    def alpha(self) -> float:
        """Weight of relative skin wettedness over skin blood flow, [dimensionless].

        Depends on age, and is at most 0.6 with obesity or illness and at most 0.5 with
        homelessness.
        """
        _, alpha = self._age_band()
        if self.obesity or self.illness:
            alpha = min(alpha, 0.6)
        if self.homelessness:
            alpha = min(alpha, 0.5)
        return alpha


def heatwatch(
    tdb: NumericInput,
    tr: NumericInput,
    v: NumericInput,
    rh: NumericInput,
    persona: Persona,
    round_output: bool = True,
) -> HeatWatch:
    """Calculate the HeatWatch heat-health risk score (HHRS) for a person.

    The score combines two outputs of the Gagge two-node model [Gagge1986]_, run with
    the person's metabolic rate, clothing, posture and maximum skin wettedness
    (``w_max``):

    * relative skin wettedness, ``D = disc / 4.7``, which is 0 at thermal comfort and
      1 when skin wettedness reaches ``w_max``;
    * normalised skin blood flow, ``B = (m_bl - 6.3) / (90 - 6.3)``, which is 0 at the
      neutral skin blood flow and 1 at the maximum.

    The person's weight ``alpha`` sets how much blood flow can add to the score:

    .. math::

        HHRS = 6 \\, c(t_{db}) \\, \\mathrm{clip}(\\max(D, \\alpha D + (1 - \\alpha) B), 0, 1)

    Blood flow can only raise the score above ``D``, never lower it. The factor
    ``c(t_db)`` is 0 below 20 °C, rises linearly to 1 at 25 °C, and is 1 above 25 °C.

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
    persona : Persona
        Personal characteristics, including ``met``, ``clo`` and ``posture``.
    round_output : bool, optional
        If True, rounds the score to one decimal place. Defaults to True.

    Returns
    -------
    HeatWatch
        A dataclass containing the score, the risk category, and the DISC and skin
        blood flow values from the Gagge model. See
        :py:class:`~pythermalcomfort.classes_return.HeatWatch` for more details.

    Raises
    ------
    TypeError
        If ``persona`` is not a :py:class:`Persona` instance.

    Examples
    --------
    .. code-block:: python

        from pythermalcomfort.models.heatwatch import Persona, heatwatch

        older_adult = Persona(
            age=75, met=1.2, clo=0.45, posture="sitting", illness=True, medication=True
        )
        result = heatwatch(tdb=[30, 38], tr=[30, 38], v=0.2, rh=50, persona=older_adult)
        print(result.hhrs)  # [2.2 6. ]
        print(result.risk_category)  # ['Moderate' 'Extreme']
    """
    if not isinstance(persona, Persona):
        raise TypeError("persona must be a Persona instance.")

    gagge = two_nodes_gagge(
        tdb=tdb,
        tr=tr,
        v=v,
        rh=rh,
        met=persona.met,
        clo=persona.clo,
        position=persona.posture,
        max_skin_blood_flow=_M_BL_MAX,
        w_max=persona.w_max,
        round_output=False,
    )
    disc = np.asarray(gagge.disc, dtype=float)
    m_bl = np.asarray(gagge.m_bl, dtype=float)
    tdb = np.asarray(tdb, dtype=float)

    relative_wettedness = disc / _DISC_AT_W_MAX
    blood_flow = (m_bl - _M_BL_NEUTRAL) / (_M_BL_MAX - _M_BL_NEUTRAL)
    combined = np.maximum(
        relative_wettedness,
        persona.alpha * relative_wettedness + (1 - persona.alpha) * blood_flow,
    )
    tdb_correction = np.clip(
        (tdb - _TDB_CORRECTION_ZERO) / (_TDB_CORRECTION_FULL - _TDB_CORRECTION_ZERO),
        0,
        1,
    )
    hhrs = _HHRS_MAX * tdb_correction * np.clip(combined, 0, 1)

    if round_output:
        hhrs = np.round(hhrs, 1)

    return HeatWatch(
        hhrs=hhrs,
        risk_category=_mapping(hhrs, _RISK_CATEGORIES),
        disc=disc,
        m_bl=m_bl,
    )
