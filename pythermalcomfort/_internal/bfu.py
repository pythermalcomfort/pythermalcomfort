"""Private constants and heat-balance calculations for biophysical fan models."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pythermalcomfort.psychrometrics import p_sat


@dataclass(frozen=True)
class _RestParameters:
    """Default physical and physiological parameters for the Morris LPH fan model."""

    metabolic_rate: float = 65.0  # [W/m2]
    external_work: float = 0.0  # [W/m2]
    t_sk: float = 35.5  # [°C]
    p_skin_sat_kpa: float = 5.78  # [kPa]
    h_r: float = 4.7  # [W/m2/K]
    body_surface_area: float = 1.83  # [m2]
    latent_heat_j_g: float = 2426.0  # [J/g]
    sweating_efficiency_min: float = 0.55
    r_cl_off: float = 0.1291  # [m2*K/W]
    r_ecl_off: float = 0.0237  # [m2*kPa/W]
    r_cl_on: float = 0.1341  # [m2*K/W]
    r_ecl_on_front: float = 0.0112  # [m2*kPa/W]
    r_ecl_on_rear: float = 0.0161  # [m2*kPa/W]
    v_on: float = 3.5  # [m/s]
    v_off: float = 0.2  # [m/s]
    sweat_rate: dict[str, float] = field(
        default_factory=lambda: {
            "YNG": 660.0,
            "OLD": 440.0,
            "MEDS": 330.0,
        },
    )
    wcrit_on: dict[str, float] = field(
        default_factory=lambda: {
            "YNG": 0.65,
            "OLD": 0.50,
            "MEDS": 0.38,
        },
    )
    wcrit_off: dict[str, float] = field(
        default_factory=lambda: {
            "YNG": 0.85,
            "OLD": 0.65,
            "MEDS": 0.49,
        },
    )


def _rest_fan_state(*, tdb, rh, population, fan_on):
    """Return required and maximum evaporation [W/m²] for one Morris fan state."""
    params = _RestParameters()
    w_crit = params.wcrit_on[population] if fan_on else params.wcrit_off[population]
    r_cl = params.r_cl_on if fan_on else params.r_cl_off
    r_ecl = (
        0.5 * (params.r_ecl_on_front + params.r_ecl_on_rear)
        if fan_on
        else params.r_ecl_off
    )
    v = params.v_on if fan_on else params.v_off
    # Sensible clothing resistance r_cl [m² K/W]; evaporative r_ecl [m² kPa/W].
    f_cl = 1 + 0.31 * r_cl / 0.155
    h_c = 8.3 * v**0.6
    p_a = rh / 100 * p_sat(tdb=tdb) / 1000  # Ambient vapour pressure [kPa].
    dry_heat = (params.t_sk - tdb) / (r_cl + 1 / (f_cl * (h_c + params.h_r)))
    respiratory_heat = 0.0014 * params.metabolic_rate * (34 - tdb) + (
        0.0173 * params.metabolic_rate * (5.87 - p_a)
    )
    e_req = params.metabolic_rate - params.external_work - dry_heat - respiratory_heat
    # Keep the paper's fixed skin vapour pressure and population wettedness limits.
    e_max_physical = (
        w_crit * (params.p_skin_sat_kpa - p_a) / (r_ecl + 1 / (16.5 * h_c * f_cl))
    )
    w_req = np.divide(
        e_req,
        e_max_physical,
        out=np.full_like(e_req, np.inf),
        where=e_max_physical != 0,
    )
    sweat_efficiency = np.maximum(1 - w_req**2 / 2, params.sweating_efficiency_min)
    e_max_sweat = (
        params.sweat_rate[population]
        * params.latent_heat_j_g
        / 3600
        * sweat_efficiency
        / params.body_surface_area
    )
    return e_req, np.minimum(e_max_physical, e_max_sweat)


def _fan_convection(*, v, v_posture, v_activity):
    """Convective heat-transfer coefficient, [W/(m² K)] (Foster Eq. 6-7)."""
    return 8.3 * np.sqrt(np.maximum(v_posture + v + v_activity, 0))


@dataclass(frozen=True)
class _OccupationalHeatBalance:
    """Shared fan/no-fan heat-balance terms in Foster supplementary Eq. 2-34.

    Speeds are m/s, temperatures °C, pressures kPa, heat fluxes W/m²,
    sensible resistances m² K/W, and evaporative resistances m² kPa/W.
    """

    v_posture: np.ndarray
    v_activity: np.ndarray
    walk_sp: np.ndarray
    t_sk: np.ndarray
    t_o: np.ndarray
    i_cl: np.ndarray
    f_cl: np.ndarray
    i_a_off: np.ndarray
    i_t_off: np.ndarray
    r_et_off: np.ndarray
    psat_tsk: np.ndarray
    pa: np.ndarray
    sweat_rate_watt: np.ndarray
    met_arr: np.ndarray
    resp: np.ndarray

    def fan_state(self, vwind: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return storage, dry, evaporative, efficiency, wettedness for a fan state."""
        hc = _fan_convection(
            v=vwind, v_posture=self.v_posture, v_activity=self.v_activity
        )

        corr_it = np.exp(
            -0.281 * (vwind - 0.15)
            + 0.044 * (vwind - 0.15) ** 2
            - 0.492 * self.walk_sp
            + 0.176 * self.walk_sp**2,
        )
        corr_ia = np.exp(
            -0.533 * (vwind - 0.15)
            + 0.069 * (vwind - 0.15) ** 2
            - 0.462 * self.walk_sp
            + 0.201 * self.walk_sp**2,
        )

        clothed_it = self.i_t_off * corr_it
        nude_it = self.i_a_off * corr_ia
        itr = np.where(
            (self.i_cl > 0) & (self.i_cl < 0.093),
            ((0.093 - self.i_cl) * nude_it + self.i_cl * clothed_it) / 0.093,
            np.where(self.i_cl == 0, nude_it, clothed_it),
        )
        dry = (self.t_sk - self.t_o) / np.maximum(itr, 1e-9)

        hedyn = 16.5 * hc
        corr_ret = np.exp(
            -0.468 * (vwind - 0.15)
            + 0.080 * (vwind - 0.15) ** 2
            - 0.87 * self.walk_sp
            + 0.358 * self.walk_sp**2,
        )
        clothed_ret = self.r_et_off * corr_ret
        nude_dyn = 1.0 / np.maximum(hedyn, 1e-9)
        ret_r = np.where(
            self.i_cl <= 1e-9,
            nude_dyn,
            np.where(
                (self.i_cl > 0) & (self.i_cl < 0.093),
                ((0.093 - self.i_cl) * nude_dyn + self.i_cl * clothed_ret) / 0.093,
                clothed_ret,
            ),
        )

        emax = np.maximum(self.psat_tsk - self.pa, 0.0) / np.maximum(ret_r, 1e-9)

        w = np.divide(
            self.sweat_rate_watt,
            emax,
            out=np.full_like(emax, np.inf),
            where=emax > 1e-12,
        )

        sweff = np.where(
            w <= 1.0,
            1.0 - w**2 / 2,
            np.where(
                w <= 1.7,
                (2.0 - w) ** 2 / 2,
                0.05,
            ),
        )
        sweff = np.clip(sweff, 0.0, 1.0)
        # Supplementary Eq. 1 uses the smaller of sweat supply and capacity.
        esweat = np.minimum(self.sweat_rate_watt * sweff, emax)
        storage = self.met_arr - (dry + esweat + self.resp)

        return storage, dry, esweat, sweff, w
