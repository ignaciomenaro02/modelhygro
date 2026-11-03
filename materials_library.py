# -*- coding: utf-8 -*-
"""
materials_library.py
====================
Hygrothermal material properties database.

Each class exposes static methods:
    rho(T, RH)      Dry density              [kg/m³]
    Cp(T, RH)       Heat capacity            [J/(kg·K)]
    k(T, RH)        Thermal conductivity     [W/(m·K)]
    w(T, RH)        Water content (sorption) [kg/m³]
    mu(T, RH)       Vapour resistance factor [-]
    delta_p(T, RH)  Vapour permeability      [kg/(m·s·Pa)]
    Avalue(T, RH)   Liquid absorption coeff  [kg/(m²·s^0.5)]
    wf(T, RH)       Free-saturation content  [kg/m³]
    Dw(T, RH)       Liquid diffusion coeff   [m²/s]
    delta_l(T, RH)  Liquid permeability      [kg/(m·s·Pa)]

Sources
-------
- Rammed_Earth  : Chabriac (2014), Saint-Antoine l'Abbaye, via Soudani et al. (2016) and
                  Bui et al. (2025): nominal values and ranges in Rammed_Earth.PARAMS
- Hempcrete     : Moujalled et al. (2018), ranges from the literature in Hempcrete.PARAMS
- *_legacy      : the two materials as used until October 2026 (poster runs)

Parameter ranges
----------------
Hempcrete and Rammed_Earth carry PARAMS = {name: (nominal, min, max, source)} and read the values in
force from the class dict `p` (nominal by default). Change them with
wall_layer.apply_material_params({"Hempcrete": {"k_dry": "max", "delta_p": 2e-11}}), which is what
simulation.py does with its MATERIAL_PARAMS setting.
- Rock_Wool     : ISO 10456 / HAMSTAD benchmark
- Wood_Fiber    : Kaemmerlen (2010), IEA Annex 55
- Concrete      : EN ISO 10456, Künzel (1995)
- Wood          : EN 13986, Siau (1984)
- Vapor_Barrier : Polyethylene, manufacturer data
- Earth_Plaster : Fabbri & Morel (2014)
- Gypsum_Plaster: Künzel (1995), ISO 10456
- Lime_Plaster  : Lawrence et al. (2009)
- Fermacell     : Fermacell technical datasheet (2023)
- BA13          : Placo / Saint-Gobain datasheet

Adding a new material
---------------------
Copy any class, adjust the values, then register it in wall_layer.py:
    from materials_library import MyMaterial
    _MAT_REGISTRY["MyMaterial"] = MyMaterial
"""

import numpy as np
import library as lib


# ══════════════════════════════════════════════════════════════════════════════
# Helper
# ══════════════════════════════════════════════════════════════════════════════

def _Dw_capillary(Avalue, wf, w):
    """Standard Krus/Künzel formula for liquid diffusion coefficient [m²/s]."""
    ratio = w / max(wf, 1e-6)
    return 3.8 * (Avalue / max(wf, 1e-6))**2 * 1000**(ratio - 1)


# ── Sorption isotherm models: w or u as a function of RH, with their analytical slopes ──────────
def gab(phi, wm, C, K):
    """GAB isotherm, w [kg/m³] (wm in kg/m³)."""
    x = K * phi
    return wm * C * x / ((1 - x) * (1 + (C - 1) * x))


def dgab_dphi(phi, wm, C, K):
    """dw/dRH of the GAB isotherm [kg/m³]."""
    x = K * phi
    return K * wm * C * (1 + (C - 1) * x**2) / ((1 - x)**2 * (1 + (C - 1) * x)**2)


T_ISO = 296.15                              # reference temperature of the isotherms [K] (23 °C)
_H_PER_LN = lib.R * T_ISO / (lib.Mw * 9.81)  # capillary head per unit -ln(RH) [m]


def vg_head(phi, usat, alpha, eta):
    """van Genuchten in capillary head (Moujalled et al. 2018, Eq. 5), u [kg/kg]:
    u = usat·[1 + (alpha·h)^eta]^(1/eta - 1),  h = -R·T_ref/(M·g)·ln(RH) [m].
    Written at the reference temperature, so that w does not depend on T (dw/dT = 0 in the model)."""
    h = -_H_PER_LN * np.log(np.clip(phi, 1e-9, 1.0))
    return usat * (1 + (alpha * h)**eta)**(1 / eta - 1)


def dvg_head_dphi(phi, usat, alpha, eta):
    """du/dRH of vg_head [kg/kg]."""
    phi = np.clip(phi, 1e-9, 1.0)
    ah = alpha * _H_PER_LN * (-np.log(phi))
    du_dh = usat * (1 / eta - 1) * (1 + ah**eta)**(1 / eta - 2) * eta * ah**(eta - 1) * alpha
    return du_dh * (-_H_PER_LN / phi)


def _p(cls):
    """Values in force of a material with PARAMS (nominal by default)."""
    return cls.p


# ══════════════════════════════════════════════════════════════════════════════
# Bio- and geo-sourced materials of the thesis (sourced values and ranges)
# ══════════════════════════════════════════════════════════════════════════════

class Hempcrete:
    """
    Hemp-lime concrete (~450 kg/m³). Nominal values: Moujalled et al. (2018), Building and
    Environment 136, Tables 2-3 (same material as Aït Ouméziane et al. 2016), except the dry thermal
    conductivity: 0.073 W/(m·K), declared value of the Saint-Astier BATICHANVRE data sheet (Dec. 2023)
    for walls (350-450 kg/m³). Ranges:
    Aït Ouméziane et al. (2016) CBM 106; Benkhaled et al. (2022) ICHMT 132, Table 1;
    Bennai et al. (2022) CBM 316; Colinart et al. (sprayed hemp concrete).
    Thermal conductivity linear in the mass water content (Moujalled 2018):
        k = k_dry + (k_sat - k_dry)·u/u_sat
    Isotherm: van Genuchten in capillary head, main adsorption or desorption curve (no hysteresis
    model): iso_branch = "ads" | "des" | "mean".
    """
    PARAMS = {
        #  name           nominal   min       max       source
        "rho":          (450.0,   300.0,    550.0,    "Moujalled 2018 | range Aït Ouméziane 2016, Benkhaled 2022"),
        "cp":           (1250.0,  1000.0,   1500.0,   "Moujalled 2018 | range Benkhaled 2022"),
        "k_dry":        (0.073,   0.060,    0.150,    "Saint-Astier BATICHANVRE data sheet, Dec. 2023 (walls, "
                                                      "350-450 kg/m³) | range Benkhaled 2022; 0.117 in Moujalled 2018"),
        "k_sat":        (0.60,    None,     None,     "Moujalled 2018 (conductivity at u_sat)"),
        "delta_p":      (5.0e-11, 1.13e-11, 1.0e-10,  "Moujalled 2018 | Bennai 2022 dry cup; Aït Ouméziane 2016"),
        "u_sat":        (1.5,     None,     None,     "Moujalled 2018 [kg/kg]"),
        "vg_alpha_ads": (0.0119,  None,     None,     "Moujalled 2018 [1/m]"),
        "vg_eta_ads":   (2.0462,  None,     None,     "Moujalled 2018"),
        "vg_alpha_des": (0.0185,  None,     None,     "Moujalled 2018 [1/m]"),
        "vg_eta_des":   (1.647,   None,     None,     "Moujalled 2018"),
        "iso_branch":   ("ads",   None,     None,     "ads | des | mean (hysteresis not modelled)"),
        "iso_scale":    (1.0,     0.75,     1.25,     "±25 % on w, as Bui et al. 2025"),
    }
    p = {k: v[0] for k, v in PARAMS.items()}

    def _u(RH):
        q = _p(Hempcrete)
        ads = vg_head(RH, q["u_sat"], q["vg_alpha_ads"], q["vg_eta_ads"])
        if q["iso_branch"] == "ads":
            return ads
        des = vg_head(RH, q["u_sat"], q["vg_alpha_des"], q["vg_eta_des"])
        return des if q["iso_branch"] == "des" else 0.5 * (ads + des)

    def _du(RH):
        q = _p(Hempcrete)
        ads = dvg_head_dphi(RH, q["u_sat"], q["vg_alpha_ads"], q["vg_eta_ads"])
        if q["iso_branch"] == "ads":
            return ads
        des = dvg_head_dphi(RH, q["u_sat"], q["vg_alpha_des"], q["vg_eta_des"])
        return des if q["iso_branch"] == "des" else 0.5 * (ads + des)

    def rho(T, RH):   return _p(Hempcrete)["rho"] + 0.0 * RH
    def Cp(T, RH):    return _p(Hempcrete)["cp"] + 0.0 * RH

    def k(T, RH):
        q = _p(Hempcrete)
        u = q["iso_scale"] * Hempcrete._u(RH)                 # mass water content [kg/kg]
        return q["k_dry"] + (q["k_sat"] - q["k_dry"]) * u / q["u_sat"]

    def w(T, RH):      return _p(Hempcrete)["iso_scale"] * _p(Hempcrete)["rho"] * Hempcrete._u(RH)
    def dw_dRH(T, RH): return _p(Hempcrete)["iso_scale"] * _p(Hempcrete)["rho"] * Hempcrete._du(RH)

    def delta_p(T, RH): return _p(Hempcrete)["delta_p"] + 0.0 * RH
    def mu(T, RH):      return lib.Da(T) / Hempcrete.delta_p(T, RH)
    def Avalue(T, RH):  return 0.05
    def wf(T, RH):      return _p(Hempcrete)["u_sat"] * _p(Hempcrete)["rho"]

    def Dw(T, RH):
        return _Dw_capillary(Hempcrete.Avalue(T, RH), Hempcrete.wf(T, RH), Hempcrete.w(T, RH))

    def delta_l(T, RH):  return 0.0 * RH


class Rammed_Earth:
    """
    Rammed earth of Saint-Antoine-l'Abbaye (Chabriac 2014). Measured data: Soudani et al. (2016),
    Energy and Buildings 116, Tables 1-2; ranges: Bui et al. (2025), J. Build. Perform. Simul., Table 1.
    Thermal conductivity linear in the volumetric water content (Bui et al. 2025, Eq. 16):
        k = k_dry + k_slope·θ,   θ = w/ρ_l
    Isotherm: GAB fitted (least squares, all 8 points) to Chabriac's adsorption isotherm at 20 °C
    (Soudani et al. 2016, Table 2), as done by Bui et al. (2025). Liquid transport neglected
    (RH < 85 %, Bui et al. 2025).
    """
    PARAMS = {
        #  name        nominal   min      max      source
        "rho":       (1730.0,  1690.0,  1770.0,  "Soudani 2016 (1730 ± 40)"),
        "cp":        (625.0,   580.0,   670.0,   "Bui et al. 2025 (Chabriac data)"),
        "k_dry":     (0.77,    0.64,    0.90,    "Bui et al. 2025"),
        "k_slope":   (5.06,    4.05,    6.07,    "Bui et al. 2025 [W/(m·K) per unit θ]"),
        "mu":        (10.0,    9.4,     10.6,    "Bui et al. 2025 | 5-13 in Earth review, Sustainability 2019"),
        "gab_wm":    (5.114,   None,    None,    "GAB fit to Soudani 2016 Table 2 [kg/m³]"),
        "gab_C":     (17.957,  None,    None,    "GAB fit to Soudani 2016 Table 2"),
        "gab_K":     (0.9804,  None,    None,    "GAB fit to Soudani 2016 Table 2"),
        "iso_scale": (1.0,     0.75,    1.25,    "Bui et al. 2025: up to ±25 % on w"),
        "A":         (0.39,    None,    None,    "Soudani 2016 [kg/(m²·s^0.5)]"),
        "wf":        (259.0,   None,    None,    "Soudani 2016 [kg/m³]"),
    }
    p = {k: v[0] for k, v in PARAMS.items()}

    def rho(T, RH):   return _p(Rammed_Earth)["rho"] + 0.0 * RH
    def Cp(T, RH):    return _p(Rammed_Earth)["cp"] + 0.0 * RH

    def k(T, RH):
        q = _p(Rammed_Earth)
        return q["k_dry"] + q["k_slope"] * Rammed_Earth.w(T, RH) / lib.rhoL

    def w(T, RH):
        q = _p(Rammed_Earth)
        return q["iso_scale"] * gab(np.clip(RH, 0.0, 1.0), q["gab_wm"], q["gab_C"], q["gab_K"])

    def dw_dRH(T, RH):
        q = _p(Rammed_Earth)
        return q["iso_scale"] * dgab_dphi(np.clip(RH, 0.0, 1.0), q["gab_wm"], q["gab_C"], q["gab_K"])

    def mu(T, RH):       return _p(Rammed_Earth)["mu"] + 0.0 * RH
    def delta_p(T, RH):  return lib.Da(T) / Rammed_Earth.mu(T, RH)
    def Avalue(T, RH):   return _p(Rammed_Earth)["A"]
    def wf(T, RH):       return _p(Rammed_Earth)["wf"]

    def Dw(T, RH):
        return _Dw_capillary(Rammed_Earth.Avalue(T, RH), Rammed_Earth.wf(T, RH), Rammed_Earth.w(T, RH))

    def delta_l(T, RH):  return 0.0 * RH


# ══════════════════════════════════════════════════════════════════════════════
# Legacy versions (poster runs, October 2026)
# ══════════════════════════════════════════════════════════════════════════════

class Rammed_Earth_legacy:
    """Rammed earth as used until October 2026: lambda = 1.15 constant (above the dry range
    0.64-0.90 of Bui et al. 2025) and a 7-point isotherm in which the 76 % value of Chabriac (1.3 %)
    was placed at 86 %. Kept to reproduce those runs."""

    def rho(T, RH):   return 1730
    def Cp(T, RH):    return 648
    def k(T, RH):     return 1.15

    def w(T, RH):
        RH_t = np.array([0, 0.23, 0.43, 0.59, 0.86, 0.97, 1.0])
        w_t  = Rammed_Earth_legacy.rho(T, RH) * np.array([0, 0.003, 0.005, 0.007, 0.013, 0.061, 0.195])
        return np.interp(RH, RH_t, w_t)

    def mu(T, RH):       return 10
    def delta_p(T, RH):  return lib.Da(T) / Rammed_Earth_legacy.mu(T, RH)
    def Avalue(T, RH):   return 0.39
    def wf(T, RH):       return 337.35

    def Dw(T, RH):
        return _Dw_capillary(Rammed_Earth_legacy.Avalue(T, RH),
                             Rammed_Earth_legacy.wf(T, RH),
                             Rammed_Earth_legacy.w(T, RH))

    def delta_l(T, RH):  return 0.0


class Hempcrete_legacy:
    """Hempcrete as used until October 2026 (poster runs): composite literature values,
    lambda = 0.076 constant, 7-point linear isotherm without source. Kept to reproduce those runs."""

    def rho(T, RH):   return 450
    def Cp(T, RH):    return 1500
    def k(T, RH):     return 0.076

    def w(T, RH):
        RH_t = np.array([0, 0.23, 0.43, 0.59, 0.86, 0.97, 1.0])
        w_t  = Hempcrete_legacy.rho(T, RH) * np.array([0, 0.010, 0.025, 0.045, 0.100, 0.200, 0.350])
        return np.interp(RH, RH_t, w_t)

    def mu(T, RH):       return 5
    def delta_p(T, RH):  return lib.Da(T) / Hempcrete_legacy.mu(T, RH)
    def Avalue(T, RH):   return 0.05
    def wf(T, RH):       return 80.0

    def Dw(T, RH):
        return _Dw_capillary(Hempcrete_legacy.Avalue(T, RH),
                             Hempcrete_legacy.wf(T, RH),
                             Hempcrete_legacy.w(T, RH))

    def delta_l(T, RH):  return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Others materials
# ══════════════════════════════════════════════════════════════════════════════

class Rock_Wool:
    """
    Rock / mineral wool.
    Mineral fiber, essentially non-hygroscopic, very vapour-permeable.
    Source: ISO 10456, HAMSTAD benchmark material 1.
    """
    def rho(T, RH):   return 100          # [kg/m³]
    def Cp(T, RH):    return 840          # [J/(kg·K)]
    def k(T, RH):     return 0.035        # [W/(m·K)]

    def w(T, RH):
        RH_t = np.array([0, 0.23, 0.43, 0.59, 0.86, 0.97, 1.0])
        w_t  = Rock_Wool.rho(T, RH) * np.array([0, 0.0005, 0.001, 0.001, 0.002, 0.003, 0.010])
        return np.interp(RH, RH_t, w_t)

    def mu(T, RH):       return 1          # nearly transparent to vapour
    def delta_p(T, RH):  return lib.Da(T) / Rock_Wool.mu(T, RH)
    def Avalue(T, RH):   return 0.0        # no capillary suction
    def wf(T, RH):       return 1.0        # symbolic

    def Dw(T, RH):       return 0.0
    def delta_l(T, RH):  return 0.0


class Wood_Fiber:
    """
    Wood fibre insulation board.
    Hygroscopic insulation — good moisture buffering.
    Source: Kaemmerlen (2010), IEA Annex 55.
    """
    def rho(T, RH):   return 160
    def Cp(T, RH):    return 2100
    def k(T, RH):     return 0.038

    def w(T, RH):
        RH_t = np.array([0, 0.35, 0.50, 0.65, 0.80, 0.93, 1.0])
        w_t  = Wood_Fiber.rho(T, RH) * np.array([0, 0.05, 0.08, 0.12, 0.20, 0.35, 0.60])
        return np.interp(RH, RH_t, w_t)

    def mu(T, RH):       return 5
    def delta_p(T, RH):  return lib.Da(T) / Wood_Fiber.mu(T, RH)
    def Avalue(T, RH):   return 0.02
    def wf(T, RH):       return 96.0

    def Dw(T, RH):
        return _Dw_capillary(Wood_Fiber.Avalue(T, RH),
                             Wood_Fiber.wf(T, RH),
                             Wood_Fiber.w(T, RH))

    def delta_l(T, RH):  return 0.0


class Concrete:
    """
    Normal-weight concrete.
    Source: EN ISO 10456, Künzel (1995).
    """
    def rho(T, RH):   return 2300
    def Cp(T, RH):    return 880
    def k(T, RH):     return 2.0

    def w(T, RH):
        RH_t = np.array([0, 0.23, 0.43, 0.59, 0.86, 0.97, 1.0])
        w_t  = Concrete.rho(T, RH) * np.array([0, 0.001, 0.002, 0.003, 0.005, 0.010, 0.050])
        return np.interp(RH, RH_t, w_t)

    def mu(T, RH):       return 80
    def delta_p(T, RH):  return lib.Da(T) / Concrete.mu(T, RH)
    def Avalue(T, RH):   return 0.10
    def wf(T, RH):       return 115.0

    def Dw(T, RH):
        return _Dw_capillary(Concrete.Avalue(T, RH),
                             Concrete.wf(T, RH),
                             Concrete.w(T, RH))

    def delta_l(T, RH):  return 0.0


class Wood:
    """
    Softwood (spruce/pine), across-grain direction.
    Source: EN 13986, Siau (1984).
    """
    def rho(T, RH):   return 500
    def Cp(T, RH):    return 1600
    def k(T, RH):     return 0.13

    def w(T, RH):
        RH_t = np.array([0, 0.35, 0.50, 0.65, 0.80, 0.93, 1.0])
        w_t  = Wood.rho(T, RH) * np.array([0, 0.030, 0.060, 0.090, 0.140, 0.200, 0.280])
        return np.interp(RH, RH_t, w_t)

    def mu(T, RH):       return 40        # across grain
    def delta_p(T, RH):  return lib.Da(T) / Wood.mu(T, RH)
    def Avalue(T, RH):   return 0.0       # low capillary suction
    def wf(T, RH):       return 140.0

    def Dw(T, RH):       return 0.0
    def delta_l(T, RH):  return 0.0


class Vapor_Barrier:
    """
    Polyethylene vapour barrier (PE film, 0.2 mm thick).
    Very high vapour resistance (Sd ≈ 100 m).
    Use a thin layer (e.g. 2 mm) in the wall stack.
    Source: manufacturer data.
    """
    def rho(T, RH):   return 950
    def Cp(T, RH):    return 1800
    def k(T, RH):     return 0.38

    def w(T, RH):
        return 0.0 * np.ones_like(np.atleast_1d(RH))   # negligible

    def mu(T, RH):       return 50000     # very high → Sd = mu * e
    def delta_p(T, RH):  return lib.Da(T) / Vapor_Barrier.mu(T, RH)
    def Avalue(T, RH):   return 0.0
    def wf(T, RH):       return 0.1

    def Dw(T, RH):       return 0.0
    def delta_l(T, RH):  return 0.0


class Rain_Barrier:
    """
    Breathable roof membrane ("pare-pluie" / weather-resistive barrier), 2 mm.
    The opposite of Vapor_Barrier: stops liquid water, stays open to vapour
    (Sd = mu * e = 10 * 0.002 ≈ 0.02 m, typical of a high-performance synthetic
    breather membrane) so moisture can still dry outward through the roof.
    Negligible thermal resistance and negligible moisture storage either way.
    Generic estimate (nonwoven/film laminate) — no specific manufacturer datasheet.
    """
    def rho(T, RH):   return 300
    def Cp(T, RH):    return 1800
    def k(T, RH):     return 0.20

    def w(T, RH):
        return 0.0 * np.ones_like(np.atleast_1d(RH))   # negligible

    def mu(T, RH):       return 10        # vapour-open → Sd = mu * e ≈ 0.02 m
    def delta_p(T, RH):  return lib.Da(T) / Rain_Barrier.mu(T, RH)
    def Avalue(T, RH):   return 0.0
    def wf(T, RH):       return 0.1

    def Dw(T, RH):       return 0.0
    def delta_l(T, RH):  return 0.0


class Earth_Plaster:
    """
    Earth / clay plaster.
    Highly hygroscopic finish coat, excellent moisture buffer.
    Source: Fabbri & Morel (2014), Chabriac (2014).
    """
    def rho(T, RH):   return 1500
    def Cp(T, RH):    return 1050
    def k(T, RH):     return 0.70

    def w(T, RH):
        RH_t = np.array([0, 0.23, 0.43, 0.59, 0.86, 0.97, 1.0])
        w_t  = Earth_Plaster.rho(T, RH) * np.array([0, 0.002, 0.004, 0.006, 0.010, 0.040, 0.150])
        return np.interp(RH, RH_t, w_t)

    def mu(T, RH):       return 10
    def delta_p(T, RH):  return lib.Da(T) / Earth_Plaster.mu(T, RH)
    def Avalue(T, RH):   return 0.20
    def wf(T, RH):       return 225.0

    def Dw(T, RH):
        return _Dw_capillary(Earth_Plaster.Avalue(T, RH),
                             Earth_Plaster.wf(T, RH),
                             Earth_Plaster.w(T, RH))

    def delta_l(T, RH):  return 0.0


class Gypsum_Plaster:
    """
    Gypsum plaster.
    Source: Künzel (1995), ISO 10456.
    """
    def rho(T, RH):   return 1200
    def Cp(T, RH):    return 1000
    def k(T, RH):     return 0.60

    def w(T, RH):
        RH_t = np.array([0, 0.23, 0.43, 0.59, 0.86, 0.97, 1.0])
        w_t  = Gypsum_Plaster.rho(T, RH) * np.array([0, 0.002, 0.004, 0.007, 0.015, 0.030, 0.080])
        return np.interp(RH, RH_t, w_t)

    def mu(T, RH):       return 10
    def delta_p(T, RH):  return lib.Da(T) / Gypsum_Plaster.mu(T, RH)
    def Avalue(T, RH):   return 0.07
    def wf(T, RH):       return 96.0

    def Dw(T, RH):
        return _Dw_capillary(Gypsum_Plaster.Avalue(T, RH),
                             Gypsum_Plaster.wf(T, RH),
                             Gypsum_Plaster.w(T, RH))

    def delta_l(T, RH):  return 0.0


class Lime_Plaster:
    """
    Air-hardening lime plaster.
    Source: Lawrence et al. (2009).
    """
    def rho(T, RH):   return 1600
    def Cp(T, RH):    return 1000
    def k(T, RH):     return 0.80

    def w(T, RH):
        RH_t = np.array([0, 0.23, 0.43, 0.59, 0.86, 0.97, 1.0])
        w_t  = Lime_Plaster.rho(T, RH) * np.array([0, 0.001, 0.003, 0.005, 0.010, 0.025, 0.060])
        return np.interp(RH, RH_t, w_t)

    def mu(T, RH):       return 15
    def delta_p(T, RH):  return lib.Da(T) / Lime_Plaster.mu(T, RH)
    def Avalue(T, RH):   return 0.12
    def wf(T, RH):       return 96.0

    def Dw(T, RH):
        return _Dw_capillary(Lime_Plaster.Avalue(T, RH),
                             Lime_Plaster.wf(T, RH),
                             Lime_Plaster.w(T, RH))

    def delta_l(T, RH):  return 0.0


class Fermacell:
    """
    Fermacell gypsum-fibre board.
    Good moisture-buffering, used as interior finish or sheathing.
    Source: Fermacell technical datasheet (2023).
    """
    def rho(T, RH):   return 1150
    def Cp(T, RH):    return 1100
    def k(T, RH):     return 0.36

    def w(T, RH):
        RH_t = np.array([0, 0.23, 0.43, 0.59, 0.86, 0.97, 1.0])
        w_t  = Fermacell.rho(T, RH) * np.array([0, 0.002, 0.005, 0.008, 0.018, 0.040, 0.100])
        return np.interp(RH, RH_t, w_t)

    def mu(T, RH):       return 13
    def delta_p(T, RH):  return lib.Da(T) / Fermacell.mu(T, RH)
    def Avalue(T, RH):   return 0.04
    def wf(T, RH):       return 115.0

    def Dw(T, RH):
        return _Dw_capillary(Fermacell.Avalue(T, RH),
                             Fermacell.wf(T, RH),
                             Fermacell.w(T, RH))

    def delta_l(T, RH):  return 0.0


class BA13:
    """
    BA13 standard gypsum plasterboard (13 mm).
    Source: Placo / Saint-Gobain technical datasheet.
    """
    def rho(T, RH):   return 840
    def Cp(T, RH):    return 1000
    def k(T, RH):     return 0.25

    def w(T, RH):
        RH_t = np.array([0, 0.23, 0.43, 0.59, 0.86, 0.97, 1.0])
        w_t  = BA13.rho(T, RH) * np.array([0, 0.003, 0.006, 0.010, 0.020, 0.045, 0.120])
        return np.interp(RH, RH_t, w_t)

    def mu(T, RH):       return 10
    def delta_p(T, RH):  return lib.Da(T) / BA13.mu(T, RH)
    def Avalue(T, RH):   return 0.04
    def wf(T, RH):       return 100.0

    def Dw(T, RH):
        return _Dw_capillary(BA13.Avalue(T, RH),
                             BA13.wf(T, RH),
                             BA13.w(T, RH))

    def delta_l(T, RH):  return 0.0
