# -*- coding: utf-8 -*-
"""
materials_library.py
====================
Hygrothermal properties of the wall materials. One class per material; each property is a
function of temperature T [K] and relative humidity RH [-], called without an instance
(Hempcrete.k(T, RH)):

    rho(T, RH)      Dry density              [kg/m³]
    Cp(T, RH)       Heat capacity            [J/(kg·K)]
    k(T, RH)        Thermal conductivity     [W/(m·K)]
    w(T, RH)        Water content (sorption) [kg/m³]
    dw_dRH(T, RH)   Slope of w (optional: central difference in wall_layer.py otherwise) [kg/m³]
    mu(T, RH)       Vapour resistance factor [-]
    delta_p(T, RH)  Vapour permeability      [kg/(m·s·Pa)]
    Avalue(T, RH)   Liquid absorption coeff  [kg/(m²·s^0.5)]
    wf(T, RH)       Free-saturation content  [kg/m³]
    Dw(T, RH)       Liquid diffusion coeff   [m²/s]
    delta_l(T, RH)  Liquid permeability      [kg/(m·s·Pa)]

Avalue and wf only feed Dw, and Dw only enters the wall solver with liquid transport (liq = 1).

Status of the values
--------------------
- Hempcrete, Rammed_Earth : values checked against the tables of the references below.
- *_legacy                : the two materials as used until October 2026 (poster runs), kept to
                            reproduce those runs.
- All other materials     : generic values, compared in November 2026 with CSTB (2012) and manufacturer
                            data; provenance stated class by class. "Not traced" means that no table
                            giving the value could be found. Regulatory defaults and declared values
                            are not measurements. Avalue and wf are not sourced, and the sorption curves
                            are not measured data except for Concrete, Earth_Plaster and BA13.

Parameter ranges
----------------
Hempcrete and Rammed_Earth carry PARAMS = {name: (nominal, min, max, source)} and read the values in
force from the class dict `p` (nominal by default). Change them with
wall_layer.apply_material_params({"Hempcrete": {"k_dry": "max", "delta_p": 2e-11}}), which is what
simulation.py does with its MATERIAL_PARAMS setting.

Adding a new material
---------------------
Copy any class, adjust the values, then register it with wall_layer.register_material("MyMaterial",
MyMaterial), or import it in wall_layer.py and add it to _MAT_REGISTRY.

References
----------
Aït Oumeziane, Y., Moissette, S., Bart, M., Lanos, C. (2016). Influence of temperature on sorption
    process in hemp concrete. Construction and Building Materials 106, 600-607.
Benkhaled, M., Ouldboukhitine, S.-E., Bakkour, A., Amziane, S. (2022). Sensitivity analysis of the
    parameters for assessing a hygrothermal transfer model HAM in bio-based hemp concrete material.
    International Communications in Heat and Mass Transfer 132, 105884.
Bennai, F., Ferroukhi, M.Y., Benmahiddine, F., Belarbi, R., Nouviaire, A. (2022). Assessment of
    hygrothermal performance of hemp concrete compared to conventional building materials at overall
    building scale. Construction and Building Materials 316, 126007.
Bui, R., Goffart, J., McGregor, F., Fabbri, A., Woloszyn, M., Grillet, A.-C. (2025). Hygrothermal
    behaviour of a rammed earth wall subjected to realistic conditions: investigation of the most
    influential parameters. Journal of Building Performance Simulation.
Bui, R. (2018). Towards a realistic estimation of the walls moisture buffering in an occupied room.
    PhD thesis, Toulouse. Tables 3.1-3.3 and 4.2-4.4 compile literature values; their original sources
    (Künzel et al. 2008, Delgado et al. 2013, McGregor et al. 2009, Laborel-Préneron et al. 2018) are
    quoted here as cited by Bui, not checked directly.
Chabriac, P.-A. (2014). Mesure du comportement hygrothermique du pisé. PhD thesis, ENTPE, Université
    de Lyon.
CSTB (2012). Règles Th-U, fascicule 2 : Matériaux. Réglementation thermique 2012. The rules were revised
    since (Règles Th-bât, April 2026): check the current edition before quoting them.
Colinart, T., Glouannec, P., Pierre, T., Chauvelon, P., Magueresse, A. (2013). Experimental study on
    the hygrothermal behavior of a coated sprayed hemp concrete wall. Buildings 3, 79-99.
Delgado, J.M.P.Q., Barreira, E., Ramos, N.M.M., de Freitas, V.P. (2013). Hygrothermal Numerical
    Simulation Tools Applied to Building Physics. Springer.
Giuffrida, G., Caponetto, R., Nocera, F. (2019). Hygrothermal properties of raw earth materials: a
    literature review. Sustainability 11, 5342.
Kaemmerlen, A. (2009). Transfert de chaleur à travers les isolants thermiques du bâtiment. PhD thesis,
    Université Henri Poincaré, Nancy.
Künzel, H.M. (1995). Simultaneous Heat and Moisture Transport in Building Components. Fraunhofer IRB
    Verlag, Stuttgart.
Moujalled, B., Aït Ouméziane, Y., Moissette, S., Bart, M., Lanos, C., Samri, D. (2018). Experimental
    and numerical evaluation of the hygrothermal performance of a hemp lime concrete building: a long
    term case study. Building and Environment 136, 11-27.
Siau, J.F. (1984). Transport Processes in Wood. Springer.
Soudani, L., Fabbri, A., Morel, J.-C., Woloszyn, M., Chabriac, P.-A., Wong, H., Grillet, A.-C. (2016).
    Assessment of the validity of some common assumptions in hygrothermal modeling of earth based
    materials. Energy and Buildings 116, 498-511.
"""

import numpy as np
import library as lib


# ══════════════════════════════════════════════════════════════════════════════
# Shared formulas
# ══════════════════════════════════════════════════════════════════════════════

def _Dw_capillary(Avalue, wf, w):
    """Liquid diffusion coefficient from the A-value [m²/s] (Krus approximation, see Künzel 1995)."""
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


T_ISO = 296.15                              # reference temperature of vg_head [K], 23 °C (Moujalled 2018)
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


# ══════════════════════════════════════════════════════════════════════════════
# Bio- and geo-sourced materials of the thesis (sourced values and ranges)
# ══════════════════════════════════════════════════════════════════════════════

class Hempcrete:
    """
    Hemp-lime concrete (~450 kg/m³). Nominal values: Moujalled et al. (2018), Tables 2-3, which take
    them from Aït Oumeziane et al. (2016); 450 kg/m³ is the manufacturer density quoted by Moujalled.
    Exception: dry thermal conductivity 0.073 W/(m·K), declared value of the Saint-Astier BATICHANVRE
    data sheet (Dec. 2023) for walls (350-450 kg/m³); this data sheet is not in the bibliography folder.
    Ranges: see PARAMS. For comparison, Colinart et al. (2013) quote 300-600 kg/m³ and 0.07-0.2 W/(m·K).
    Thermal conductivity linear in the mass water content (Moujalled 2018, Table 2):
        k = k_dry + (k_sat - k_dry)·u/u_sat
    Isotherm: van Genuchten in capillary head (Moujalled 2018, Eq. 5), main adsorption or desorption
    curve (no hysteresis model): iso_branch = "ads" | "des" | "mean".
    delta_l = 0: the liquid transport of Moujalled (2018) is not included. Avalue has no identified
    source; wf = u_sat·rho.
    """
    PARAMS = {
        #  name           nominal   min       max       source
        "rho":          (450.0,   300.0,    550.0,    "Moujalled 2018 | min Aït Oumeziane 2016 (300-500); max Benkhaled 2022 (450-550)"),
        "cp":           (1250.0,  1000.0,   1500.0,   "Moujalled 2018 | range Benkhaled 2022"),
        "k_dry":        (0.073,   0.060,    0.150,    "Saint-Astier BATICHANVRE data sheet, Dec. 2023 (walls, "
                                                      "350-450 kg/m³) | max Benkhaled 2022 (0.05-0.15); min 0.060 not "
                                                      "traced; 0.117 in Moujalled 2018"),
        "k_sat":        (0.60,    None,     None,     "Moujalled 2018 (conductivity at u_sat)"),
        "delta_p":      (5.0e-11, 1.13e-11, 1.0e-10,  "Moujalled 2018 | min Bennai 2022 (dry cup); max Aït Oumeziane 2016 (1e-11 to 1e-10)"),
        "u_sat":        (1.5,     None,     None,     "Moujalled 2018 [kg/kg]"),
        "vg_alpha_ads": (0.0119,  None,     None,     "Moujalled 2018 [1/m]"),
        "vg_eta_ads":   (2.0462,  None,     None,     "Moujalled 2018"),
        "vg_alpha_des": (0.0185,  None,     None,     "Moujalled 2018 [1/m]"),
        "vg_eta_des":   (1.647,   None,     None,     "Moujalled 2018"),
        "iso_branch":   ("ads",   None,     None,     "ads | des | mean (hysteresis not modelled)"),
        "iso_scale":    (1.0,     0.75,     1.25,     "±25 % on w, by analogy with rammed earth (Bui et al. 2025)"),
    }
    p = {k: v[0] for k, v in PARAMS.items()}

    def _u(RH):
        q = Hempcrete.p
        ads = vg_head(RH, q["u_sat"], q["vg_alpha_ads"], q["vg_eta_ads"])
        if q["iso_branch"] == "ads":
            return ads
        des = vg_head(RH, q["u_sat"], q["vg_alpha_des"], q["vg_eta_des"])
        return des if q["iso_branch"] == "des" else 0.5 * (ads + des)

    def _du(RH):
        q = Hempcrete.p
        ads = dvg_head_dphi(RH, q["u_sat"], q["vg_alpha_ads"], q["vg_eta_ads"])
        if q["iso_branch"] == "ads":
            return ads
        des = dvg_head_dphi(RH, q["u_sat"], q["vg_alpha_des"], q["vg_eta_des"])
        return des if q["iso_branch"] == "des" else 0.5 * (ads + des)

    def rho(T, RH):   return Hempcrete.p["rho"] + 0.0 * RH
    def Cp(T, RH):    return Hempcrete.p["cp"] + 0.0 * RH

    def k(T, RH):
        q = Hempcrete.p
        u = q["iso_scale"] * Hempcrete._u(RH)                 # mass water content [kg/kg]
        return q["k_dry"] + (q["k_sat"] - q["k_dry"]) * u / q["u_sat"]

    def w(T, RH):      return Hempcrete.p["iso_scale"] * Hempcrete.p["rho"] * Hempcrete._u(RH)
    def dw_dRH(T, RH): return Hempcrete.p["iso_scale"] * Hempcrete.p["rho"] * Hempcrete._du(RH)

    def delta_p(T, RH): return Hempcrete.p["delta_p"] + 0.0 * RH
    def mu(T, RH):      return lib.Da(T) / Hempcrete.delta_p(T, RH)
    def Avalue(T, RH):  return 0.05
    def wf(T, RH):      return Hempcrete.p["u_sat"] * Hempcrete.p["rho"]

    def Dw(T, RH):
        return _Dw_capillary(Hempcrete.Avalue(T, RH), Hempcrete.wf(T, RH), Hempcrete.w(T, RH))

    def delta_l(T, RH):  return 0.0 * RH


class Rammed_Earth:
    """
    Rammed earth of Saint-Antoine-l'Abbaye (Chabriac 2014). Measured data: Soudani et al. (2016),
    Tables 1-2; ranges: Bui et al. (2025), Table 1.
    Thermal conductivity linear in the volumetric water content (Bui et al. 2025, Eq. 16):
        k = k_dry + k_slope·θ,   θ = w/ρ_l
    Isotherm: GAB fitted (least squares, all 8 points) to the adsorption isotherm at 20 °C of Soudani
    et al. (2016), Table 2. Bui et al. (2025) use the same model, on their own measurements.
    delta_l = 0: liquid transport neglected. Bui et al. (2025) keep a constant 1e-15 kg/(m·s·Pa), as
    liquid transfer may only predominate above 85 % RH.
    """
    PARAMS = {
        #  name        nominal   min      max      source
        "rho":       (1730.0,  1690.0,  1770.0,  "Soudani 2016 (1730 ± 40)"),
        "cp":        (625.0,   580.0,   670.0,   "Bui et al. 2025 (Chabriac data)"),
        "k_dry":     (0.77,    0.64,    0.90,    "Bui et al. 2025"),
        "k_slope":   (5.06,    4.05,    6.07,    "Bui et al. 2025 [W/(m·K) per unit θ]"),
        "mu":        (10.0,    9.4,     10.6,    "Bui et al. 2025 | 5-13 in Giuffrida et al. 2019"),
        "gab_wm":    (5.114,   None,    None,    "GAB fit to Soudani 2016 Table 2 [kg/m³]"),
        "gab_C":     (17.957,  None,    None,    "GAB fit to Soudani 2016 Table 2"),
        "gab_K":     (0.9804,  None,    None,    "GAB fit to Soudani 2016 Table 2"),
        "iso_scale": (1.0,     0.75,    1.25,    "Bui et al. 2025: w varies by up to 25 % (shown for RH 0-80 %)"),
        "A":         (0.39,    None,    None,    "Soudani 2016 [kg/(m²·s^0.5)]"),
        "wf":        (259.0,   None,    None,    "Soudani 2016 [kg/m³]"),
    }
    p = {k: v[0] for k, v in PARAMS.items()}

    def rho(T, RH):   return Rammed_Earth.p["rho"] + 0.0 * RH
    def Cp(T, RH):    return Rammed_Earth.p["cp"] + 0.0 * RH

    def k(T, RH):
        q = Rammed_Earth.p
        return q["k_dry"] + q["k_slope"] * Rammed_Earth.w(T, RH) / lib.rhoL

    def w(T, RH):
        q = Rammed_Earth.p
        return q["iso_scale"] * gab(np.clip(RH, 0.0, 1.0), q["gab_wm"], q["gab_C"], q["gab_K"])

    def dw_dRH(T, RH):
        q = Rammed_Earth.p
        return q["iso_scale"] * dgab_dphi(np.clip(RH, 0.0, 1.0), q["gab_wm"], q["gab_C"], q["gab_K"])

    def mu(T, RH):       return Rammed_Earth.p["mu"] + 0.0 * RH
    def delta_p(T, RH):  return lib.Da(T) / Rammed_Earth.mu(T, RH)
    def Avalue(T, RH):   return Rammed_Earth.p["A"]
    def wf(T, RH):       return Rammed_Earth.p["wf"]

    def Dw(T, RH):
        return _Dw_capillary(Rammed_Earth.Avalue(T, RH), Rammed_Earth.wf(T, RH), Rammed_Earth.w(T, RH))

    def delta_l(T, RH):  return 0.0 * RH


# ══════════════════════════════════════════════════════════════════════════════
# Legacy versions (poster runs, October 2026)
# ══════════════════════════════════════════════════════════════════════════════

class Rammed_Earth_legacy:
    """Rammed earth as used until October 2026: lambda = 1.15 constant (above the dry range
    0.64-0.90 of Bui et al. 2025) and a 7-point isotherm in which the 76 % value of Soudani et al.
    2016, Table 2 (1.3 %) was placed at 86 %. Kept to reproduce those runs."""

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


class Rock_Wool:
    """
    Rock / mineral wool.
    Mineral fibre, essentially non-hygroscopic, very vapour-permeable.
    k, Cp, mu: CSTB (2012), §2.6.2.1, class 100-125 kg/m³ (default value: products with a certified
    conductivity can be lower, e.g. 0.039 at 120-145 kg/m³ for Rockwool). Sorption curve: not traced.
    """
    def rho(T, RH):   return 100          # [kg/m³]
    def Cp(T, RH):    return 1030         # [J/(kg·K)]
    def k(T, RH):     return 0.044        # [W/(m·K)]

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
    rho, Cp, k, mu: declared values of a 160 kg/m³ board (STEICOtherm data sheet). Sorption curve: not
    traced. Kaemmerlen (2009) deals with heat transfer in insulation, not with these moisture values.
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
    rho, k, Cp, mu: CSTB (2012), §2.2.1.1, 2300-2600 kg/m³ (mu: wet value, 130 when dry). For comparison,
    Bui (2018): 2300 kg/m³, 850 J/(kg·K), 1.6 W/(m·K), mu ≈ 180.
    Sorption: Bui (2018), Table 3.2 (from Künzel et al. 2008); the RH = 1 point extrapolates the last
    segment, and wf is set to it so that w/wf ≤ 1.
    """
    def rho(T, RH):   return 2300
    def Cp(T, RH):    return 1000
    def k(T, RH):     return 2.0

    def w(T, RH):
        RH_t = np.array([0, 0.20, 0.40, 0.60, 0.80, 0.95, 0.97, 1.0])
        w_t  = np.array([0, 24.5, 34.1, 40.4, 64.6, 108.6, 126.1, 152.35])   # [kg/m³]
        return np.interp(RH, RH_t, w_t)

    def mu(T, RH):       return 80
    def delta_p(T, RH):  return lib.Da(T) / Concrete.mu(T, RH)
    def Avalue(T, RH):   return 0.10
    def wf(T, RH):       return 152.35

    def Dw(T, RH):
        return _Dw_capillary(Concrete.Avalue(T, RH),
                             Concrete.wf(T, RH),
                             Concrete.w(T, RH))

    def delta_l(T, RH):  return 0.0


class Wood:
    """
    Softwood (spruce/pine), across-grain direction.
    k, Cp, mu: CSTB (2012), §2.5.1, softwood of 435-520 kg/m³ (mu: dry value, 20 when wet). Sorption curve:
    not traced. Siau (1984) is the general reference on transport in wood; these values were not checked
    against it. EN 13986 is not a source: it covers wood-based panels, not solid wood.
    """
    def rho(T, RH):   return 500
    def Cp(T, RH):    return 1600
    def k(T, RH):     return 0.15

    def w(T, RH):
        RH_t = np.array([0, 0.35, 0.50, 0.65, 0.80, 0.93, 1.0])
        w_t  = Wood.rho(T, RH) * np.array([0, 0.030, 0.060, 0.090, 0.140, 0.200, 0.280])
        return np.interp(RH, RH_t, w_t)

    def mu(T, RH):       return 50
    def delta_p(T, RH):  return lib.Da(T) / Wood.mu(T, RH)
    def Avalue(T, RH):   return 0.0       # capillary suction neglected
    def wf(T, RH):       return 140.0

    def Dw(T, RH):       return 0.0
    def delta_l(T, RH):  return 0.0


class Vapor_Barrier:
    """
    Polyethylene vapour barrier, modelled as a thin layer of the wall stack.
    Sd = mu·e = 100 m for the 2 mm layer, as declared for real 0.2 mm films (Sd > 100 m), so mu is an
    effective value for that thickness. rho, Cp, k: generic, between the LDPE and HDPE values of CSTB
    (2012), §2.9 (920-980 kg/m³, 1800-2200 J/(kg·K), 0.33-0.50 W/(m·K)).
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
    Provenance: Sd as declared for such membranes (e.g. Rothoblaas TRASPIR 75: Sd = 0.02 m, 0.3 mm, about
    250 kg/m³), so mu is an effective value for 2 mm. Cp and k: generic estimates, not traced.
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
    rho, k, Cp, mu, sorption: Bui (2018), Tables 4.2-4.3, representative of a 2 cm plaster (from McGregor
    et al. 2009 and Laborel-Préneron et al. 2018); mu = 10 comes from delta_v = 1.97e-11 s. The RH = 1
    point extrapolates the last segment. Values depend strongly on the mix: DIN 18947 class 1.8 plasters
    declare about 1800 kg/m³, 0.91 W/(m·K), mu = 5/10. Avalue and wf: not traced.
    """
    def rho(T, RH):   return 1660
    def Cp(T, RH):    return 800
    def k(T, RH):     return 0.57

    def w(T, RH):
        RH_t = np.array([0, 0.23, 0.43, 0.59, 0.75, 0.85, 0.97, 1.0])
        w_t  = np.array([0, 3.77, 5.74, 7.42, 10.20, 13.70, 31.76, 36.275])   # [kg/m³]
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
    rho, Cp, k, mu close to CSTB (2012), §2.3.1, common plaster of 1000-1300 kg/m³ (0.57 W/(m·K),
    1000 J/(kg·K), mu = 10 dry, 4 wet). Sorption curve: not traced.
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
    rho, Cp, k, mu: CSTB (2012), §2.9.2, mortar of 1450-1600 kg/m³ (mu: dry value, 6 when wet); rho, Cp
    and k also agree with Delgado et al. (2013), Table 2.2. Sorption curve and Avalue: not traced. The
    reference "Lawrence et al. (2009)" could not be identified.
    """
    def rho(T, RH):   return 1600
    def Cp(T, RH):    return 1000
    def k(T, RH):     return 0.80

    def w(T, RH):
        RH_t = np.array([0, 0.23, 0.43, 0.59, 0.86, 0.97, 1.0])
        w_t  = Lime_Plaster.rho(T, RH) * np.array([0, 0.001, 0.003, 0.005, 0.010, 0.025, 0.060])
        return np.interp(RH, RH_t, w_t)

    def mu(T, RH):       return 10
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
    rho, Cp, k, mu: manufacturer data (1150 ± 50 kg/m³, 1.1 kJ/(kg·K), 0.32 W/(m·K), mu = 13). Sorption
    curve: not traced.
    """
    def rho(T, RH):   return 1150
    def Cp(T, RH):    return 1100
    def k(T, RH):     return 0.32

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
    rho, Cp, k, mu: CSTB (2012), §2.3.1, plasterboard of 750-900 kg/m³ (mu: dry value, 4 when wet).
    Sorption: gypsum board of Bui (2018), Table 3.2 (from Delgado et al. 2013); the RH = 1 point
    extrapolates the last segment. Avalue and wf: not traced.
    """
    def rho(T, RH):   return 840
    def Cp(T, RH):    return 1000
    def k(T, RH):     return 0.25

    def w(T, RH):
        RH_t = np.array([0, 0.20, 0.40, 0.60, 0.80, 0.95, 0.97, 1.0])
        w_t  = np.array([0, 1.5, 2.2, 3.2, 4.4, 5.5, 6.7, 8.5])   # [kg/m³]
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
