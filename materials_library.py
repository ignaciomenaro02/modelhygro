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
- Rammed_Earth  : Chabriac (2014), Saint-Antoine l'Abbaye
- Hempcrete     : Literature composite values
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


# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════

class Rammed_Earth:
    """Rammed earth — Saint-Antoine l'Abbaye (Chabriac 2014)."""

    def rho(T, RH):   return 1730
    def Cp(T, RH):    return 648
    def k(T, RH):     return 1.15

    def w(T, RH):
        RH_t = np.array([0, 0.23, 0.43, 0.59, 0.86, 0.97, 1.0])
        w_t  = Rammed_Earth.rho(T, RH) * np.array([0, 0.003, 0.005, 0.007, 0.013, 0.061, 0.195])
        return np.interp(RH, RH_t, w_t)

    def mu(T, RH):       return 10
    def delta_p(T, RH):  return lib.Da(T) / Rammed_Earth.mu(T, RH)
    def Avalue(T, RH):   return 0.39
    def wf(T, RH):       return 337.35

    def Dw(T, RH):
        return _Dw_capillary(Rammed_Earth.Avalue(T, RH),
                             Rammed_Earth.wf(T, RH),
                             Rammed_Earth.w(T, RH))

    def delta_l(T, RH):  return 0.0


class Hempcrete:
    """Hempcrete — composite literature values."""

    def rho(T, RH):   return 450
    def Cp(T, RH):    return 1500
    def k(T, RH):     return 0.12

    def w(T, RH):
        RH_t = np.array([0, 0.23, 0.43, 0.59, 0.86, 0.97, 1.0])
        w_t  = Hempcrete.rho(T, RH) * np.array([0, 0.010, 0.025, 0.045, 0.100, 0.200, 0.350])
        return np.interp(RH, RH_t, w_t)

    def mu(T, RH):       return 5
    def delta_p(T, RH):  return lib.Da(T) / Hempcrete.mu(T, RH)
    def Avalue(T, RH):   return 0.05
    def wf(T, RH):       return 80.0

    def Dw(T, RH):
        return _Dw_capillary(Hempcrete.Avalue(T, RH),
                             Hempcrete.wf(T, RH),
                             Hempcrete.w(T, RH))

    def delta_l(T, RH):  return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# New materials
# ══════════════════════════════════════════════════════════════════════════════

class Rock_Wool:
    """
    Laine de roche (rock / mineral wool).
    Mineral fiber — essentially non-hygroscopic, very vapour-permeable.
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
    Laine / panneau de fibres de bois (wood fibre insulation board).
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
    Béton normal (normal-weight concrete).
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
    Bois résineux (softwood — spruce/pine, across-grain direction).
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
    Pare-vapeur polyéthylène (PE film, 0.2 mm thick).
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


class Earth_Plaster:
    """
    Enduit de terre (earth / clay plaster).
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
    Enduit de plâtre (gypsum plaster / enduit plâtre).
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
    Enduit de chaux (lime plaster — aérien).
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
    Panneau Fermacell (gypsum-fibre board).
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
    Plaque de plâtre BA13 (standard gypsum plasterboard).
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
