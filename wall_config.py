# -*- coding: utf-8 -*-
"""
wall_config.py
==============
Configuration dataclasses for walls, windows, thermal bridges and openings.
All user-facing building geometry is defined here and passed to RoomSimulation.

Quick reference — orientation codes
------------------------------------
    'N'     North-facing vertical wall
    'S'     South-facing vertical wall
    'E'     East-facing vertical wall
    'W'     West-facing vertical wall
    'NE'    North-East, etc.
    'roof'  Horizontal roof (receives maximum solar)
    'floor' Ground slab (no solar, adiabatic or ground temp BC)

Quick reference — mesh options
-------------------------------
    Mesh_Opt = 0   Uniform mesh (mesh_size controls element size, default 1 cm)
    Mesh_Opt = 1   Graded mesh  (finer near surfaces — recommended for thin
                   finish layers or when studying surface moisture)

Quick reference — liquid transport
------------------------------------
    liq = 0   Vapour-phase transport only (faster, sufficient for most cases)
    liq = 1   Include liquid capillary transport (needed for driving-rain or
              very high RH conditions)
"""

from dataclasses import dataclass, field
from typing import List


def hm_lewis(h_conv: float, T_ref: float = 293.15) -> float:
    """
    Convective moisture transfer coefficient from the Lewis analogy [kg/(m²·s·Pa)].

    Lewis relation (Le ≈ 1 for air/water-vapour):
        hm = h / (rho_air × Cp_air) × Mw / (R × T)

    where the first factor (h / rho_air / Cp_air) converts a HEAT transfer
    coefficient [W/(m²K)] into a MASS transfer velocity [m/s], and the
    second factor converts that velocity into a vapour-pressure-based
    permeance [kg/(m²·s·Pa)] via the ideal-gas law for water vapour.

    Parameters
    ----------
    h_conv : float   Convective heat transfer coefficient [W/(m²·K)].
                     Typical values: exterior h = 25 W/(m²K) (wind, ISO 6946),
                                     interior h = 8  W/(m²K) (still air, ISO 6946).
    T_ref  : float   Reference temperature [K]. Default 293.15 K (20 °C).

    Returns
    -------
    hm : float   Moisture transfer coefficient [kg/(m²·s·Pa)].

    Examples
    --------
    >>> hm_lewis(25.0)   # exterior surface
    1.53e-7              # kg/(m²·s·Pa)
    >>> hm_lewis(8.0)    # interior surface
    4.91e-8              # kg/(m²·s·Pa)

    Reference: Künzel (1995), Simultaneous Heat and Moisture Transport in
    Building Components, §2.3; ASHRAE Fundamentals Ch.6 (Lewis relation).
    """
    rho_air = 1.2       # [kg/m³]    dry air at ~20 °C, sea level
    Cp_air  = 1004.0    # [J/(kg·K)]
    Mw      = 18.015e-3 # [kg/mol]   molar mass of water
    R       = 8.314     # [J/(mol·K)] universal gas constant
    return h_conv * Mw / (rho_air * Cp_air * R * T_ref)


# ══════════════════════════════════════════════════════════════════════════════
# Wall
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WallConfig:
    """
    Full configuration for one opaque wall.

    Parameters
    ----------
    name : str
        Descriptive label (e.g. "South wall", "North gable").
    mat : list of str
        Layer material names, exterior → interior.
        Each name must exist in wall_layer._MAT_REGISTRY.
    emat : list of float
        Layer thicknesses [m], same order as mat.
    area : float
        Net opaque area of this wall [m²] (after subtracting windows/doors).
    orientation : str
        Cardinal direction: 'N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW',
        'roof', or 'floor'.
    tilt : float
        Surface tilt from horizontal [°].
        90° = vertical wall (default), 0° = horizontal (roof/floor).

    Mesh & physics
    --------------
    Mesh_Opt : int       0 = uniform, 1 = graded.
    liq      : int       0 = vapour only, 1 = with liquid.
    mesh_size: float     Uniform element size [m] (Mesh_Opt=0 only).

    Convective coefficients
    -----------------------
    h_ext  : float   Exterior convective heat transfer [W/(m²·K)].
                     25 W/(m²K) ≈ wind speed ~4 m/s (ISO 6946).
    hm_ext : float   Exterior vapour mass transfer [kg/(m²·s·Pa)].
    h_int  : float   Interior convective heat transfer [W/(m²·K)].
                     8 W/(m²K) for vertical walls (ISO 6946).
    hm_int : float   Interior vapour mass transfer [kg/(m²·s·Pa)].
    """
    name        : str
    mat         : List[str]
    emat        : List[float]
    area        : float
    orientation : str           = 'S'
    tilt        : float         = 90.0

    # Mesh
    Mesh_Opt    : int           = 0
    liq         : int           = 0
    mesh_size   : float         = 1e-2

    # Convective coefficients
    h_ext       : float         = 25.0
    hm_ext      : float         = 25e-9
    h_int       : float         = 8.0
    hm_int      : float         = 25e-9


def make_wall(name, area, orientation='S', layers=None, **kwargs):
    """
    Easy way to build a wall from a compact list of layers.


    Parameters
    ----------
    name        : str     Wall label.
    area        : float   Net opaque area [m²].
    orientation : str     'N','S','E','W', 'roof', 'floor', ...
    layers      : list of (str, float)
                  Each pair = (material_name, thickness_in_metres),
                  exterior layer first.
    **kwargs    : forwarded to WallConfig (Mesh_Opt, liq, mesh_size,
                  h_int, h_ext, tilt, ...).

    Example
    -------
        south = make_wall("South wall", area=18, orientation='S', layers=[
            ("Hempcrete",    0.20),   # exterior
            ("Rammed_Earth", 0.30),   # interior
        ])

    Returns
    -------
    WallConfig
    """
    layers = layers or []
    mats   = [m for m, _ in layers]
    emats  = [float(e) for _, e in layers]

    # Validate material names against the registry (skipped if it can't load).
    try:
        from wall_layer import _MAT_REGISTRY
        unknown = [m for m in mats if m not in _MAT_REGISTRY]
        if unknown:
            raise ValueError(
                f"Unknown material(s) {unknown} in wall '{name}'.\n"
                f"Available materials: {sorted(_MAT_REGISTRY)}")
    except ImportError:
        pass

    return WallConfig(name=name, mat=mats, emat=emats,
                      area=area, orientation=orientation, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Window / Glazing
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WindowConfig:
    """
    Configuration for a window or glazing element.

    Parameters
    ----------
    name        : str    Descriptive label.
    area        : float  Glazing area [m²] (frame excluded if desired).
    orientation : str    Same codes as WallConfig.
    U_value     : float  Overall U-value [W/(m²·K)].
                         Typical values: triple 0.6, double 1.1, single 5.8.
    g_value     : float  Solar heat gain coefficient (SHGC) [-].
                         Typical: 0.3–0.7.  Lower = more shading.
    shading     : float  External shading factor [0=no shade, 1=fully shaded].
                         0.3 is a reasonable summer overhang estimate.
    frame_factor: float  Fraction of total area that is glazing [-].  Default 0.7.
    """
    name        : str
    area        : float
    orientation : str   = 'S'
    U_value     : float = 1.1
    g_value     : float = 0.6
    shading     : float = 0.0
    frame_factor: float = 0.7


# ══════════════════════════════════════════════════════════════════════════════
# Thermal bridges
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ThermalBridge:
    """
    Linear or point thermal bridge.

    For a LINEAR bridge (e.g. floor/wall junction, window lintel):
        heat_loss = psi [W/(m·K)] × length [m] × ΔT

    For a POINT bridge (e.g. structural tie, anchor):
        model it as a linear bridge with length = 1 m and
        psi = chi [W/K] (the point transmittance).

    Typical ψ values (ISO 14683)
    --------------------------------
    Floor/wall junction (insulated)   : 0.10 – 0.20 W/(m·K)
    Window frame (good installation)  : 0.00 – 0.04 W/(m·K)
    Balcony slab penetration          : 0.50 – 0.80 W/(m·K)
    Corner (convex)                   : −0.05 – 0.0 W/(m·K)
    """
    name   : str
    psi    : float          # linear transmittance [W/(m·K)]
    length : float          # junction length [m]  (use 1 for point bridges)


# ══════════════════════════════════════════════════════════════════════════════
# Opening (infiltration / natural ventilation)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OpeningConfig:
    """
    Represents an uncontrolled air opening (crack, door gap, etc.).

    The air flow rate is estimated from the pressure difference:
        Q_air [m³/s] = Cd × area × sqrt(2 × ΔP / rho_air)

    For a simplified approach, specify directly the equivalent air
    change rate contribution in [1/h] via ach_contribution.
    """
    name             : str
    ach_contribution : float = 0.1   # [1/h] equivalent air changes from this opening
