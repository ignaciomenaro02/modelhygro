# -*- coding: utf-8 -*-
"""
wall_config.py
==============
Descriptions of the building elements handed to RoomSimulation: opaque walls, windows,
thermal bridges and air leaks. They only hold data; the physics is in wall_class.py and
room_simulation.py.

Orientation codes: 'N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW', 'roof' (horizontal, sunlit)
and 'floor' (no sun).
"""

from dataclasses import dataclass
from typing import List


@dataclass
class WallConfig:
    """
    One opaque wall. Build it with make_wall() rather than directly.

    Layers are listed from the outside to the inside, `mat` holding the material names
    (keys of wall_layer._MAT_REGISTRY) and `emat` the thicknesses [m].
    """
    name        : str
    mat         : List[str]
    emat        : List[float]
    area        : float                 # net opaque area [m²]
    orientation : str   = "S"
    tilt        : float = 90.0          # 90° = vertical wall, 0° = horizontal roof

    # Mesh (see wall_layer.WallLayer): 0 = uniform, 1 = refined near the surfaces,
    # 2 = equal elements inside each layer
    Mesh_Opt    : int   = 0
    mesh_size   : float = 1e-2          # target element size [m]
    liq         : int   = 0             # 0 = vapour transport only, 1 = vapour + liquid

    # Surface exchange coefficients
    h_ext       : float = 25.0          # exterior heat transfer [W/(m²·K)]
    hm_ext      : float = 25e-9         # exterior vapour transfer [kg/(m²·s·Pa)]
    h_int       : float = 8.0           # interior heat transfer [W/(m²·K)]
    hm_int      : float = 25e-9         # interior vapour transfer [kg/(m²·s·Pa)]

    absorptance : float = 0.6           # solar absorptance of the outer face (light 0.3, dark 0.9)

    # False = dry wall, heat conduction only (as in dynamic thermal simulation tools).
    # Use it together with hm_ext = hm_int ≈ 0.
    hygro       : bool  = True


def make_wall(name, area, orientation="S", layers=None, **kwargs):
    """
    Build a WallConfig from a list of (material, thickness [m]) pairs, outside first:

        make_wall("South", area=12.5, orientation="S",
                  layers=[("Hempcrete", 0.30), ("Earth_Plaster", 0.02)])

    Other keyword arguments (h_int, Mesh_Opt, hygro, ...) are passed to WallConfig.
    Unknown material names stop the program with the list of the available ones.
    """
    from wall_layer import _MAT_REGISTRY

    layers = layers or []
    mats   = [m for m, _ in layers]
    emats  = [float(e) for _, e in layers]

    unknown = [m for m in mats if m not in _MAT_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown material(s) {unknown} in wall '{name}'.\n"
                         f"Available materials: {sorted(_MAT_REGISTRY)}")

    return WallConfig(name=name, mat=mats, emat=emats, area=area, orientation=orientation, **kwargs)


@dataclass
class WindowConfig:
    """
    A window. Heat losses U·A·ΔT; solar gains g·(1 − shading)·frame_factor·A·I.

    U_value      : W/(m²·K)  (triple glazing 0.6, double 1.1, single 5.8)
    g_value      : solar factor of the glazing [-]
    shading      : 0 = no shading, 1 = fully shaded
    frame_factor : glazed fraction of the window area [-]
    """
    name        : str
    area        : float
    orientation : str   = "S"
    U_value     : float = 1.1
    g_value     : float = 0.6
    shading     : float = 0.0
    frame_factor: float = 0.7


@dataclass
class ThermalBridge:
    """
    Linear thermal bridge: losses psi·length·ΔT. A point bridge is entered with length = 1 m
    and psi = its point transmittance chi [W/K].
    Typical psi (ISO 14683): insulated floor/wall junction 0.10-0.20, window frame 0-0.04,
    balcony slab 0.5-0.8 W/(m·K).
    """
    name   : str
    psi    : float          # [W/(m·K)]
    length : float          # [m]


@dataclass
class OpeningConfig:
    """Uncontrolled air leak (cracks, door gaps), given directly as an air change rate."""
    name             : str
    ach_contribution : float = 0.1   # [1/h]
