# -*- coding: utf-8 -*-
"""
wall_layer.py
=============
The wall as a stack of material layers: its mesh, and the material properties at every node.

    layer = WallLayer(["Hempcrete", "Earth_Plaster"], [0.30, 0.02], Mesh_Opt=2)
    k = layer.k(T, RH)          # conductivity at each node, shape (N_tot, 1)

Nodes are numbered from the outer surface (node 0) to the inner surface (node N_tot - 1).
A node on an interface belongs to the layer on its outer side.

Materials are looked up by name in _MAT_REGISTRY. To add one, write its class in
materials_library.py and add it to the registry below (or call register_material).
"""

import numpy as np
import library as lib
from materials_library import (
    Rammed_Earth, Hempcrete, Rammed_Earth_legacy, Hempcrete_legacy,
    Rock_Wool, Wood_Fiber, Concrete, Wood,
    Vapor_Barrier, Rain_Barrier, Earth_Plaster, Gypsum_Plaster,
    Lime_Plaster, Fermacell, BA13,
)


_MAT_REGISTRY = {
    "Rammed_Earth":        Rammed_Earth,
    "Hempcrete":           Hempcrete,
    "Rammed_Earth_legacy": Rammed_Earth_legacy,
    "Hempcrete_legacy":    Hempcrete_legacy,
    "Lime_Plaster":        Lime_Plaster,
    "Fermacell":           Fermacell,
    "BA13":                BA13,
    "Earth_Plaster":       Earth_Plaster,
    "Vapor_Barrier":       Vapor_Barrier,
    "Rain_Barrier":        Rain_Barrier,
    "Gypsum_Plaster":      Gypsum_Plaster,
    "Rock_Wool":           Rock_Wool,
    "Wood_Fiber":          Wood_Fiber,
    "Concrete":            Concrete,
    "Wood":                Wood,
}


def register_material(name, cls):
    """Make a material class usable by name (used by the validation cases)."""
    _MAT_REGISTRY[name] = cls


def apply_material_params(overrides):
    """
    Set the property values of the materials that carry a PARAMS table (Hempcrete, Rammed_Earth).

    overrides is {material: {parameter: value}}, where value is a number (or a text for text
    parameters such as iso_branch), or one of "nominal", "min", "max". For example
        {"Hempcrete": {"k_dry": "max", "iso_branch": "mean"}}

    All these materials are first put back to their nominal values, so a run never inherits
    the settings of a previous one. A number outside the literature range is accepted with a
    warning. Returns the values in force, {material: {parameter: value}}.
    """
    with_params = {n: c for n, c in _MAT_REGISTRY.items() if hasattr(c, "PARAMS")}
    for cls in with_params.values():
        cls.p = {k: v[0] for k, v in cls.PARAMS.items()}

    for name, params in (overrides or {}).items():
        if name not in with_params:
            raise ValueError(f"MATERIAL_PARAMS: '{name}' has no PARAMS table "
                             f"(materials with ranges: {list(with_params)})")
        cls = with_params[name]
        for key, val in params.items():
            if key not in cls.PARAMS:
                raise ValueError(f"MATERIAL_PARAMS: unknown parameter '{key}' for {name}. "
                                 f"Available: {list(cls.PARAMS)}")
            nominal, lo, hi, _source = cls.PARAMS[key]
            if isinstance(val, str) and val in ("nominal", "min", "max"):
                val = {"nominal": nominal, "min": lo, "max": hi}[val]
                if val is None:
                    raise ValueError(f"MATERIAL_PARAMS: {name}.{key} has no range in the literature table")
            elif isinstance(val, (int, float)) and lo is not None and not (lo <= val <= hi):
                print(f"  [!] {name}.{key} = {val} is outside the literature range [{lo}, {hi}]")
            cls.p[key] = val

    return {n: dict(c.p) for n, c in with_params.items()}


class WallLayer:
    """
    Layers and mesh of one wall.

    mat       : material names, outside -> inside
    emat      : layer thicknesses [m], same order
    Mesh_Opt  : 0 = uniform elements of mesh_size over the whole wall
                1 = elements growing geometrically from each face of every layer (fine near
                    the surfaces, where the gradients are steep)
                2 = equal elements inside each layer, about mesh_size, at least 2 per layer:
                    every layer keeps its exact thickness, even a thin one
    liq       : 0 = vapour transport only, 1 = vapour + liquid (capillary) transport
    mesh_size : target element size [m] for options 0 and 2
    label     : name shown in the messages
    """

    def __init__(self, mat, emat, Mesh_Opt=0, liq=0, mesh_size=1e-2, label=""):
        if len(mat) != len(emat):
            raise ValueError("mat and emat must have the same length.")
        for name in mat:
            if name not in _MAT_REGISTRY:
                raise ValueError(f"Material '{name}' not in registry. Available: {list(_MAT_REGISTRY)}.")

        self.mat       = list(mat)
        self.emat      = np.array(emat, dtype=float)
        self.Nb_layer  = len(mat)
        self.Mesh_Opt  = Mesh_Opt
        self.liq       = liq
        self.mesh_size = mesh_size
        self.label     = label or " / ".join(mat)

        # dx[i]  : distance between node i-1 and node i (dx[0] = 0)
        # N_sum  : index of the first node of each layer (N_sum[-1] = N_tot)
        self.dx, self.N_nodes, self.N_tot, self.N_sum = self._build_mesh()

        self.x_pos           = np.cumsum(self.dx)                       # node positions [m]
        self.layer_bounds    = np.concatenate([[0.0], np.cumsum(self.emat)])
        self.total_thickness = float(self.emat.sum())

    # ── Mesh ──────────────────────────────────────────────────────────────────

    def _build_mesh(self):
        if self.Mesh_Opt == 0:
            n = [int(round(e / self.mesh_size)) for e in self.emat]
            steps = [np.full(k, self.mesh_size) for k in n]
        elif self.Mesh_Opt == 1:
            n, steps = [], []
            for e in self.emat:
                s = self._graded_steps(e)
                n.append(len(s)); steps.append(s)
        elif self.Mesh_Opt == 2:
            n = [max(2, int(round(e / self.mesh_size))) for e in self.emat]
            steps = [np.full(k, e / k) for k, e in zip(n, self.emat)]
        else:
            raise ValueError(f"Unknown Mesh_Opt={self.Mesh_Opt}: use 0 (uniform), 1 (graded) or 2 (per layer).")

        dx = np.concatenate([[0.0]] + steps).reshape(-1, 1)
        N_nodes = np.array(n, dtype=float).reshape(-1, 1)
        N_nodes[0] += 1                                   # node 0 sits on the outer surface
        N_sum = np.concatenate([[0.0], np.cumsum(N_nodes)]).reshape(-1, 1)
        return dx, N_nodes, len(dx), N_sum

    @staticmethod
    def _graded_steps(e, u0=5e-4, q=1.10, tol=2):
        """
        Element sizes for one layer of thickness e, symmetric about its middle: they start at u0
        on each face and grow by the ratio q towards the centre. The gap left in the middle is
        shared between the central elements; their number is increased until a central element
        is less than tol times its neighbour, so that the sizes never jump abruptly.
        """
        n = 2 * int(np.log(1 - e / (2 * u0) * (1 - q)) / np.log(q) - 1)
        j, m = 1, 2
        while True:
            half = np.zeros(n)
            half[0] = u0
            for k in range(n // 2 - j - 1):
                half[k + 1] = q * half[k]
            gap = e - 2 * half.sum()
            for l in range(1, j + 1):
                half[n // 2 - l] = gap / m
            if half[n // 2 - j] >= tol * half[n // 2 - j - 1]:
                n += 2; j += 1; m *= 2
            else:
                return half + half[::-1]

    # ── Material properties at the nodes ──────────────────────────────────────

    def _fill(self, prop, T, RH):
        """Evaluate the property `prop` of each layer on its own nodes."""
        out = np.zeros([self.N_tot, 1])
        for i in range(self.Nb_layer):
            s, e = int(self.N_sum[i, 0]), int(self.N_sum[i + 1, 0])
            out[s:e] = getattr(_MAT_REGISTRY[self.mat[i]], prop)(T[s:e], RH[s:e])
        return out

    def rho(self, T, RH):      return self._fill("rho",     T, RH)   # density [kg/m³]
    def Cp(self, T, RH):       return self._fill("Cp",      T, RH)   # heat capacity [J/(kg·K)]
    def k(self, T, RH):        return self._fill("k",       T, RH)   # conductivity [W/(m·K)]
    def w(self, T, RH):        return self._fill("w",       T, RH)   # water content [kg/m³]
    def delta_p(self, T, RH):  return self._fill("delta_p", T, RH)   # vapour permeability [kg/(m·s·Pa)]
    def delta_l(self, T, RH):  return self._fill("delta_l", T, RH)   # liquid permeability [kg/(m·s·Pa)]
    def Dw(self, T, RH):       return self._fill("Dw",      T, RH)   # liquid diffusivity [m²/s]

    def Xi_RH(self, T, RH, h=1e-5):
        """Moisture capacity dw/dRH [kg/m³]: from the material when it gives it, else by central difference."""
        out = np.zeros([self.N_tot, 1])
        for i in range(self.Nb_layer):
            s, e = int(self.N_sum[i, 0]), int(self.N_sum[i + 1, 0])
            mat = _MAT_REGISTRY[self.mat[i]]
            if hasattr(mat, "dw_dRH"):
                out[s:e] = mat.dw_dRH(T[s:e], RH[s:e])
            else:
                out[s:e] = (mat.w(T[s:e], RH[s:e] + h) - mat.w(T[s:e], RH[s:e] - h)) / (2 * h)
        return out

    def Xi_Pc(self, T, RH):
        """Moisture capacity dw/dPc = dw/dRH · dRH/dPc, with dRH/dPc = −RH/(rhoL·Rv·T) (Kelvin law)."""
        return self.Xi_RH(T, RH) * (-RH / (lib.rhoL * lib.Rv * T))

    def U_value(self):
        """Conductance of the layers at 20 °C / 50 % [W/(m²·K)], without surface resistances."""
        R = sum(e / _MAT_REGISTRY[m].k(293.15, 0.5) for m, e in zip(self.mat, self.emat))
        return 1.0 / R if R > 0 else float("inf")

    def __repr__(self):
        layers = ", ".join(f"{m} ({e*100:.1f} cm)" for m, e in zip(self.mat, self.emat))
        return f"WallLayer([{layers}], N_tot={self.N_tot}, U={self.U_value():.2f} W/(m²K))"
