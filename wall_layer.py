# -*- coding: utf-8 -*-
"""
wall_layer.py
=============
WallLayer — geometry, mesh, and material property evaluation.
The mesh is built directly inside each WallLayer instance (no global mesh.py).

    Usage
    -----
        layer = WallLayer(
            mat      = ["Hempcrete", "Rammed_Earth"],
            emat     = [0.20, 0.30],
            Mesh_Opt = 0,
            liq      = 0,
            label    = "South wall",
    )
    k_field = layer.k(T, RH)    # (N_tot, 1) array

Adding a new material
---------------------
1. Create its class in materials_library.py following the existing pattern.
2. Import it here and add it to _MAT_REGISTRY.
"""

import numpy as np
import library as lib
from materials_library import (
    Rammed_Earth, Hempcrete,
    Rock_Wool, Wood_Fiber, Concrete, Wood,
    Vapor_Barrier, Earth_Plaster, Gypsum_Plaster,
    Lime_Plaster, Fermacell, BA13,
)


# ── Material registry ──────────────────────────────────────────────────────────
_MAT_REGISTRY = {
    "Rammed_Earth":   Rammed_Earth,
    "Hempcrete":      Hempcrete,
    "Rock_Wool":      Rock_Wool,
    "Wood_Fiber":     Wood_Fiber,
    "Concrete":       Concrete,
    "Wood":           Wood,
    "Vapor_Barrier":  Vapor_Barrier,
    "Earth_Plaster":  Earth_Plaster,
    "Gypsum_Plaster": Gypsum_Plaster,
    "Lime_Plaster":   Lime_Plaster,
    "Fermacell":      Fermacell,
    "BA13":           BA13,
}


def register_material(name, cls):
    """Register a new material class so WallLayer can use it by name."""
    _MAT_REGISTRY[name] = cls


def available_materials():
    """Return the list of registered material names."""
    return list(_MAT_REGISTRY.keys())


# ── WallLayer class ────────────────────────────────────────────────────────────

class WallLayer:
    """
    Defines the layered composition and mesh of a single wall.

    Parameters
    ----------
    mat : list of str
        Material name for each layer (exterior → interior).
        Must match a key in _MAT_REGISTRY.
    emat : list of float
        Thickness [m] of each layer (same order as mat).
    Mesh_Opt : int
        0 = uniform mesh (default)
        1 = graded mesh (refined near surfaces — better for transient)
    liq : int
        Liquid transport:  0 = vapour only,  1 = Dw formulation.
    mesh_size : float
        Uniform element size [m] used when Mesh_Opt == 0.  Default 1 cm.
    label : str
        Human-readable name shown in plots and print statements.
    """

    def __init__(self, mat, emat, Mesh_Opt=0, liq=0, mesh_size=1e-2, label=""):
        if len(mat) != len(emat):
            raise ValueError("mat and emat must have the same length.")
        for name in mat:
            if name not in _MAT_REGISTRY:
                raise ValueError(
                    f"Material '{name}' not in registry. "
                    f"Available: {available_materials()}. "
                    f"Use register_material() to add new ones."
                )

        self.mat       = list(mat)
        self.emat      = np.array(emat, dtype=float)
        self.Nb_layer  = len(mat)
        self.Mesh_Opt  = Mesh_Opt
        self.liq       = liq
        self.mesh_size = mesh_size
        self.label     = label or " / ".join(mat)

        self.dx, self.N_nodes, self.N_tot, self.N_sum = self._build_mesh()
        self.dx_sum = self._compute_dx_sum()

        self.x_pos           = self.dx_sum.flatten()
        self.interface_pos   = np.cumsum(self.emat)
        self.layer_bounds    = np.concatenate([[0.0], self.interface_pos])
        self.total_thickness = float(self.emat.sum())

    # ── Mesh construction ──────────────────────────────────────────────────────

    def _build_mesh(self):
        if self.Mesh_Opt == 0:
            return self._uniform_mesh()
        if self.Mesh_Opt == 1:
            return self._refined_mesh()
        raise ValueError(f"Unknown Mesh_Opt={self.Mesh_Opt}. Use 0 (uniform) or 1 (graded).")

    def _uniform_mesh(self):
        size = self.mesh_size
        N_nodes = np.zeros([self.Nb_layer, 1])
        for i in range(self.Nb_layer):
            N_nodes[i] = round(self.emat[i] / size)

        N_tot_inner = int(N_nodes.sum())
        dx = np.full([N_tot_inner, 1], size)
        dx = np.vstack(([[0.0]], dx))
        N_nodes[0] += 1
        N_tot = len(dx)

        N_sum = np.zeros([self.Nb_layer + 1, 1])
        for i in range(self.Nb_layer):
            N_sum[i + 1] = N_nodes[i] + N_sum[i]

        return dx, N_nodes, N_tot, N_sum

    def _refined_mesh(self):
        u0  = 5e-4
        q   = 1.10
        Tol = 2

        N_nodes = np.zeros([self.Nb_layer, 1])
        dx_list = None

        for i in range(self.Nb_layer):
            N_nodes[i] = 2 * int(
                np.log(1 - self.emat[i] / (2 * u0) * (1 - q)) / np.log(q) - 1
            )
            j, m, converged = 1, 2, False

            while not converged:
                n = int(N_nodes[i])
                dx = np.zeros([n, 1])
                dx[0] = u0
                for k in range(n // 2 - j - 1):
                    dx[k + 1] = q * dx[k]
                res = self.emat[i] - 2 * dx.sum()
                for l in range(1, j + 1):
                    dx[n // 2 - l] = res / m

                if dx[n // 2 - j] >= Tol * dx[n // 2 - j - 1]:
                    N_nodes[i] += 2
                    j += 1
                    m *= 2
                else:
                    converged = True

                dx = dx + np.flipud(dx)

            dx_list = dx if dx_list is None else np.vstack((dx_list, dx))

        N_nodes[0] += 1
        N_tot  = len(dx_list) + 1
        dx_out = np.vstack(([[0.0]], dx_list))

        N_sum = np.zeros([self.Nb_layer + 1, 1])
        for i in range(self.Nb_layer):
            N_sum[i + 1] = N_nodes[i] + N_sum[i]

        return dx_out, N_nodes, N_tot, N_sum

    def _compute_dx_sum(self):
        ds = np.zeros([int(self.N_tot), 1])
        for i in range(len(self.dx)):
            ds[i] = ds[i - 1] + self.dx[i]
        return ds

    # ── Material property evaluation ───────────────────────────────────────────

    def _fill(self, prop, T, RH):
        out = np.zeros([self.N_tot, 1])
        for i in range(self.Nb_layer):
            s = int(self.N_sum[i])
            e = int(self.N_sum[i + 1])
            out[s:e] = getattr(_MAT_REGISTRY[self.mat[i]], prop)(T[s:e], RH[s:e])
        return out

    def rho(self, T, RH):      return self._fill("rho",     T, RH)
    def Cp(self, T, RH):       return self._fill("Cp",      T, RH)
    def k(self, T, RH):        return self._fill("k",       T, RH)
    def w(self, T, RH):        return self._fill("w",       T, RH)
    def delta_p(self, T, RH):  return self._fill("delta_p", T, RH)
    def delta_l(self, T, RH):  return self._fill("delta_l", T, RH)
    def Dw(self, T, RH):       return self._fill("Dw",      T, RH)

    def Xi_Pc(self, T, RH):
        return (
            lib.partial_derivative(self.w, T, RH, var="y")
            * np.exp(-lib.Pc(T, RH) / (lib.rhoL * lib.Rv * T))
            / (-lib.rhoL * lib.Rv * T)
        )

    def Xi_RH(self, T, RH):
        return lib.partial_derivative(self.w, T, RH, var="y")

    # ── Convenience ────────────────────────────────────────────────────────────

    def U_value(self):
        """Static U-value [W/(m²·K)] — sum of layer resistances (no surface Rs)."""
        R = sum(e / _MAT_REGISTRY[m].k(293.15, 0.5)
                for m, e in zip(self.mat, self.emat))
        return 1.0 / R if R > 0 else float('inf')

    def __repr__(self):
        layers = ", ".join(
            f"{m} ({e*100:.1f} cm)" for m, e in zip(self.mat, self.emat)
        )
        return (f"WallLayer([{layers}], "
                f"N_tot={self.N_tot}, Mesh_Opt={self.Mesh_Opt}, "
                f"U={self.U_value():.2f} W/(m²K))")
