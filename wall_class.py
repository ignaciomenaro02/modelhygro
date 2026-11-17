# -*- coding: utf-8 -*-
"""
wall_class.py
=============
Coupled heat and moisture transfer through one multi-layer wall, in one dimension.

Unknowns
--------
At every node of the mesh the state is the capillary pressure Pc [Pa] and the temperature T [K],
stored in one vector U = [Pc ; T]. Pc replaces the relative humidity because it stays smooth
across the interface between two materials; the two are linked by the Kelvin law
RH = exp(−Pc / (rhoL·Rv·T)).

Equations
---------
    moisture :  C11·∂Pc/∂t = ∂/∂x( R11·∂Pc/∂x + R12·∂T/∂x )
    energy   :  C22·∂T/∂t  = ∂/∂x( R22·∂T/∂x  + R21·∂Pc/∂x )

C11 and C22 are the moisture and heat capacities, R11 and R22 the direct transport coefficients,
R12 and R21 the cross terms (vapour driven by temperature, latent heat carried by vapour).
All of them depend on the local T and RH and are listed in _coeffs.

Boundary conditions
-------------------
Node 0 is the outer surface, node N−1 the inner surface. Each surface exchanges heat
(h·ΔT) and vapour (hm·ΔPv) with the air beside it; the vapour also carries its enthalpy
(cp,v·T + Lv).

Time scheme
-----------
Finite differences in space, implicit Euler in time. The coefficients are taken at the start
of the step and one linear system M·U(n+1) = N is solved per step:

    M = [ Res1  Res2 ]    Res1: moisture ← Pc     Res2: moisture ← T
        [ Res3  Res4 ]    Res3: energy   ← Pc     Res4: energy   ← T
"""

import numpy as np
import library as lib


class Wall:
    """
    layer           : WallLayer (mesh and materials)
    h_ext, h_int    : surface heat transfer coefficients [W/(m²·K)]
    hm_ext, hm_int  : surface vapour transfer coefficients [kg/(m²·s·Pa)]
    T_init, RH_init : initial temperature [K] and relative humidity [-], a number (uniform wall)
                      or one value per node
    hygro           : False = dry wall, heat conduction only (see the end of _coeffs)
    """

    def __init__(self, layer, h_ext, hm_ext, h_int, hm_int, T_init, RH_init, hygro=True):
        self.layer  = layer
        self.hygro  = bool(hygro)
        self.h_ext  = float(h_ext)
        self.hm_ext = float(hm_ext)
        self.h_int  = float(h_int)
        self.hm_int = float(hm_int)

        N = layer.N_tot
        self.T  = np.broadcast_to(np.asarray(T_init, dtype=float).reshape(-1, 1), (N, 1)).copy()
        self.RH = np.broadcast_to(np.asarray(RH_init, dtype=float).reshape(-1, 1), (N, 1)).copy()
        self.Pc = lib.Pc(self.T, self.RH)
        self.U  = np.vstack([self.Pc, self.T])

        # History of the stored steps (one per hour in the room simulation)
        self.StockT  = [self.T.copy()]
        self.StockRH = [self.RH.copy()]
        self.StockPc = [self.Pc.copy()]

    def step(self, T_ext, RH_ext, T_int, RH_int, dt, Pv_ext=None, store=True):
        """
        Advance the wall by dt [s].

        T_ext, T_int   : temperature of the outdoor (or sol-air) and indoor air [K]
        RH_ext, RH_int : relative humidity of the outdoor and indoor air [-]
        Pv_ext         : outdoor vapour pressure [Pa]. Give it when T_ext is a sol-air temperature:
                         RH_ext belongs to the outdoor air, not to the sol-air temperature, so
                         Pv(T_ext, RH_ext) would be too high in the sun.
        store          : keep the new state in the history (False for the intermediate sub-steps)
        """
        T, RH = self.T, self.RH

        # Surrounding conditions: [Pv_ext, Pv_int, T_ext, T_int]
        L = np.array([[lib.Pv(T_ext, RH_ext) if Pv_ext is None else float(Pv_ext)],
                      [lib.Pv(T_int, RH_int)],
                      [float(T_ext)],
                      [float(T_int)]])

        C11, R11, R12, C22, R22, R21, k0, k1 = self._coeffs(T, RH)
        M = self._matrix(dt, T, C11, R11, R12, C22, R22, R21, k0)
        rhs = (np.vstack([C11, C22]) / dt * self.U
               + self._B(T) @ L
               + self._K(T, k1))

        self.U  = np.linalg.solve(M, rhs)
        N       = self.layer.N_tot
        self.Pc = self.U[:N]
        self.T  = self.U[N:]
        # RH from Kelvin; the floor at 1e-3 keeps log(RH) finite in the next coefficients
        self.RH = np.clip(np.exp(-self.Pc / (lib.rhoL * lib.Rv * self.T)), 1e-3, 1.0)

        if store:
            self.StockT.append(self.T.copy())
            self.StockRH.append(self.RH.copy())
            self.StockPc.append(self.Pc.copy())

    # ── Coefficients ──────────────────────────────────────────────────────────

    def _coeffs(self, T, RH):
        """
        Coefficients of the two equations at every node, from the current T and RH:

            C11  moisture capacity ∂w/∂Pc
            R11  moisture flow driven by ∇Pc (vapour, plus liquid when liq = 1)
            R12  moisture flow driven by ∇T  (vapour pressure rises with T)
            C22  heat capacity ρ·cp + w·cp,liquid
            R22  heat flow driven by ∇T      (conduction + latent heat of the vapour)
            R21  heat flow driven by ∇Pc     (latent heat of the vapour flow)
            k0, k1  linearisation Pv ≈ k0·Pc + k1 at the surface nodes, for the vapour exchange
        """
        lay = self.layer
        RH  = np.maximum(RH, 1e-3)

        dp   = lay.delta_p(T, RH)
        dl   = lay.delta_l(T, RH)
        rhoV = lib.rhoV(T, RH)
        # ∂Pv/∂T at constant Pc, divided by delta_p later: RH·dPsat/dT − Pv·ln(RH)/T
        dPv_dT = RH * lib.dPsat(T, RH) - lib.Pv(T, RH) * np.log(RH) / T

        C11 = lay.Xi_Pc(T, RH)

        if lay.liq == 0:
            R11 = -(dl + dp * rhoV / lib.rhoL)
            R12 = dp * dPv_dT
        else:
            # Liquid flow −Dw·∂w/∂x, with w(RH) and the Kelvin law:
            #   ∂RH/∂Pc = RH·ln(RH)/Pc     ∂RH/∂T (Pc constant) = −RH·ln(RH)/T
            # (checked against HAMSTAD benchmark 2, see validation/hamstad_bm2.py)
            Dw_xi = lay.Dw(T, RH) * lay.Xi_RH(T, RH)
            R11 = -(dp * rhoV / lib.rhoL - RH * np.log(RH) * Dw_xi / lib.Pc(T, RH))
            R12 = dp * (dPv_dT - RH * np.log(RH) * Dw_xi / T)

        # Energy balance of Bui et al. (2025, J. Build. Perform. Simul., eq. 5): conduction plus the
        # latent heat of the vapour flow. The sensible heat carried by the vapour (cp,v·T) is left out
        # inside the wall (about 1.5 % of the latent term) but kept in the surface exchange (eq. 12).
        C22 = lay.rho(T, RH) * lay.Cp(T, RH) + lay.w(T, RH) * lib.CpL
        R22 = lay.k(T, RH) + lib.Lv * dp * dPv_dT
        R21 = dl * lib.CpL * T - lib.Lv * dp * rhoV / lib.rhoL

        # Surface vapour pressure linearised around the current state: Pv = k0·Pc + k1
        a  = lib.rhoL * lib.Rv * T
        e  = lib.Psat(T, RH) * np.exp(-lib.Pc(T, RH) / a)
        k0 = -e / a
        k1 = e * (1 + lib.Pc(T, RH) / a)

        if not self.hygro:
            # Dry wall, as in dynamic thermal simulation: conduction with the dry heat capacity.
            # The moisture equation still runs but no longer acts on the temperature.
            R12 = np.zeros_like(R12)
            R21 = np.zeros_like(R21)
            R22 = lay.k(T, RH)
            C22 = lay.rho(T, RH) * lay.Cp(T, RH)

        return C11, R11, R12, C22, R22, R21, k0, k1

    # ── Linear system ─────────────────────────────────────────────────────────

    def _diffusion(self, R):
        """
        Finite-difference matrix of −∂/∂x(R·∂u/∂x) on the non-uniform mesh, R averaged between
        neighbouring nodes. The surface rows use a half cell (hence the factor 2); the surface
        exchange terms are added in _matrix.
        """
        N  = self.layer.N_tot
        dx = self.layer.dx.ravel()
        A  = np.zeros([N, N])

        i   = np.arange(1, N - 1)
        dxc = dx[i] + dx[i + 1]
        up  = (R[i] + R[i + 1]) / (dx[i + 1] * dxc)
        lo  = (R[i] + R[i - 1]) / (dx[i] * dxc)
        A[i, i + 1] = -up
        A[i, i - 1] = -lo
        A[i, i]     = up + lo

        A[0, 0], A[0, 1]                 =  2 * R[0] / dx[1]**2,       -2 * R[0] / dx[1]**2
        A[N - 1, N - 1], A[N - 1, N - 2] =  2 * R[N - 1] / dx[N - 1]**2, -2 * R[N - 1] / dx[N - 1]**2
        return A

    def _enthalpy_v(self, T):
        """Enthalpy of the vapour leaving or entering each surface [J/kg]."""
        return lib.CpV * (T - 273.15) + lib.Lv

    def _matrix(self, dt, T, C11, R11, R12, C22, R22, R21, k0):
        N  = self.layer.N_tot
        dx = self.layer.dx.ravel()
        T, C11, R11, R12, C22, R22, R21, k0 = (np.ravel(a) for a in (T, C11, R11, R12, C22, R22, R21, k0))
        hv = self._enthalpy_v(T)
        o, n = 0, N - 1                                   # outer and inner surface nodes
        so, si = 2 / dx[1], 2 / dx[N - 1]                 # half-cell factors

        Res1 = self._diffusion(R11) + np.diag(C11 / dt)
        Res2 = self._diffusion(R12)
        Res3 = self._diffusion(R21)
        Res4 = self._diffusion(R22) + np.diag(C22 / dt)

        # Surface exchanges: vapour hm·(Pv_surface − Pv_air), with Pv_surface = k0·Pc + k1,
        # its enthalpy in the energy balance, and heat h·(T_surface − T_air)
        Res1[o, o] += so * self.hm_ext * k0[o]
        Res1[n, n] += si * self.hm_int * k0[n]
        Res3[o, o] += so * hv[o] * self.hm_ext * k0[o]
        Res3[n, n] += si * hv[n] * self.hm_int * k0[n]
        Res4[o, o] += so * self.h_ext
        Res4[n, n] += si * self.h_int

        return np.block([[Res1, Res2], [Res3, Res4]])

    def _B(self, T):
        """Right-hand side terms from the surrounding air, B · [Pv_ext, Pv_int, T_ext, T_int]."""
        N  = self.layer.N_tot
        dx = self.layer.dx.ravel()
        hv = self._enthalpy_v(np.ravel(T))
        so, si = 2 / dx[1], 2 / dx[N - 1]
        B  = np.zeros([2 * N, 4])
        B[0,         0] = so * self.hm_ext
        B[N - 1,     1] = si * self.hm_int
        B[N,         0] = so * hv[0] * self.hm_ext
        B[2 * N - 1, 1] = si * hv[N - 1] * self.hm_int
        B[N,         2] = so * self.h_ext
        B[2 * N - 1, 3] = si * self.h_int
        return B

    def _K(self, T, k1):
        """Constant part k1 of the linearised surface vapour pressure, moved to the right-hand side."""
        N  = self.layer.N_tot
        dx = self.layer.dx.ravel()
        hv, k1 = self._enthalpy_v(np.ravel(T)), np.ravel(k1)
        so, si = 2 / dx[1], 2 / dx[N - 1]
        K  = np.zeros([2 * N, 1])
        K[0]         = -so * self.hm_ext * k1[0]
        K[N - 1]     = -si * self.hm_int * k1[N - 1]
        K[N]         = -so * hv[0] * self.hm_ext * k1[0]
        K[2 * N - 1] = -si * hv[N - 1] * self.hm_int * k1[N - 1]
        return K

    def __repr__(self):
        mid = self.layer.N_tot // 2
        return (f"Wall('{self.layer.label}', T_mid={self.T[mid, 0] - 273.15:.1f}°C, "
                f"RH_mid={self.RH[mid, 0] * 100:.1f}%)")
