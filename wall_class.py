# -*- coding: utf-8 -*-
"""
wall_class.py
=============
`Wall` — 1D coupled heat-and-moisture transfer solver for one multi-layer wall.

The model
---------
Two physical balances are solved together at every node of the wall mesh:

  • Moisture balance  (water vapour + liquid)
  • Energy balance    (heat, including latent heat carried by vapour)

The two unknowns at each node are stored in the state vector

        U = [ Pc ; T ]

where  Pc = capillary pressure [Pa]  (a strictly-monotonic stand-in for RH,
            related by Kelvin's law: RH = exp(-Pc / (rhoL·Rv·T)) )
       T  = temperature [K].

Numerical scheme
----------------
Finite differences in space, fully *implicit* in time. Each step assembles a
linear system  M · U^(n+1) = N(U^n, boundary)  and solves it with
`np.linalg.solve`. M is a 2×2 block matrix coupling the moisture and energy
equations:

        M = [[ Res1 , Res2 ],      Res1 : moisture ← Pc      Res2 : moisture ← T
             [ Res3 , Res4 ]]      Res3 : energy   ← Pc      Res4 : energy   ← T

Boundary conditions
-------------------
Robin/Neumann surface exchange with the outdoor and indoor air, through the
convective coefficients h_ext/h_int (heat) and hm_ext/hm_int (vapour). Node 0
is the EXTERIOR surface, node N-1 the INTERIOR surface.

This is the trusted reference implementation, validated against the B30
experiment in `calcul_wall_2layer.py`.
"""

import numpy as np
import library as lib


class Wall:
    """
    Parameters
    ----------
    layer : WallLayer
        Geometry, mesh, and material property evaluation for this wall.
    h_ext : float
        Convective heat transfer coefficient at exterior surface [W/(m²·K)].
    hm_ext : float
        Convective mass transfer coefficient at exterior surface [kg/(m²·s·Pa)].
    h_int : float
        Convective heat transfer coefficient at interior surface [W/(m²·K)].
    hm_int : float
        Convective mass transfer coefficient at interior surface [kg/(m²·s·Pa)].
    T_init : float or (N_tot,1) array
        Initial temperature field [K]. A scalar initialises the wall uniformly.
    RH_init : float or (N_tot,1) array
        Initial relative humidity field [-]. Same broadcasting rule.
    """

    def __init__(self, layer, h_ext, hm_ext, h_int, hm_int, T_init, RH_init):
        self.layer  = layer
        self.h_ext  = float(h_ext)
        self.hm_ext = float(hm_ext)
        self.h_int  = float(h_int)
        self.hm_int = float(hm_int)

        N = layer.N_tot
        T0  = (np.full([N, 1], T_init)  if np.isscalar(T_init)
               else np.asarray(T_init, dtype=float).reshape(N, 1))
        RH0 = (np.full([N, 1], RH_init) if np.isscalar(RH_init)
               else np.asarray(RH_init, dtype=float).reshape(N, 1))

        self.T  = T0
        self.RH = RH0
        self.Pc = lib.Pc(T0, RH0)

        self.U = np.vstack([self.Pc, self.T])   # state vector [Pc; T]

        # Result storage 
        self.StockT  = [T0.copy()]
        self.StockRH = [RH0.copy()]
        self.StockPc = [self.Pc.copy()]
        self.Stockw  = [layer.w(T0, RH0)]

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self, T_ext, RH_ext, T_int, RH_int, dt):
        """
        Advance the wall by one time step *dt* [s].

        Parameters
        ----------
        T_ext, T_int : float   Outdoor / indoor air temperature [K].
        RH_ext, RH_int: float  Outdoor / indoor relative humidity [-].
        dt           : float   Time step [s].

        Method: build the implicit system M·U = N for this step and solve it,
        then recover RH from the new (Pc, T) via Kelvin's law.
        """
        T, RH = self.T, self.RH

        # Boundary condition vector  [Pv_ext, Pv_int, T_ext, T_int]
        L = np.array([
            [lib.Pv(T_ext, RH_ext)],
            [lib.Pv(T_int, RH_int)],
            [float(T_ext)],
            [float(T_int)],
        ])

        C11, R11, R12, C22, R22, R21, k0, k1 = self._coeffs(T, RH)

        M = self._Res(dt, T, C11, R11, R12, C22, R22, R21, k0)
        N_vec = (np.dot(self._Capa(dt, C11, C22), self.U)
                 + np.dot(self._B(T), L)
                 + self._K(T, k1))

        self.U  = np.linalg.solve(M, N_vec)
        N_tot   = self.layer.N_tot
        self.Pc = self.U[:N_tot]
        self.T  = self.U[N_tot:]
        # Lower bound 1e-3 prevents log(RH)=-inf → NaN in R12/R22 coefficients
        self.RH = np.clip(
            np.exp(-self.Pc / (lib.rhoL * lib.Rv * self.T)), 1e-3, 1.0
        )

        self.StockT.append(self.T.copy())
        self.StockRH.append(self.RH.copy())
        self.StockPc.append(self.Pc.copy())
        self.Stockw.append(self.layer.w(self.T, self.RH))

    def surface_fluxes(self):
        """
        Conductive heat flux and vapour flux at both wall surfaces.

        Returns
        -------
        q_ext : float   Heat flux at exterior surface [W/m²]  (+ = into wall)
        q_int : float   Heat flux at interior surface [W/m²]  (+ = into wall)
        gv_ext : float  Vapour flux at exterior surface [kg/(m²·s)]
        gv_int : float  Vapour flux at interior surface [kg/(m²·s)]
        """
        lay = self.layer
        T, RH = self.T, self.RH
        dx  = lay.dx.flatten()
        N   = lay.N_tot

        k_f  = lay.k(T, RH).flatten()
        dp_f = lay.delta_p(T, RH).flatten()
        Pv   = lib.Pv(T, RH).flatten()
        T_f  = T.flatten()

        # Exterior surface: gradient from node 0 to node 1
        dT_e  = (T_f[1]  - T_f[0])  / dx[1]
        dPv_e = (Pv[1]   - Pv[0])   / dx[1]
        q_ext  = -k_f[0]  * dT_e  - lib.Lv * dp_f[0]  * dPv_e
        gv_ext = -dp_f[0] * dPv_e

        # Interior surface: gradient from node N-2 to node N-1
        dT_i  = (T_f[N-1]  - T_f[N-2]) / dx[N-1]
        dPv_i = (Pv[N-1]   - Pv[N-2])  / dx[N-1]
        q_int  = -k_f[N-1]  * dT_i  - lib.Lv * dp_f[N-1]  * dPv_i
        gv_int = -dp_f[N-1] * dPv_i

        return float(q_ext), float(q_int), float(gv_ext), float(gv_int)

    def mid_state(self):
        """
        Temperature [°C] and RH [%] at the node closest to the wall midpoint.
        Useful for quick monitoring without re-reading the full StockT arrays.
        """
        mid_pos = self.layer.total_thickness / 2.0
        mid_idx = int(np.argmin(np.abs(self.layer.x_pos - mid_pos)))
        return float(self.T[mid_idx, 0]) - 273.15, float(self.RH[mid_idx, 0]) * 100.0

    # ── Matrix assembly ───────────────────────────────────────────────────────

    def _coeffs(self, T, RH):
        """
        Compute all finite-difference coefficients from the current T and RH.

        These are the (temperature- and humidity-dependent) coefficients of the
        coupled PDEs, evaluated node by node:

            C11  moisture storage capacity        (∂w/∂Pc)
            R11  moisture transport driven by Pc  (vapour + liquid)
            R12  moisture transport driven by T
            C22  heat storage capacity            (ρ·Cp + w·CpL)
            R22  heat transport driven by T       (conduction + latent)
            R21  heat transport driven by Pc      (enthalpy carried by moisture)
            k0,k1  linearisation terms of Pv(Pc, T) used in the surface BCs

        `liq = 0` → vapour transport only; `liq = 1` adds liquid (capillary) flow.
        """
        lay = self.layer
        liq = lay.liq
        RH  = np.maximum(RH, 1e-3)   # guard: prevents log(RH)→-inf when RH≈0

        C11 = lay.Xi_Pc(T, RH)

        if liq == 0:
            R11 = -(lay.delta_l(T, RH)
                    + lay.delta_p(T, RH) * lib.rhoV(T, RH) / lib.rhoL)
            R12 = lay.delta_p(T, RH) * (
                RH * lib.dPsat(T, RH) - lib.Pv(T, RH) * np.log(RH) / T
            )
        else:
            R11 = -(lay.delta_p(T, RH) * lib.rhoV(T, RH) / lib.rhoL
                    - np.log(RH) * lay.Dw(T, RH) / lib.Pc(T, RH) * lay.Xi_RH(T, RH))
            R12 = lay.delta_p(T, RH) * (
                (RH * lib.dPsat(T, RH) - lib.Pv(T, RH) * np.log(RH) / T)
                - np.log(RH) * lay.Dw(T, RH) / T * lay.Xi_RH(T, RH)
            )

        C22 = lay.rho(T, RH) * lay.Cp(T, RH) + lay.w(T, RH) * lib.CpL
        R22 = lay.k(T, RH) + lib.Lv * lay.delta_p(T, RH) * (
            RH * lib.dPsat(T, RH) - lib.Pv(T, RH) * np.log(RH) / T
        )
        R21 = (lay.delta_l(T, RH) * lib.CpL * T
               - (lib.CpV * (T - 273.15) + lib.Lv)
               * lay.delta_p(T, RH) * lib.rhoV(T, RH) / lib.rhoL)

        k0 = (-lib.Psat(T, RH) / (lib.rhoL * lib.Rv * T)
              * np.exp(-lib.Pc(T, RH) / (lib.rhoL * lib.Rv * T)))
        k1 = (lib.Psat(T, RH)
              * np.exp(-lib.Pc(T, RH) / (lib.rhoL * lib.Rv * T))
              * (1 + lib.Pc(T, RH) / (lib.rhoL * lib.Rv * T)))

        return C11, R11, R12, C22, R22, R21, k0, k1

    def _Res(self, dt, T, C11, R11, R12, C22, R22, R21, k0):
        N  = self.layer.N_tot
        dx = self.layer.dx

        Res1 = np.zeros([N, N])
        Res2 = np.zeros([N, N])
        Res3 = np.zeros([N, N])
        Res4 = np.zeros([N, N])

        for i in range(1, N - 1):
            dxc = dx[i] + dx[i + 1]

            Res1[i, i+1] = -(R11[i]+R11[i+1]) / (dx[i+1]*dxc)
            Res1[i, i]   = (R11[i+1]+R11[i])/(dx[i+1]*dxc) + (R11[i-1]+R11[i])/(dx[i]*dxc) + C11[i]/dt
            Res1[i, i-1] = -(R11[i]+R11[i-1]) / (dx[i]*dxc)

            Res2[i, i+1] = -(R12[i]+R12[i+1]) / (dx[i+1]*dxc)
            Res2[i, i]   = (R12[i+1]+R12[i])/(dx[i+1]*dxc) + (R12[i-1]+R12[i])/(dx[i]*dxc)
            Res2[i, i-1] = -(R12[i]+R12[i-1]) / (dx[i]*dxc)

            Res3[i, i+1] = -(R21[i]+R21[i+1]) / (dx[i+1]*dxc)
            Res3[i, i]   = (R21[i+1]+R21[i])/(dx[i+1]*dxc) + (R21[i-1]+R21[i])/(dx[i]*dxc)
            Res3[i, i-1] = -(R21[i]+R21[i-1]) / (dx[i]*dxc)

            Res4[i, i+1] = -(R22[i]+R22[i+1]) / (dx[i+1]*dxc)
            Res4[i, i]   = (R22[i+1]+R22[i])/(dx[i+1]*dxc) + (R22[i-1]+R22[i])/(dx[i]*dxc) + C22[i]/dt
            Res4[i, i-1] = -(R22[i]+R22[i-1]) / (dx[i]*dxc)

        # Neumann boundary conditions — exterior (row 0) and interior (row N-1)
        Res1[0, 1]     = -2*R11[0]/dx[1]**2
        Res1[0, 0]     =  2*R11[0]/dx[1]**2 + C11[0]/dt + 2*self.hm_ext*k0[0]/dx[1]
        Res1[N-1,N-2]  = -2*R11[N-1]/dx[N-1]**2
        Res1[N-1,N-1]  =  2*R11[N-1]/dx[N-1]**2 + C11[N-1]/dt + 2*self.hm_int*k0[N-1]/dx[N-1]

        Res2[0, 1]     = -2*R12[0]/dx[1]**2
        Res2[0, 0]     =  2*R12[0]/dx[1]**2
        Res2[N-1,N-2]  = -2*R12[N-1]/dx[N-1]**2
        Res2[N-1,N-1]  =  2*R12[N-1]/dx[N-1]**2

        Res3[0, 1]     = -2*R21[0]/dx[1]**2
        Res3[0, 0]     =  2*R21[0]/dx[1]**2 + 2*(lib.CpV*(T[0]-273.15)+lib.Lv)*self.hm_ext*k0[0]/dx[1]
        Res3[N-1,N-2]  = -2*R21[N-1]/dx[N-1]**2
        Res3[N-1,N-1]  =  2*R21[N-1]/dx[N-1]**2 + 2*(lib.CpV*(T[N-1]-273.15)+lib.Lv)*self.hm_int*k0[N-1]/dx[N-1]

        Res4[0, 1]     = -2*R22[0]/dx[1]**2
        Res4[0, 0]     =  2*R22[0]/dx[1]**2 + C22[0]/dt + 2*self.h_ext/dx[1]
        Res4[N-1,N-2]  = -2*R22[N-1]/dx[N-1]**2
        Res4[N-1,N-1]  =  2*R22[N-1]/dx[N-1]**2 + C22[N-1]/dt + 2*self.h_int/dx[N-1]

        return np.hstack((np.vstack((Res1, Res3)), np.vstack((Res2, Res4))))

    def _Capa(self, dt, C11, C22):
        N     = self.layer.N_tot
        Capa1 = (C11/dt).flatten() * np.eye(N)
        Capa4 = (C22/dt).flatten() * np.eye(N)
        zeros = np.zeros([N, N])
        return np.hstack((np.vstack((Capa1, zeros)), np.vstack((zeros, Capa4))))

    def _B(self, T):
        N  = self.layer.N_tot
        dx = self.layer.dx
        B  = np.zeros([2*N, 4])

        B[0,     0] = 2 * self.hm_ext / dx[1]
        B[N-1,   1] = 2 * self.hm_int / dx[N-1]
        B[N,     0] = 2 * (lib.CpV*(T[0]-273.15)+lib.Lv) * self.hm_ext / dx[1]
        B[2*N-1, 1] = 2 * (lib.CpV*(T[N-1]-273.15)+lib.Lv) * self.hm_int / dx[N-1]
        B[N,     2] = 2 * self.h_ext / dx[1]
        B[2*N-1, 3] = 2 * self.h_int / dx[N-1]

        return B

    def _K(self, T, k1):
        N  = self.layer.N_tot
        dx = self.layer.dx
        K  = np.zeros([2*N, 1])

        K[0]     = -2 * self.hm_ext * k1[0]   / dx[1]
        K[N-1]   = -2 * self.hm_int * k1[N-1] / dx[N-1]
        K[N]     = -2 * (lib.CpV*(T[0]-273.15)+lib.Lv)   * self.hm_ext * k1[0]   / dx[1]
        K[2*N-1] = -2 * (lib.CpV*(T[N-1]-273.15)+lib.Lv) * self.hm_int * k1[N-1] / dx[N-1]

        return K

    # ── String representation ─────────────────────────────────────────────────

    def __repr__(self):
        T_mid, RH_mid = self.mid_state()
        return (f"Wall('{self.layer.label}', "
                f"T_mid={T_mid:.1f}°C, RH_mid={RH_mid:.1f}%)")
