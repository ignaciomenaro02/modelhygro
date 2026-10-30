# -*- coding: utf-8 -*-
"""
hamstad_bm2.py
==============
HAMSTAD benchmark 2 — isothermal drying of a homogeneous 200 mm wall (Hagentoft 2002, report R-02:9).

Set-up (as described in the "Better Building" HAMSTAD test report, §4):
    material A: rho = 525 kg/m3, c = 800 J/(kg.K), lambda = 0.15 W/(m.K)
                w(phi) = 121.4 / (1 + (-ln(phi) / 0.129)^0.907)   [kg/m3]  (reproduces the report's table A.2)
                delta_p = 1e-15 kg/(m.s.Pa), D_w = 6e-10 m2/s (constant)
    initial state: 20 °C, RH = 95 %
    boundaries: 20 °C, RH = 45 % (x = 0) and 65 % (x = 0.2 m), h = 1000 W/(m2.K), beta = 1e-7 s/m
    output: water-content profiles at 100, 300 and 1000 h

The official analytical solution (Bench2.xls) is not available here, so the model is compared with an
independent reference: the same 1-D moisture balance written in water content and solved with the method
of lines on a fine mesh (scipy BDF):
    dw/dt = d/dx[ D(w) dw/dx ],  D(w) = D_w + delta_p * psat * dphi/dw
    surface flux into the wall = beta * psat * (phi_air - phi(w_surface))
The report also quotes the analytical boundary values at 1000 h: w(0) = 19.54, w(0.2) = 30.51 kg/m3.

    python validation/hamstad_bm2.py
"""

import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.sparse import diags

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import library as lib
from wall_layer import WallLayer, register_material
from wall_class import Wall

# ── benchmark data ─────────────────────────────────────────────────────────────
L, T0 = 0.200, 293.15
RH_INIT, RH_LEFT, RH_RIGHT = 0.95, 0.45, 0.65
H, BETA = 1000.0, 1e-7
DW, DELTA_P = 6e-10, 1e-15
W_F, A_VG, N_VG = 121.4, 0.129, 0.907
TIMES_H = [100, 300, 1000]
ANALYTICAL_1000H = {0.0: 19.54, 0.2: 30.51}          # boundary values quoted in the report


def w_of_phi(phi):
    phi = np.clip(phi, 1e-6, 1.0)
    return W_F / (1.0 + (-np.log(phi) / A_VG) ** N_VG)


def phi_of_w(w):
    w = np.clip(w, 1e-9, W_F - 1e-9)
    return np.exp(-A_VG * (W_F / w - 1.0) ** (1.0 / N_VG))


class HAMSTAD_BM2_A:
    """HAMSTAD benchmark 2, material A."""
    def rho(T, RH):     return 525.0
    def Cp(T, RH):      return 800.0
    def k(T, RH):       return 0.15
    def w(T, RH):       return w_of_phi(RH)
    def delta_p(T, RH): return DELTA_P
    def Dw(T, RH):      return DW
    def delta_l(T, RH): return 0.0


register_material("HAMSTAD_BM2_A", HAMSTAD_BM2_A)


# ── the model under test ───────────────────────────────────────────────────────
def run_model(mesh_size, dt, n_sub=1):
    layer = WallLayer(["HAMSTAD_BM2_A"], [L], Mesh_Opt=0, liq=1, mesh_size=mesh_size)
    wall = Wall(layer, h_ext=H, hm_ext=BETA, h_int=H, hm_int=BETA, T_init=T0, RH_init=RH_INIT)
    out, t = {}, 0.0
    for t_h in TIMES_H:
        while t < t_h * 3600 - 1e-6:
            wall.step(T0, RH_LEFT, T0, RH_RIGHT, dt, store=False)
            t += dt
        out[t_h] = layer.w(wall.T, wall.RH).flatten().copy()
    return layer.x_pos.copy(), out


# ── independent reference (method of lines, finite volumes in w) ──────────────
def run_reference(n_cells=400, rh_factor=True):
    """rh_factor=False reproduces a liquid flux D_w/phi * dw/dx (diagnostic of a missing RH factor)."""
    dx = L / n_cells
    xc = (np.arange(n_cells) + 0.5) * dx
    psat = float(lib.Psat(T0, 0.5))

    def D(w):
        phi = phi_of_w(w)
        dphi_dw = (phi_of_w(w * (1 + 1e-6)) - phi_of_w(w * (1 - 1e-6))) / (2e-6 * w)
        Dl = DW if rh_factor else DW / phi
        return Dl + DELTA_P * psat * dphi_dw

    def rhs(t, w):
        Dc = D(w)
        Df = 0.5 * (Dc[1:] + Dc[:-1])
        flux = np.zeros(n_cells + 1)                          # positive in +x
        flux[1:-1] = -Df * (w[1:] - w[:-1]) / dx
        # surface cells: half-cell resistance in series with the surface vapour transfer
        flux[0] = -BETA * psat * (phi_of_w(w[0]) - RH_LEFT)        # into the wall (+x) when air is wetter
        flux[-1] = BETA * psat * (phi_of_w(w[-1]) - RH_RIGHT)       # out of the wall (+x) when wall is wetter
        return -(flux[1:] - flux[:-1]) / dx

    w0 = np.full(n_cells, float(w_of_phi(RH_INIT)))
    sparsity = diags([1.0, 1.0, 1.0], [-1, 0, 1], shape=(n_cells, n_cells))   # tridiagonal Jacobian
    sol = solve_ivp(rhs, (0, TIMES_H[-1] * 3600), w0, method="BDF", t_eval=[t * 3600 for t in TIMES_H],
                    rtol=1e-7, atol=1e-6, jac_sparsity=sparsity)
    return xc, {t: sol.y[:, i] for i, t in enumerate(TIMES_H)}


def main():
    t0 = time.time()
    xr, ref = run_reference()
    _, ref_norh = run_reference(rh_factor=False)
    print(f"reference solved in {time.time() - t0:.0f} s", flush=True)
    runs = {}
    for mesh, dt in ((4e-3, 600.0), (2e-3, 600.0), (2e-3, 120.0), (1e-3, 120.0)):
        t0 = time.time()
        runs[(mesh, dt)] = run_model(mesh, dt)
        print(f"model mesh {mesh*1e3:.0f} mm, dt {dt:.0f} s: {time.time() - t0:.0f} s", flush=True)

    print("\nMax |w_model - w_reference| [kg/m3]   (reference with RH factor | without)")
    for (mesh, dt), (xm, wm) in runs.items():
        errs = []
        for t in TIMES_H:
            e1 = np.max(np.abs(wm[t] - np.interp(xm, xr, ref[t])))
            e2 = np.max(np.abs(wm[t] - np.interp(xm, xr, ref_norh[t])))
            errs.append(f"{t:>4} h: {e1:6.2f} | {e2:6.2f}")
        print(f"  mesh {mesh*1e3:.0f} mm, dt {dt:>3.0f} s   " + "   ".join(errs))
    xm, wm = runs[(1e-3, 120.0)]
    print(f"\nBoundary values at 1000 h — model: {wm[1000][0]:.2f} / {wm[1000][-1]:.2f}, "
          f"reference: {ref[1000][0]:.2f} / {ref[1000][-1]:.2f}, "
          f"analytical (report): {ANALYTICAL_1000H[0.0]} / {ANALYTICAL_1000H[0.2]} kg/m3")

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for t, c in zip(TIMES_H, ["#2A78D6", "#EB6834", "#1BAF7A"]):
        ax.plot(xr * 100, ref[t], color=c, lw=2.5, label=f"reference, {t} h")
        ax.plot(xr * 100, ref_norh[t], color=c, lw=1.2, ls=":")
        ax.plot(xm * 100, wm[t], ls="", marker="o", ms=4, color=c, mfc="white", markevery=8,
                label=f"model, {t} h")
    ax.plot([0, 20], [ANALYTICAL_1000H[0.0], ANALYTICAL_1000H[0.2]], ls="", marker="x", ms=10, color="k",
            label="analytical surface values, 1000 h")
    ax.set_xlabel("x [cm]"); ax.set_ylabel("w [kg/m³]")
    ax.set_title("HAMSTAD BM2 — isothermal drying (dotted: reference without RH factor)", loc="left")
    ax.legend(fontsize=9, ncol=2)
    out = os.path.join(HERE, "hamstad_bm2.png")
    fig.savefig(out, dpi=150)
    print("figure:", out)


if __name__ == "__main__":
    main()
