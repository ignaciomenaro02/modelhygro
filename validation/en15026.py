# -*- coding: utf-8 -*-
"""
en15026.py
==========
NF EN 15026 (2007), Annex B (normative) — benchmark of coupled heat and moisture transfer.

Semi-infinite homogeneous layer, initially at 20 °C / 50 % RH. At t = 0 the surface is set to
30 °C / 95 % RH (no surface resistance, no rain). Moisture-content and temperature profiles after
7, 30 and 365 days are compared with the analytical solution of J. Claesson (Tables B.1 and B.2);
accepted deviation: ±2.5 % for the moisture content, ±1 % for the temperature.

Material (Annex B.2), properties evaluated at T = Tref = 293.15 K:
    w(phi)  = wsat / (1 + (8e-8 · s)^1.6)^0.375,   s = -Rw·T·rho_w·ln(phi),   wsat = 146 kg/m3
    delta_p = Mw/(R·T) · 26.1e-6/200 · (1 - w/wsat) / (0.503·(1 - w/wsat)^2 + 0.497)
    K       = exp(-39.2619 + 0.0704·(w-73) - 1.7420e-4·(w-73)^2 - 2.7953e-6·(w-73)^3
                  - 1.1566e-7·(w-73)^4 + 2.5969e-9·(w-73)^5)        liquid conductivity [kg/(m.s.Pa)]
    lambda  = 1.5 + 15.8·w/1000,   rho0·c0 = 1.84e6 J/(m3.K)

In the model the liquid flux is K·dPc/dx, i.e. the liquid-permeability term (delta_l) of the vapour-only
formulation (liq = 0) of wall_class.py.

    python validation/en15026.py
"""
import os, sys, time, csv
import numpy as np
import matplotlib
matplotlib.use ("Agg")
import matplotlib.pyplot as plt 

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import library as lib
from wall_layer import WallLayer, register_material 
from wall_class import Wall

TREF, RW, RHO_W, WSAT = 293.15, 462.0, 1000.0, 146.0
DAYS = [7, 30, 365]


# Annex B tables B.1 (w) and B.2 (T): reference values of J. Claesson and accepted bands,
# read from en15026_annexB.csv (columns: quantity, day, x_m, min, ref, max)
TABLES_CSV = os.path.join(HERE, "en15026_annexB.csv")


def load_tables(path=TABLES_CSV):
    """Return {quantity: (x, {day: min}, {day: ref}, {day: max})} for quantity "w" and "T"."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(line for line in f if line.strip() and not line.startswith("#")))
    tables = {}
    for q in ("w", "T"):
        sel = sorted((r for r in rows if r["quantity"] == q), key=lambda r: (int(r["day"]), float(r["x_m"])))
        x = sorted({float(r["x_m"]) for r in sel})
        per_day = lambda col: {d: [float(r[col]) for r in sel if int(r["day"]) == d]
                               for d in sorted({int(r["day"]) for r in sel})}
        tables[q] = (x, per_day("min"), per_day("ref"), per_day("max"))
    return tables


_TABLES = load_tables()
X_W, W_MIN, W_REF, W_MAX = _TABLES["w"]
X_T, T_MIN, T_REF, T_MAX = _TABLES["T"]
assert sorted(W_REF) == DAYS and sorted(T_REF) == DAYS, "en15026_annexB.csv must give the days 7, 30, 365"


def w_of_s(s):
    """Water retention curve, function of the suction s [Pa] only."""
    s = np.maximum(s, 0.0)                    # RH slightly above 1 in the numerical derivatives
    return WSAT / (1.0 + (8e-8 * s) ** 1.6) ** 0.375


def w_of_phi(phi, T=TREF):
    """Same curve expressed with RH: s = -Rw·T·rho_w·ln(phi) (Kelvin), at the local temperature T."""
    return w_of_s(-RW * np.asarray(T) * RHO_W * np.log(np.clip(phi, 1e-9, 1.0)))


class EN15026_Material:
    """NF EN 15026 Annex B benchmark material. The storage is a function of the suction, i.e. of the
    model's capillary pressure Pc = -rho_w·Rv·T·ln(RH) evaluated at the local temperature: a temperature
    change at constant Pc must not create or remove water."""
    def rho(T, RH):     return 1840.0
    def Cp(T, RH):      return 1000.0                        # rho0·c0 = 1.84e6 J/(m3.K)
    def w(T, RH):       return w_of_s(lib.Pc(T, RH))
    def k(T, RH):       return 1.5 + 15.8 * EN15026_Material.w(T, RH) / 1000.0

    def delta_p(T, RH):
        r = 1.0 - EN15026_Material.w(T, RH) / WSAT
        return 1.0 / (RW * T) * 26.1e-6 / 200.0 * r / (0.503 * r ** 2 + 0.497)

    def delta_l(T, RH):
        u = EN15026_Material.w(T, RH) - 73.0
        return np.exp(-39.2619 + 0.0704 * u - 1.7420e-4 * u ** 2 - 2.7953e-6 * u ** 3
                      - 1.1566e-7 * u ** 4 + 2.5969e-9 * u ** 5)

    def Dw(T, RH):      return 0.0


register_material("EN15026_Material", EN15026_Material)


def run(depth=30.0, h=1e5, hm=1e-5, dt_steps=((2, 60.0), (30, 600.0), (365, 1800.0))):
    """Semi-infinite layer approximated by `depth` m with a graded mesh (0.5 mm at the surface).
    dt_steps: (until day, time step [s]) pairs. Returns x and the T [°C], w [kg/m3] profiles."""
    layer = WallLayer(["EN15026_Material"], [depth], Mesh_Opt=1, liq=0)
    wall = Wall(layer, h_ext=h, hm_ext=hm, h_int=h, hm_int=hm, T_init=TREF, RH_init=0.50)
    out, t = {}, 0.0
    for day in DAYS:
        while t < day * 86400 - 1e-6:
            dt = next(d for until, d in dt_steps if t < until * 86400 - 1e-6)
            wall.step(303.15, 0.95, TREF, 0.50, dt, store=False)       # far face kept at the initial state
            t += dt
        out[day] = ((wall.T.flatten() - 273.15).copy(), layer.w(wall.T, wall.RH).flatten().copy())
    return layer.x_pos.copy(), out


def check(x, out):
    ok_all = True
    print(f"\n{'':10}{'moisture content w [kg/m3]':^60}")
    for d in DAYS:
        T, w = out[d]
        wm = np.interp(X_W, x, w)
        Tm = np.interp(X_T, x, T)
        okw = [lo <= v <= hi for v, lo, hi in zip(wm, W_MIN[d], W_MAX[d])]
        okT = [lo <= v <= hi for v, lo, hi in zip(Tm, T_MIN[d], T_MAX[d])]
        ok_all &= all(okw) and all(okT)
        print(f"  {d:>3} d  w: " + " ".join(f"{v:6.1f}{'' if o else '*'}" for v, o in zip(wm, okw))
              + f"   max dev {np.max(np.abs(wm / np.array(W_REF[d]) - 1)) * 100:4.1f} %")
        print(f"         T: " + " ".join(f"{v:6.2f}{'' if o else '*'}" for v, o in zip(Tm, okT))
              + f"   max dev {np.max(np.abs(Tm / np.array(T_REF[d]) - 1)) * 100:4.2f} %")
    print("  (* = outside the accepted band)")
    print("RESULT:", "PASS — all values within the EN 15026 tolerances" if ok_all else "FAIL")
    return ok_all


def plot(x, out, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for d, c in zip(DAYS, ["#2A78D6", "#EB6834", "#1BAF7A"]):
        T, w = out[d]
        a.plot(x, w, color=c, lw=2, label=f"model, {d} d")
        a.errorbar(X_W, W_REF[d], yerr=[np.subtract(W_REF[d], W_MIN[d]), np.subtract(W_MAX[d], W_REF[d])],
                   fmt="o", color=c, mfc="white", capsize=3, label=f"EN 15026 ref., {d} d")
        b.plot(x, T, color=c, lw=2)
        b.errorbar(X_T, T_REF[d], yerr=[np.subtract(T_REF[d], T_MIN[d]), np.subtract(T_MAX[d], T_REF[d])],
                   fmt="o", color=c, mfc="white", capsize=3)
    a.set_xlim(0, 0.11); a.set_xlabel("x [m]"); a.set_ylabel("w [kg/m³]"); a.legend(fontsize=8, ncol=2)
    b.set_xlim(0, 5.5); b.set_ylim(19.5, 30.5); b.set_xlabel("x [m]"); b.set_ylabel("T [°C]")
    a.set_title("Moisture content", loc="left"); b.set_title("Temperature", loc="left")
    fig.suptitle("NF EN 15026 Annex B benchmark — model vs analytical reference (bars: accepted band)")
    fig.savefig(path, dpi=150)


def main():
    t0 = time.time()
    x, out = run()
    print(f"model: {len(x)} nodes, {time.time() - t0:.0f} s", flush=True)
    check(x, out)
    path = os.path.join(HERE, "en15026.png")
    plot(x, out, path)
    print("figure:", path)


if __name__ == "__main__":
    main()
