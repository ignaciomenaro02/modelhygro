# -*- coding: utf-8 -*-
"""
sim_dephasage_mur.py
====================
Isola el efecto PURO del muro: excitación sinusoidal T_ext, T_int fija.
Mide el déphasage y la atenuación real de cada solución constructiva.

Sin sala → sin ganancias solares, sin ventilación, sin cargas internas.
Esto revela la inercia térmica intrínseca del muro.

Comparativa:
  A — Chanvre 30 cm          (solución higroscópica)
  B — Béton normal 20 cm     (masa densa, referencia)
  C — ITE: Laine roche 12cm + Béton 18cm  (aislamiento exterior)
  D — ITI: Béton 18cm + Laine roche 12cm  (aislamiento interior)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Ruta del modelo ────────────────────────────────────────────────────────────
DIR = r"C:\Users\IMena\Desktop\modelhygro"
if DIR not in sys.path:
    sys.path.insert(0, DIR)

import library as lib
from wall_layer import WallLayer
from wall_class  import Wall

OUT_DIR = DIR

# ── Parámetros de simulación ───────────────────────────────────────────────────
DT       = 3600.0          # [s] paso horario
N_DAYS   = 20              # días totales (15 warmup + 5 análisis)
N_WARMUP = 15              # días de calentamiento descartados
N_STEPS  = int(N_DAYS * 86400 / DT)
N_ANA    = int((N_DAYS - N_WARMUP) * 86400 / DT)  # pasos para análisis

# Condiciones de frontera
T_MEAN   = 20.0 + 273.15   # [K] temperatura media
A_EXT    = 10.0            # [K] amplitud oscilación exterior (±10°C)
RH_EXT   = 0.70            # HR exterior (constante)
T_INT    = 20.0 + 273.15   # [K] temperatura interior fija
RH_INT   = 0.50            # HR interior fija

H_EXT  = 25.0              # [W/(m²·K)] coeficiente conv. exterior
HM_EXT = 25e-9             # [kg/(m²·s·Pa)]
H_INT  = 8.0               # [W/(m²·K)] coeficiente conv. interior
HM_INT = 8e-9              # [kg/(m²·s·Pa)]

T_INIT  = T_MEAN
RH_INIT = 0.60

# ── Definición de los muros ────────────────────────────────────────────────────
WALLS = {
    "Chanvre 30 cm": {
        "mat":   ["Hempcrete"],
        "e":     [0.30],
        "color": "#7B9E3E",
    },
    "Béton 20 cm": {
        "mat":   ["Concrete"],
        "e":     [0.20],
        "color": "#888888",
    },
    "ITE: LR 12cm + Béton 18cm\n(isolation extérieure)": {
        "mat":   ["Rock_Wool", "Concrete"],
        "e":     [0.12, 0.18],
        "color": "#D95F02",
    },
    "ITI: Béton 18cm + LR 12cm\n(isolation intérieure)": {
        "mat":   ["Concrete", "Rock_Wool"],
        "e":     [0.18, 0.12],
        "color": "#1B7DB8",
    },
}

# ── Fonction FFT déphasage ─────────────────────────────────────────────────────
def dephasage_fft(sig_in, sig_out, dt_h=1.0):
    """
    Phase shift and attenuation at 24h period (FFT method).
    sig_in, sig_out: 1D arrays (same length), dt_h in hours.
    Returns (attenuation [-], dephasage [h]).
    """
    n      = len(sig_in)
    f      = np.fft.rfftfreq(n, d=dt_h)
    F_in   = np.fft.rfft(sig_in  - sig_in.mean())
    F_out  = np.fft.rfft(sig_out - sig_out.mean())
    idx    = np.argmin(np.abs(f - 1.0/24.0))
    amp_in  = abs(F_in[idx])
    amp_out = abs(F_out[idx])
    atten   = amp_out / amp_in if amp_in > 1e-10 else 0.0
    dphi    = np.angle(F_out[idx]) - np.angle(F_in[idx])
    dphi    = (dphi + np.pi) % (2*np.pi) - np.pi
    dephas  = -dphi / (2*np.pi) * 24.0
    if dephas < 0:
        dephas += 24.0
    return float(atten), float(dephas)

# ── Calcul U-value ─────────────────────────────────────────────────────────────
def u_value(mat_list, e_list):
    """Approximate U-value [W/(m²·K)] from conductivities at T=293K, RH=0.5."""
    from materials_library import (Hempcrete, Concrete, Rock_Wool, Lime_Plaster,
                                   Wood_Fiber, Rammed_Earth)
    registry = {"Hempcrete": Hempcrete, "Concrete": Concrete,
                 "Rock_Wool": Rock_Wool, "Lime_Plaster": Lime_Plaster,
                 "Wood_Fiber": Wood_Fiber, "Rammed_Earth": Rammed_Earth}
    T0, RH0 = 293.15, 0.5
    R = 1/H_EXT + 1/H_INT
    for mat, e in zip(mat_list, e_list):
        cls = registry.get(mat)
        if cls:
            R += e / cls.k(T0, RH0)
    return 1.0 / R

# ── Calcul théorique (mur homogène équivalent) ─────────────────────────────────
def theo_dephasage(mat_list, e_list):
    """
    Theoretical déphasage using the thermal damping depth formula.
    For multi-layer: use effective values weighted by thickness.
    """
    from materials_library import (Hempcrete, Concrete, Rock_Wool, Lime_Plaster,
                                   Wood_Fiber, Rammed_Earth)
    registry = {"Hempcrete": Hempcrete, "Concrete": Concrete,
                 "Rock_Wool": Rock_Wool, "Lime_Plaster": Lime_Plaster,
                 "Wood_Fiber": Wood_Fiber, "Rammed_Earth": Rammed_Earth}
    T0, RH0 = 293.15, 0.5
    omega = 2*np.pi/86400.0
    # Per-layer déphasage (additive approximation)
    t_lag = 0.0
    for mat, e in zip(mat_list, e_list):
        cls = registry.get(mat)
        if cls is None:
            continue
        lam = cls.k(T0, RH0)
        rho = cls.rho(T0, RH0)
        cp  = cls.Cp(T0, RH0)
        a   = lam / (rho * cp)
        d   = np.sqrt(2*a/omega)
        t_lag += e * 86400.0 / (2*np.pi*d)   # seconds
    return t_lag / 3600.0  # hours

# ── Simulation de chaque muro ──────────────────────────────────────────────────
print("=" * 60)
print("Simulation déphasage intrinsèque des parois")
print("Excitation sinusoïdale pure — sans dynamique de pièce")
print("=" * 60)

results = {}

for name, cfg in WALLS.items():
    label_short = name.split("\n")[0]
    print(f"\n{'-'*50}")
    print(f"  {label_short}")

    layer = WallLayer(
        mat      = cfg["mat"],
        emat     = cfg["e"],
        Mesh_Opt = 0,
        liq      = 0,
        label    = label_short,
    )

    wall = Wall(
        layer  = layer,
        h_ext  = H_EXT,  hm_ext = HM_EXT,
        h_int  = H_INT,  hm_int = HM_INT,
        T_init = T_INIT, RH_init = RH_INIT,
    )

    T_ext_arr    = []
    T_surf_ext_arr = []
    T_surf_int_arr = []
    q_int_arr    = []

    for i in range(N_STEPS):
        t_s    = i * DT
        T_e    = T_MEAN + A_EXT * np.sin(2*np.pi * t_s / 86400.0)
        T_ext_arr.append(float(T_e) - 273.15)

        wall.step(T_e, RH_EXT, T_INT, RH_INT, DT)

        T_surf_ext_arr.append(float(wall.T[0,  0]) - 273.15)
        T_surf_int_arr.append(float(wall.T[-1, 0]) - 273.15)
        _, q_i, _, _ = wall.surface_fluxes()
        q_int_arr.append(q_i)

    T_ext_arr      = np.array(T_ext_arr)
    T_surf_ext_arr = np.array(T_surf_ext_arr)
    T_surf_int_arr = np.array(T_surf_int_arr)
    q_int_arr      = np.array(q_int_arr)

    # Analyse sur les N_ANA derniers pas
    T_ext_ana      = T_ext_arr[-N_ANA:]
    T_se_ana       = T_surf_ext_arr[-N_ANA:]
    T_si_ana       = T_surf_int_arr[-N_ANA:]
    q_int_ana      = q_int_arr[-N_ANA:]

    atten_mat, dep_mat = dephasage_fft(T_se_ana, T_si_ana, dt_h=DT/3600.0)
    atten_tot, dep_tot = dephasage_fft(T_ext_ana, T_si_ana, dt_h=DT/3600.0)
    dep_theo = theo_dephasage(cfg["mat"], cfg["e"])
    U = u_value(cfg["mat"], cfg["e"])

    results[name] = {
        "cfg":           cfg,
        "T_ext_ana":     T_ext_ana,
        "T_se_ana":      T_se_ana,
        "T_si_ana":      T_si_ana,
        "q_int_ana":     q_int_ana,
        "atten_mat":     atten_mat,
        "dep_mat":       dep_mat,
        "atten_tot":     atten_tot,
        "dep_tot":       dep_tot,
        "dep_theo":      dep_theo,
        "U":             U,
    }

    print(f"  U-value      = {U:.3f} W/(m²·K)")
    print(f"  Théorique    = {dep_theo:.1f} h")
    print(f"  FFT surf_ext->surf_int: dep={dep_mat:.1f} h, atten={atten_mat*100:.1f}%")
    print(f"  FFT T_ext->surf_int:    dep={dep_tot:.1f} h, atten={atten_tot*100:.1f}%")

print("\n[OK] Simulation terminee\n")

# ── Graphiques ─────────────────────────────────────────────────────────────────

# Couleurs et étiquettes courtes
COLORS = [cfg["color"] for cfg in WALLS.values()]
LABELS = [n.replace("\n", " ") for n in WALLS.keys()]
NAMES  = list(WALLS.keys())
t_h    = np.arange(N_ANA) * (DT/3600.0)  # axe temps en heures

# ─── Figure 1 : Séries temporelles (5 jours) ───────────────────────────────
fig, axes = plt.subplots(len(WALLS), 1, figsize=(14, 4*len(WALLS)), sharex=True)
fig.suptitle("Déphasage intrinsèque des parois\n"
             "(Excitation sinusoïdale T_ext = 20 ± 10 °C, T_int fixe = 20 °C)",
             fontsize=13, fontweight="bold")

for ax, name in zip(axes, NAMES):
    r  = results[name]
    c  = r["cfg"]["color"]
    ax.plot(t_h, r["T_ext_ana"],  color="tomato", lw=1.5, ls="--", label="T_ext (air extérieur)")
    ax.plot(t_h, r["T_se_ana"],   color="orange", lw=1.5, ls=":",  label="T surf. ext. (nœud 0)")
    ax.plot(t_h, r["T_si_ana"],   color=c,        lw=2.5,          label="T surf. int. (nœud N)")

    # Annoter le déphasage
    dep = r["dep_mat"]
    att = r["atten_mat"] * 100
    ax.set_title(
        f"{LABELS[NAMES.index(name)]}   —   "
        f"Déphasage = {dep:.1f} h   |   Atténuation = {att:.0f}%   |   U = {r['U']:.3f} W/(m²·K)",
        loc="left", fontsize=10, color=c
    )
    ax.set_ylabel("T [°C]", fontsize=9)
    ax.set_ylim(5, 35)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    # Marquer les 24h
    for d in range(0, N_ANA, 24):
        ax.axvline(d, color="lightgray", lw=0.8, ls=":")

axes[-1].set_xlabel("Temps [h] (5 jours d'analyse)", fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dephasage_series.pdf"), dpi=150)
plt.close()
print("  [OK] dephasage_series.pdf")

# ─── Figure 2 : Résumé comparatif (barres) ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Comparaison des solutions constructives — Inercie thermique intrinsèque",
             fontsize=12, fontweight="bold")

x     = np.arange(len(NAMES))
w     = 0.55
deps  = [results[n]["dep_mat"]    for n in NAMES]
attns = [(1 - results[n]["atten_mat"]) * 100 for n in NAMES]  # taux d'amortissement
Us    = [results[n]["U"]          for n in NAMES]
theos = [results[n]["dep_theo"]   for n in NAMES]
short = [n.split("\n")[0] for n in NAMES]

# Déphasage
ax = axes[0]
bars = ax.bar(x, deps, width=w, color=COLORS, alpha=0.85, edgecolor="white")
ax.bar(x, theos, width=w*0.5, color="none", edgecolor="black", lw=1.5,
       linestyle="--", label="Théorique (semi-infini)")
for i, (d, dt) in enumerate(zip(deps, theos)):
    ax.text(i, d + 0.3, f"{d:.1f}h", ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=COLORS[i])
ax.set_xticks(x)
ax.set_xticklabels(short, fontsize=8)
ax.set_ylabel("Déphasage [h]", fontsize=10)
ax.set_title("Déphasage\n(surf. ext → surf. int)", fontsize=10)
ax.set_ylim(0, max(max(deps), max(theos)) * 1.3)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")
ax.axhline(8, color="gray", lw=1, ls=":", alpha=0.7)
ax.text(len(NAMES)-0.5, 8.2, "8h (seuil confort été)", fontsize=7, color="gray")

# Amortissement
ax = axes[1]
bars = ax.bar(x, attns, width=w, color=COLORS, alpha=0.85, edgecolor="white")
for i, a in enumerate(attns):
    ax.text(i, a + 0.5, f"{a:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=COLORS[i])
ax.set_xticks(x)
ax.set_xticklabels(short, fontsize=8)
ax.set_ylabel("Amortissement [%]", fontsize=10)
ax.set_title("Amortissement de l'onde thermique\n(1 − atténuation)", fontsize=10)
ax.set_ylim(0, 110)
ax.grid(True, alpha=0.3, axis="y")

# U-value
ax = axes[2]
bars = ax.bar(x, Us, width=w, color=COLORS, alpha=0.85, edgecolor="white")
for i, u in enumerate(Us):
    ax.text(i, u + 0.01, f"{u:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=COLORS[i])
ax.set_xticks(x)
ax.set_xticklabels(short, fontsize=8)
ax.set_ylabel("U [W/(m²·K)]", fontsize=10)
ax.set_title("Coefficient de transmission\nthermique U", fontsize=10)
ax.axhline(0.36, color="red", lw=1.5, ls="--", label="RE2020 limite paroi")
ax.legend(fontsize=8)
ax.set_ylim(0, max(Us) * 1.4)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dephasage_bilan.pdf"), dpi=150)
plt.close()
print("  [OK] dephasage_bilan.pdf")

# ─── Figure 3 : Flux de chaleur à la surface intérieure ────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
fig.suptitle("Flux de chaleur à la surface intérieure des parois\n"
             "(> 0 = flux entrant dans la pièce)",
             fontsize=12, fontweight="bold")
axes_flat = axes.flatten()
for ax, name in zip(axes_flat, NAMES):
    r = results[name]
    c = r["cfg"]["color"]
    ax.plot(t_h, r["q_int_ana"], color=c, lw=2)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    q_max = np.max(np.abs(r["q_int_ana"]))
    ax.set_ylim(-q_max*1.3, q_max*1.3)
    ax.fill_between(t_h, r["q_int_ana"], 0,
                    where=r["q_int_ana"] > 0, color=c, alpha=0.3, label="Chaleur → pièce")
    ax.fill_between(t_h, r["q_int_ana"], 0,
                    where=r["q_int_ana"] < 0, color="steelblue", alpha=0.2, label="Chaleur ← pièce")
    dep = r["dep_mat"]
    att = r["atten_mat"] * 100
    ax.set_title(f"{name.split(chr(10))[0]}\n"
                 f"Dép={dep:.1f}h, Att={att:.0f}%, "
                 f"q_max={q_max:.1f} W/m²",
                 fontsize=9, color=c)
    ax.set_ylabel("q_int [W/m²]", fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    for d in range(0, N_ANA, 24):
        ax.axvline(d, color="lightgray", lw=0.8, ls=":")

for ax in axes[1]:
    ax.set_xlabel("Temps [h]", fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dephasage_flux.pdf"), dpi=150)
plt.close()
print("  [OK] dephasage_flux.pdf")

# ─── Figure 4 : Diagramme de phase (ellipse de Lissajous) ──────────────────
fig, axes = plt.subplots(1, len(WALLS), figsize=(4*len(WALLS), 4))
fig.suptitle("Diagramme de phase — T_ext vs T_surf_int\n"
             "(l'ellipse montre: largeur = atténuation, inclinaison = déphasage)",
             fontsize=11, fontweight="bold")

for ax, name in zip(axes, NAMES):
    r  = results[name]
    c  = r["cfg"]["color"]
    # Dernier cycle: 24h
    n24 = int(24 * 3600 / DT)
    T_e_last = r["T_ext_ana"][-n24:]
    T_i_last = r["T_si_ana"][-n24:]
    ax.plot(T_e_last, T_i_last, color=c, lw=2)
    ax.plot([10, 30], [10, 30], "k--", lw=0.8, alpha=0.4, label="Sans déphasage")
    ax.set_xlabel("T_ext [°C]", fontsize=9)
    ax.set_ylabel("T_surf_int [°C]", fontsize=9)
    dep = r["dep_mat"]
    att = r["atten_mat"] * 100
    ax.set_title(f"{name.split(chr(10))[0]}\n{dep:.1f}h / {att:.0f}%", fontsize=9, color=c)
    ax.set_xlim(8, 32)
    ax.set_ylim(15, 25)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dephasage_lissajous.pdf"), dpi=150)
plt.close()
print("  [OK] dephasage_lissajous.pdf")

print("\n[DONE] Tous les graphiques generes dans:", OUT_DIR)
print("   dephasage_series.pdf    - series temporelles T_surf_ext -> T_surf_int")
print("   dephasage_bilan.pdf     - resume comparatif (dephasage + amortissement + U)")
print("   dephasage_flux.pdf      - flux de chaleur a la surface interieure")
print("   dephasage_lissajous.pdf - diagrammes de phase Lissajous")
