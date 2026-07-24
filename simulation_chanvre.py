# -*- coding: utf-8 -*-
"""
simulation_chanvre.py
=====================
Simulation hygro-thermique d'une pièce en béton de chanvre 30 cm.
Dimensions : 4 × 4 m, hauteur 2.5 m  (16 m² / 40 m³)
Période    : 21 juin → 21 septembre (92 jours)
Météo      : Mâcon prospectif +2 °C

Graphiques produits
-------------------
  modele_3d.pdf          — vue 3D de la pièce avec légendes R, U, λ
  profils_parois_NS.pdf  — profils T et HR parois Nord et Sud
  climat.pdf             — données climatiques utilisées
  confort_map.pdf        — confort adaptatif EN 15251
  degres_heures.pdf      — degrés-heures d'inconfort RE2020
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

DIR = r"C:\Users\IMena\Desktop\modelhygro"
if DIR not in sys.path:
    sys.path.insert(0, DIR)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from wall_config     import make_wall, WindowConfig, ThermalBridge, OpeningConfig
from sources         import (OccupantConfig, EquipmentConfig, LightingConfig,
                              VentilationConfig, build_daily_schedule)
from solar           import SolarCalculator
from re2020          import RE2020Evaluator
from room_simulation import RoomSimulation
from weather         import load_weather_csv


# ══════════════════════════════════════════════════════════════════════════════
# 1. PARAMÈTRES
# ══════════════════════════════════════════════════════════════════════════════

# ── Géométrie ─────────────────────────────────────────────────────────────────
Lx = 4.0          # [m]  longueur intérieure (Est–Ouest)
Ly = 4.0          # [m]  profondeur intérieure (Sud–Nord)
H  = 2.5          # [m]  hauteur sous plafond
EPAISSEUR_M   = 0.30
SURFACE_SOL   = Lx * Ly          # 16 m²
VOLUME_AIR    = Lx * Ly * H      # 40 m³
MASSE_INTERNE = 110e3 * SURFACE_SOL   # [J/K]  ISO 13790 masse moyenne

# ── Paroi ─────────────────────────────────────────────────────────────────────
MESH_SIZE  = 0.01
LAMBDA_REF = 0.067    # [W/(m·K)]  conductivité de référence pour R/U

R_mat = EPAISSEUR_M / LAMBDA_REF          # 4.48 m²·K/W
R_tot = R_mat + 1/8.0 + 1/25.0           # + résistances superficielles
U_ref = 1.0 / R_tot                       # 0.215 W/(m²·K)

# ── Effet hygroscopique ───────────────────────────────────────────────────────
# True  = transfert vapeur actif (HAM complet)
# False = paroi imperméable — pour désactiver changer ici
EFFET_HYGRO = True
HM_EXT = 25e-9 if EFFET_HYGRO else 1e-30
HM_INT = HM_EXT * (8.0 / 25.0)

# ── Ventilation — RE2020 ──────────────────────────────────────────────────────
# Arrêté du 3 mai 2007 : chambre > 15 m²  →  20 m³/h minimum hygiène
# Soit : 20 / 40 m³ = 0.50 vol/h en débit de base (VMC)
#
# Surventilation nocturne RE2020 (calcul DH) : 4 vol/h de 22h à 8h
# → Modifier ACH_NUIT pour tester d'autres débits
ACH_JOUR    = 0.50              # [vol/h]  VMC hygiène (jour)
ACH_NUIT    = 4.0               # [vol/h]  surventilation nocturne  ← MODIFIER
HEURES_NUIT = list(range(0,8)) + list(range(22,24))   # 22h → 8h

# ── Heure pour les profils de paroi ──────────────────────────────────────────
HEURE_PROFIL = 14   # [h]  ← MODIFIER


# ══════════════════════════════════════════════════════════════════════════════
# 2. MÉTÉO
# ══════════════════════════════════════════════════════════════════════════════

FICHIER_METEO = os.path.join(DIR,
    "donnees-climatiques-prospectives-france-2c_macon.csv")
meteo = load_weather_csv(FICHIER_METEO)
print(f"Météo : {meteo.station}  "
      f"T {meteo.T_ext.min():.1f}..{meteo.T_ext.max():.1f} C")

# 21 juin = jour 172 (doy)  →  début à la seconde 171×86400
# 21 sep  = jour 264 (doy)  →  début à la seconde 263×86400
T_21JUN = 171 * 86400.0
T_21SEP = 263 * 86400.0
DOY_DEBUT = 172    # jour julien du 21 juin


# ══════════════════════════════════════════════════════════════════════════════
# 3. BÂTIMENT
# ══════════════════════════════════════════════════════════════════════════════

# Fenêtres : proportionnées pour 16 m² (ratio vitrage/sol ~12.5%)
# Sud 1.5 m²  →  apports solaires principaux
# Nord 0.5 m² →  éclairage naturel
vitrages = [
    WindowConfig("Vitrage Sud",  area=1.5, orientation="S",
                 U_value=1.1, g_value=0.60, shading=0.30),
    WindowConfig("Vitrage Nord", area=0.5, orientation="N",
                 U_value=1.1, g_value=0.60, shading=0.0),
]
ponts_thermiques = [
    ThermalBridge("Encadrements", psi=0.04, length=6.0),
]
infiltrations = [OpeningConfig("Infiltrations", ach_contribution=0.1)]


def creer_parois():
    """4 parois en béton de chanvre 30 cm — géométrie 4×4×2.5 m."""
    kw = dict(layers=[("Hempcrete", EPAISSEUR_M)], mesh_size=MESH_SIZE,
              h_ext=25.0, hm_ext=HM_EXT, h_int=8.0, hm_int=HM_INT)
    return [
        make_wall("Sud",   area=Lx*H - 1.5, orientation="S", **kw),  # 8.5 m²
        make_wall("Nord",  area=Lx*H - 0.5, orientation="N", **kw),  # 9.5 m²
        make_wall("Est",   area=Ly*H,        orientation="E", **kw),  # 10.0 m²
        make_wall("Ouest", area=Ly*H,        orientation="W", **kw),  # 10.0 m²
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 4. SIMULATION ÉTÉ  (21 juin → 21 septembre)
# ══════════════════════════════════════════════════════════════════════════════

DT = 3600.0

mask     = (meteo.time_s >= T_21JUN) & (meteo.time_s < T_21SEP)
time_bc  = meteo.time_s[mask] - T_21JUN
Text_bc  = meteo.T_ext[mask]
RHext_bc = meteo.RH_ext[mask]
n_steps  = len(time_bc) - 1
steps_j  = int(86400 / DT)

# Plannings
sched_occ = build_daily_schedule(list(range(8, 22)),  n_steps, int(DT))
sched_lum = build_daily_schedule(list(range(20, 24)), n_steps, int(DT))

ach_total = ACH_NUIT + ACH_JOUR
sched_vent= build_daily_schedule(
    hours_on=HEURES_NUIT, n_steps=n_steps, dt=int(DT),
    value_on=1.0, value_off=ACH_JOUR / ach_total)
vent = VentilationConfig(n_ach=ach_total, hrv_efficiency=0.0,
                         moisture_recovery=0.0, schedule=sched_vent)

re_ev = RE2020Evaluator(floor_area=SURFACE_SOL, climate_zone="H1b")
re_ev.set_carrier(heating="electricity", cooling="electricity")

sim = RoomSimulation(
    wall_configs   = creer_parois(),
    window_configs = vitrages,
    bridge_configs = ponts_thermiques,
    opening_configs= infiltrations,
    occupants  = OccupantConfig(1.0, 80.0, 60.0, sched_occ),
    equipment  = EquipmentConfig(100.0, schedule=sched_occ),
    lighting   = LightingConfig(5.0, SURFACE_SOL, schedule=sched_lum),
    ventilation= vent,
    hvac       = None,
    solar_calc = SolarCalculator(meteo.lat, meteo.lon, start_doy=DOY_DEBUT,
                                 cloud_factor=0.30, ground_albedo=0.20,
                                 timezone_offset=1.0),
    volume        = VOLUME_AIR,
    internal_mass = MASSE_INTERNE,
    T_room_init   = float(Text_bc[0]),
    RH_room_init  = float(RHext_bc[0]),
    re2020        = re_ev,
)

print(f"\nSimulation 21 jun → 21 sep ({n_steps} pas)  "
      f"ACH nuit={ACH_NUIT} vol/h  hygro={'ON' if EFFET_HYGRO else 'OFF'}")
T_arr, RH_arr = sim.run(time_bc=time_bc, Text_bc=Text_bc,
                         RHext_bc=RHext_bc, dt=DT, verbose=True)

T_arr    = np.array(T_arr)
RH_arr   = np.array(RH_arr) * 100
T_ext_f  = np.interp(np.arange(len(T_arr)) * DT, time_bc, Text_bc)
RH_ext_f = np.interp(np.arange(len(T_arr)) * DT, time_bc, RHext_bc) * 100

x_pos = sim.walls[0].layer.x_pos   # positions noeuds [m], ext → int

# Jour le plus chaud
n_j      = len(T_arr) // steps_j
T_max_j  = [T_ext_f[d*steps_j:(d+1)*steps_j].max() for d in range(n_j)]
JOUR_C   = int(np.argmax(T_max_j))

DH_total = float(np.maximum(T_arr - 28.0, 0.0).sum())   # [°C·h]

print(f"\n  T max ext={T_ext_f.max():.1f}°C  T max pièce={T_arr.max():.1f}°C")
print(f"  DH RE2020 = {DH_total:.0f} °C·h  ({DH_total/1250*100:.0f}% seuil H1b)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. GRAPHIQUE 1 — Modèle 3D
# ══════════════════════════════════════════════════════════════════════════════

print("\n  Tracé : modele_3d.pdf ...")

t = EPAISSEUR_M

def face(pts):
    """Retourne une face Poly3D à partir d'une liste de 4 points [x,y,z]."""
    return pts

def box_faces(x0,x1,y0,y1,z0,z1):
    """6 faces d'une boîte rectangulaire."""
    return [
        [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],  # bas
        [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],  # haut
        [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],  # façade y=y0
        [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],  # façade y=y1
        [[x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]],  # flanc x=x0
        [[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[x1,y0,z1]],  # flanc x=x1
    ]

fig5 = plt.figure(figsize=(13, 9))
ax3d = fig5.add_subplot(111, projection="3d")
ax3d.set_facecolor("#F8F8F8")

# ── Parois (boîtes 3D jointives aux coins) ────────────────────────────────────
# S/N spans full exterior width including corners; E/W fills between
walls_3d = [
    #  nom     x0    x1      y0    y1     couleur    label_face_ext
    ("Sud",   -t,   Lx+t,  -t,   0,    "#D4956A",  (Lx/2, -t/2,  H*0.55)),
    ("Nord",  -t,   Lx+t,   Ly,  Ly+t, "#7BA7C4",  (Lx/2, Ly+t/2,H*0.55)),
    ("Ouest", -t,   0,      0,    Ly,   "#A8C888",  (-t/2, Ly/2,  H*0.55)),
    ("Est",    Lx,  Lx+t,   0,    Ly,   "#C8A8C8",  (Lx+t/2,Ly/2,H*0.55)),
]

for nom, x0, x1, y0, y1, col, lpos in walls_3d:
    poly = Poly3DCollection(box_faces(x0,x1,y0,y1,0,H),
                            alpha=0.78, facecolor=col,
                            edgecolor="#444444", linewidth=0.5)
    ax3d.add_collection3d(poly)
    # Étiquette sur la face extérieure visible
    ax3d.text(*lpos, nom, ha="center", va="center",
              fontsize=11, fontweight="bold", color="#222222",
              bbox=dict(boxstyle="round,pad=0.15", fc=col, ec="#444444",
                        alpha=0.85, linewidth=0.8))

# ── Sol intérieur ─────────────────────────────────────────────────────────────
sol = [[[0,0,0],[Lx,0,0],[Lx,Ly,0],[0,Ly,0]]]
ax3d.add_collection3d(Poly3DCollection(sol, alpha=0.30,
                      facecolor="#E8D8B0", edgecolor="#999999", linewidth=0.6))

# ── Fenêtres (rectangles sur les parois Sud et Nord) ─────────────────────────
# Vitrage Sud : 1.5 m² ≈ 1.2 m large × 1.25 m haut, centré
w_s_w=1.2; w_s_h=1.25; w_s_x=(Lx-w_s_w)/2; w_s_z=0.90
win_sud = [[[w_s_x,-t+0.01,w_s_z],[w_s_x+w_s_w,-t+0.01,w_s_z],
            [w_s_x+w_s_w,-t+0.01,w_s_z+w_s_h],[w_s_x,-t+0.01,w_s_z+w_s_h]]]
ax3d.add_collection3d(Poly3DCollection(win_sud, alpha=0.55,
                      facecolor="#AED6F1", edgecolor="#2980B9", linewidth=1.2))
ax3d.text(w_s_x+w_s_w/2, -t+0.02, w_s_z+w_s_h+0.07,
          f"Vitrage S\n1.5 m²", ha="center", va="bottom",
          fontsize=7.5, color="#1A5276")

# Vitrage Nord : 0.5 m² ≈ 0.7 m × 0.71 m, centré
w_n_w=0.70; w_n_h=0.71; w_n_x=(Lx-w_n_w)/2; w_n_z=1.20
win_nord = [[[w_n_x,Ly+t-0.01,w_n_z],[w_n_x+w_n_w,Ly+t-0.01,w_n_z],
             [w_n_x+w_n_w,Ly+t-0.01,w_n_z+w_n_h],[w_n_x,Ly+t-0.01,w_n_z+w_n_h]]]
ax3d.add_collection3d(Poly3DCollection(win_nord, alpha=0.55,
                      facecolor="#AED6F1", edgecolor="#2980B9", linewidth=1.2))
ax3d.text(w_n_x+w_n_w/2, Ly+t-0.02, w_n_z+w_n_h+0.07,
          f"Vitrage N\n0.5 m²", ha="center", va="bottom",
          fontsize=7.5, color="#1A5276")

# ── Flèche Nord ──────────────────────────────────────────────────────────────
ax3d.quiver(Lx/2, Ly+t+0.25, H+0.05, 0, 0.35, 0,
            color="#CC3030", lw=2.5, arrow_length_ratio=0.45)
ax3d.text(Lx/2, Ly+t+0.65, H+0.05, "N", ha="center", va="bottom",
          fontsize=13, fontweight="bold", color="#CC3030")

# ── Cotes ────────────────────────────────────────────────────────────────────
# Largeur (axe x)
ax3d.plot([0, Lx], [-t-0.18, -t-0.18], [0,0], color="#555555", lw=1.2)
ax3d.text(Lx/2, -t-0.22, 0, f"{Lx:.0f} m", ha="center", va="top",
          fontsize=9, color="#555555")
# Profondeur (axe y) — à droite
ax3d.plot([Lx+t+0.12]*2, [0, Ly], [0,0], color="#555555", lw=1.2)
ax3d.text(Lx+t+0.16, Ly/2, 0, f"{Ly:.0f} m", ha="left", va="center",
          fontsize=9, color="#555555", rotation=90)
# Hauteur (axe z)
ax3d.plot([Lx+t+0.12]*2, [Ly+t+0.12]*2, [0, H], color="#555555", lw=1.2)
ax3d.text(Lx+t+0.16, Ly+t+0.15, H/2, f"h = {H:.1f} m",
          ha="left", va="center", fontsize=9, color="#555555")

# ── Légende matériaux ─────────────────────────────────────────────────────────
info = (
    f"Béton de chanvre\n"
    f"─────────────────\n"
    f"e  = {EPAISSEUR_M*100:.0f} cm\n"
    f"λ  = {LAMBDA_REF} W/(m·K)\n"
    f"ρ  = 450 kg/m³\n"
    f"R  = {R_mat:.2f} m²·K/W\n"
    f"U  = {U_ref:.3f} W/(m²·K)"
)
ax3d.text2D(0.01, 0.97, info, transform=ax3d.transAxes,
            fontsize=9.5, va="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.55", fc="#FFFDF0",
                      ec="#C8A86A", alpha=0.95))

ax3d.set_xlim(-t-0.3, Lx+t+0.5)
ax3d.set_ylim(-t-0.3, Ly+t+0.8)
ax3d.set_zlim(0, H+0.35)
ax3d.set_xlabel(""); ax3d.set_ylabel(""); ax3d.set_zlabel("")
ax3d.set_xticks([]); ax3d.set_yticks([]); ax3d.set_zticks([])
ax3d.set_title(
    f"Béton de chanvre — Pièce {Lx:.0f}×{Ly:.0f} m  h = {H:.1f} m\n"
    f"Surface : {SURFACE_SOL:.0f} m²    Volume : {VOLUME_AIR:.0f} m³    "
    f"Parois : {EPAISSEUR_M*100:.0f} cm",
    fontsize=12, fontweight="bold", pad=10)
ax3d.view_init(elev=24, azim=-48)
ax3d.grid(False)

_out = os.path.join(DIR, "modele_3d.pdf")
try:    fig5.savefig(_out, dpi=150, bbox_inches="tight"); print("  [OK] modele_3d.pdf")
except PermissionError:
    fig5.savefig(_out.replace(".pdf","_new.pdf"), dpi=150, bbox_inches="tight")
plt.close(fig5)


# ══════════════════════════════════════════════════════════════════════════════
# 6. GRAPHIQUE 2 — Profils T et HR parois Nord et Sud  (3 heures)
# ══════════════════════════════════════════════════════════════════════════════
# Trois snapshots sur le jour le plus chaud :
#   6h  = fin de nuit  (ventilation encore active, paroi la plus froide)
#  14h  = pic de chaleur extérieure
#  22h  = début de nuit (paroi encore chaude, ventilation qui reprend)

print("  Tracé : profils_parois_NS.pdf ...")

HEURES_3 = [6, 14, 22]
C_3      = ["#1F5F99", "#E07020", "#CC3030"]   # bleu, orange, rouge
i0_j     = JOUR_C * steps_j   # premier pas du jour le plus chaud
x_cm     = x_pos * 100        # [cm]

PAROIS_NS = [("Sud", 0), ("Nord", 1)]

# Layout : 2 lignes (T, HR) × 2 colonnes (Sud, Nord)
fig6, axes6 = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

from datetime import date, timedelta
date_c = (date(2000,6,21) + timedelta(days=JOUR_C)).strftime("%d %B").lstrip("0")  # année fictive — données prospectives

fig6.suptitle(
    f"Profils de température et d'humidité relative — parois Sud et Nord\n"
    f"Jour le plus chaud : {date_c}  |  "
    f"Béton de chanvre {EPAISSEUR_M*100:.0f} cm  |  "
    f"Hygroscopique : {'actif' if EFFET_HYGRO else 'désactivé'}",
    fontsize=11, fontweight="bold"
)

# Légende commune (3 heures)
leg_handles = [Line2D([0],[0], color=c, lw=2.5, label=f"{h:02d}h")
               for h, c in zip(HEURES_3, C_3)]

for col, (nom, idx) in enumerate(PAROIS_NS):
    wall = sim.walls[idx]

    ax_T = axes6[0, col]   # Température
    ax_H = axes6[1, col]   # Humidité relative

    for h, couleur in zip(HEURES_3, C_3):
        i_snap = i0_j + h
        T_prof = wall.StockT[i_snap].flatten()  - 273.15
        H_prof = wall.StockRH[i_snap].flatten() * 100

        ax_T.plot(x_cm, T_prof, color=couleur, lw=2.5, label=f"{h:02d}h")
        ax_H.plot(x_cm, H_prof, color=couleur, lw=2.5, label=f"{h:02d}h")

        # Valeur à la surface intérieure
        ax_T.annotate(f"{T_prof[-1]:.1f}°C",
                      xy=(x_cm[-1], T_prof[-1]),
                      xytext=(x_cm[-1]-1.5, T_prof[-1]+0.3),
                      fontsize=8, color=couleur, ha="right")
        ax_H.annotate(f"{H_prof[-1]:.0f}%",
                      xy=(x_cm[-1], H_prof[-1]),
                      xytext=(x_cm[-1]-1.5, H_prof[-1]+1.0),
                      fontsize=8, color=couleur, ha="right")

    for ax in [ax_T, ax_H]:
        ax.axvline(0,               color="#CCCCCC", lw=1.2, ls="--")
        ax.axvline(EPAISSEUR_M*100, color="#CCCCCC", lw=1.2, ls="--")
        ax.set_xlim(0, EPAISSEUR_M*100)
        ax.set_xticks(np.linspace(0, EPAISSEUR_M*100, 7))
        ax.grid(True, alpha=0.18)
        y_top = ax.get_ylim()[1]
        ax.text(0.8,   y_top, "EXT", fontsize=9, color="#888888", va="top")
        ax.text(EPAISSEUR_M*100-0.8, y_top, "INT", fontsize=9,
                color="#888888", va="top", ha="right")
        ax.legend(handles=leg_handles, fontsize=10, loc="center",
                  framealpha=0.90, title="Heure", title_fontsize=9)

    ax_T.set_title(f"Paroi {nom} — Température [°C]",
                   fontsize=11, fontweight="bold")
    ax_T.set_ylabel("T [°C]", fontsize=11)
    ax_T.tick_params(labelbottom=False)

    ax_H.set_title(f"Paroi {nom} — Humidité relative [%]",
                   fontsize=11, fontweight="bold")
    ax_H.set_ylabel("HR [%]", fontsize=11)
    ax_H.set_xlabel("Position dans la paroi [cm]   (ext → int)", fontsize=10)
    ax_H.set_ylim(0, 105)

    # Note explicative sur la paroi Sud uniquement
    if col == 0:
        axes6[0,0].text(0.02, 0.03,
            "6h  : fin de surventilation nocturne\n"
            "14h : pic de chaleur extérieure\n"
            "22h : début de surventilation",
            transform=axes6[0,0].transAxes, fontsize=8.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5",
                      ec="#AAAAAA", alpha=0.92))

_out = os.path.join(DIR, "profils_parois_NS.pdf")
try:    fig6.savefig(_out, dpi=150); print("  [OK] profils_parois_NS.pdf")
except PermissionError:
    fig6.savefig(_out.replace(".pdf","_new.pdf"), dpi=150)
plt.close(fig6)


# ══════════════════════════════════════════════════════════════════════════════
# 7. GRAPHIQUE 3 — Données climatiques
# ══════════════════════════════════════════════════════════════════════════════

print("  Tracé : climat.pdf ...")

jours = np.arange(n_j)
T_moy_e = np.array([T_ext_f[d*steps_j:(d+1)*steps_j].mean() for d in range(n_j)])
T_min_e = np.array([T_ext_f[d*steps_j:(d+1)*steps_j].min()  for d in range(n_j)])
T_max_e = np.array([T_ext_f[d*steps_j:(d+1)*steps_j].max()  for d in range(n_j)])
HR_moy_e= np.array([RH_ext_f[d*steps_j:(d+1)*steps_j].mean()for d in range(n_j)])
HR_min_e= np.array([RH_ext_f[d*steps_j:(d+1)*steps_j].min() for d in range(n_j)])
HR_max_e= np.array([RH_ext_f[d*steps_j:(d+1)*steps_j].max() for d in range(n_j)])

# Étiquettes de mois
from datetime import date, timedelta
import locale
try:
    locale.setlocale(locale.LC_TIME, "French_France.1252")
except Exception:
    pass
date0   = date(2000, 6, 21)  # année fictive — données prospectives
tick_j  = []
tick_lb = []
for d in range(n_j):
    dd = date0 + timedelta(days=d)
    if dd.day == 1:
        tick_j.append(d); tick_lb.append(dd.strftime("%B"))

fig7, (ax_Tc, ax_HRc) = plt.subplots(2, 1, figsize=(14, 8),
                                       constrained_layout=True)
fig7.suptitle(
    f"Données climatiques utilisées — Mâcon prospectif +2 °C\n"
    f"21 juin → 21 septembre  |  {n_j} jours",
    fontsize=12, fontweight="bold"
)

ax_Tc.fill_between(jours, T_min_e, T_max_e, color="#E07020", alpha=0.20,
                   label=f"Plage journalière (min–max)  amplitude moy. {(T_max_e-T_min_e).mean():.1f} °C")
ax_Tc.plot(jours, T_moy_e, color="#CC5500", lw=2.0, label="Moyenne journalière")
ax_Tc.axhline(28, color="#CC3030", lw=1.0, ls=":", alpha=0.7,
              label="Seuil RE2020 DH = 28 °C")
ax_Tc.set_ylabel("Température extérieure [°C]", fontsize=11)
ax_Tc.set_xlim(0, n_j-1)
if tick_j: ax_Tc.set_xticks(tick_j); ax_Tc.set_xticklabels(tick_lb, fontsize=10)
ax_Tc.legend(fontsize=10, loc="upper right", framealpha=0.92)
ax_Tc.grid(True, alpha=0.18)

ax_HRc.fill_between(jours, HR_min_e, HR_max_e, color="#5577CC", alpha=0.20,
                    label=f"Plage journalière (min–max)")
ax_HRc.plot(jours, HR_moy_e, color="#334499", lw=2.0, label="Moyenne journalière")
ax_HRc.set_ylim(0, 105)
ax_HRc.set_ylabel("Humidité relative extérieure [%]", fontsize=11)
ax_HRc.set_xlabel("Période simulée", fontsize=11)
ax_HRc.set_xlim(0, n_j-1)
if tick_j: ax_HRc.set_xticks(tick_j); ax_HRc.set_xticklabels(tick_lb, fontsize=10)
ax_HRc.legend(fontsize=10, loc="upper right", framealpha=0.92)
ax_HRc.grid(True, alpha=0.18)

_out = os.path.join(DIR, "climat.pdf")
try:    fig7.savefig(_out, dpi=150); print("  [OK] climat.pdf")
except PermissionError:
    fig7.savefig(_out.replace(".pdf","_new.pdf"), dpi=150)
plt.close(fig7)


# ══════════════════════════════════════════════════════════════════════════════
# 8. GRAPHIQUE 4 — Confort adaptatif EN 15251
# ══════════════════════════════════════════════════════════════════════════════

print("  Tracé : confort_map.pdf ...")

# T_rm : warm-up sur mai
alpha = 0.8
T_MAI1 = (31+28+31+30)*86400.0
mask_mai = (meteo.time_s >= T_MAI1) & (meteo.time_s < T_21JUN)
T_ext_mai= meteo.T_ext[mask_mai]
T_moy_mai= np.array([T_ext_mai[d*steps_j:(d+1)*steps_j].mean()
                     for d in range(len(T_ext_mai)//steps_j)])
T_rm_init = T_moy_mai[0]
for Td in T_moy_mai:
    T_rm_init = (1-alpha)*Td + alpha*T_rm_init

T_moy_ete = np.array([T_ext_f[d*steps_j:(d+1)*steps_j].mean()
                       for d in range(n_j)])
T_rm_jour = np.zeros(n_j)
T_rm_jour[0] = (1-alpha)*T_moy_mai[-1] + alpha*T_rm_init
for d in range(1, n_j):
    T_rm_jour[d] = (1-alpha)*T_moy_ete[d-1] + alpha*T_rm_jour[d-1]
T_rm_h = np.repeat(T_rm_jour, steps_j)[:len(T_arr)]

T_comf = 0.33*T_rm_h + 18.8
T_sup  = T_comf + 3.0
T_inf  = T_comf - 3.0
dans_zone  = (T_arr >= T_inf) & (T_arr <= T_sup)
trop_chaud = T_arr > T_sup
trop_froid = T_arr < T_inf
pct_c = 100*dans_zone.sum()/len(T_arr)
pct_ch= 100*trop_chaud.sum()/len(T_arr)
pct_fr= 100*trop_froid.sum()/len(T_arr)

fig8, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
T_rm_range = np.linspace(T_rm_h.min()-1, T_rm_h.max()+1, 200)
ax.fill_between(T_rm_range, 0.33*T_rm_range+18.8-3, 0.33*T_rm_range+18.8+3,
                color="#B8E0B0", alpha=0.55, label="Zone de confort EN 15251 Cat. II (±3 °C)")
ax.plot(T_rm_range, 0.33*T_rm_range+18.8,
        color="#2E8B57", lw=1.5, ls="--", label="Température de confort optimale")
ax.scatter(T_rm_h[dans_zone],  T_arr[dans_zone],  c="#2E8B57", s=5,
           alpha=0.35, label=f"En confort  {pct_c:.0f} %")
ax.scatter(T_rm_h[trop_chaud], T_arr[trop_chaud], c="#CC3030", s=5,
           alpha=0.45, label=f"Trop chaud  {pct_ch:.0f} %")
ax.scatter(T_rm_h[trop_froid], T_arr[trop_froid], c="#1F5F99", s=5,
           alpha=0.45, label=f"Trop froid  {pct_fr:.0f} %")
ax.set_xlabel("Température extérieure de référence T_rm [°C]\n"
              "(moyenne glissante pondérée, α = 0.8, EN 15251)", fontsize=11)
ax.set_ylabel("Température de la pièce [°C]", fontsize=11)
ax.set_title(
    f"Confort adaptatif EN 15251 Cat. II — Béton de chanvre {EPAISSEUR_M*100:.0f} cm\n"
    f"21 juin → 21 sep  |  Surventilation {ACH_NUIT} vol/h (RE2020)  |  "
    f"Pièce {Lx:.0f}×{Ly:.0f} m",
    fontsize=11, fontweight="bold")
ax.legend(fontsize=10, loc="upper left", framealpha=0.92, markerscale=3)
ax.grid(True, alpha=0.20)
ax.text(0.98, 0.03,
    f"Résultats :\n  En confort : {pct_c:.0f} %\n"
    f"  Trop chaud : {pct_ch:.0f} %\n  Trop froid : {pct_fr:.0f} %\n\n"
    f"T_comf = 0.33·T_rm + 18.8 °C\nLimite Cat. II : ± 3 °C",
    transform=ax.transAxes, fontsize=9.5, va="bottom", ha="right",
    family="monospace",
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#BBBBBB", alpha=0.92))

_out = os.path.join(DIR, "confort_map.pdf")
try:    fig8.savefig(_out, dpi=150); print("  [OK] confort_map.pdf")
except PermissionError:
    fig8.savefig(_out.replace(".pdf","_new.pdf"), dpi=150)
plt.close(fig8)


# ══════════════════════════════════════════════════════════════════════════════
# 9. GRAPHIQUE 5 — Degrés-heures RE2020
# ══════════════════════════════════════════════════════════════════════════════

print("  Tracé : degres_heures.pdf ...")

DH_jour  = np.array([float(np.maximum(
    T_arr[d*steps_j:min((d+1)*steps_j,len(T_arr))]-28., 0.).sum())
    for d in range(n_j)])
DH_cumul = np.cumsum(DH_jour)

fig9, (ax_bar, ax_cum) = plt.subplots(2, 1, figsize=(13, 8),
                                        constrained_layout=True)
fig9.suptitle(
    f"Degrés-heures d'inconfort RE2020 — Béton de chanvre {EPAISSEUR_M*100:.0f} cm\n"
    f"DH total = {DH_total:.0f} °C·h  ({DH_total/1250*100:.0f}% du seuil H1b = 1 250 °C·h/an)",
    fontsize=12, fontweight="bold"
)

ax_bar.bar(jours, DH_jour,
           color=["#CC3030" if v > 0 else "#AED6F1" for v in DH_jour],
           alpha=0.85, width=0.9)
ax_bar.set_ylabel("DH journalier [°C·h]", fontsize=11)
ax_bar.set_xlim(-0.5, n_j-0.5)
if tick_j: ax_bar.set_xticks(tick_j); ax_bar.set_xticklabels(tick_lb, fontsize=10)
ax_bar.grid(True, alpha=0.18, axis="y")
ax_bar.set_title("Degrés-heures d'inconfort par jour  (rouge = inconfort actif)", fontsize=10)

ax_cum.plot(jours, DH_cumul, color="#CC3030", lw=2.2, label="DH cumulés")
ax_cum.axhline(1250, color="#333333", lw=1.5, ls="--",
               label="Seuil RE2020 zone H1b = 1 250 °C·h/an")
ax_cum.fill_between(jours, DH_cumul, 1250,
                    where=(DH_cumul >= 1250),
                    color="#CC3030", alpha=0.15, label="Dépassement du seuil")
ax_cum.set_ylabel("DH cumulés [°C·h]", fontsize=11)
ax_cum.set_xlabel("Période simulée", fontsize=11)
ax_cum.set_xlim(-0.5, n_j-0.5)
if tick_j: ax_cum.set_xticks(tick_j); ax_cum.set_xticklabels(tick_lb, fontsize=10)
ax_cum.legend(fontsize=10, framealpha=0.92)
ax_cum.grid(True, alpha=0.18)

_out = os.path.join(DIR, "degres_heures.pdf")
try:    fig9.savefig(_out, dpi=150); print("  [OK] degres_heures.pdf")
except PermissionError:
    fig9.savefig(_out.replace(".pdf","_new.pdf"), dpi=150)
plt.close(fig9)


# ══════════════════════════════════════════════════════════════════════════════
# 10. RÉSUMÉ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  RÉSULTATS")
print("-"*60)
print(f"  Pièce : {Lx:.0f}×{Ly:.0f} m  h={H:.1f} m  "
      f"S={SURFACE_SOL:.0f} m²  V={VOLUME_AIR:.0f} m³")
print(f"  λ = {LAMBDA_REF}  R = {R_mat:.2f} m²K/W  U = {U_ref:.3f} W/(m²K)")
print(f"  Ventilation : {ACH_JOUR} vol/h (jour)  +  {ACH_NUIT} vol/h (nuit RE2020)")
print(f"  Hygroscopique : {'ACTIF' if EFFET_HYGRO else 'DESACTIVE'}")
print(f"  T max ext   : {T_ext_f.max():.1f} °C")
print(f"  T max pièce : {T_arr.max():.1f} °C")
print(f"  DH RE2020   : {DH_total:.0f} °C·h  ({DH_total/1250*100:.0f}% seuil H1b)")
print(f"  Confort EN 15251 : {pct_c:.0f}% OK  |  {pct_ch:.0f}% trop chaud  |  {pct_fr:.0f}% trop froid")
print("-"*60)
print("  FICHIERS")
for f in ["modele_3d.pdf","profils_parois_NS.pdf","climat.pdf",
          "confort_map.pdf","degres_heures.pdf"]:
    print(f"  {f}")
print("="*60)
