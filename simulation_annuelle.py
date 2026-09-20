# -*- coding: utf-8 -*-
"""
simulation_annuelle.py
======================
Simulation hygro-thermique annuelle (365 jours) — même pièce que simulation_chanvre.py
Dimensions : 4 × 4 m, hauteur 2.5 m  (16 m² / 40 m³)
Période    : 1 janvier → 31 décembre  (année complète)
Météo      : fichiers RE2020 par zone (H1a…H3) — voir ZONE / MODE dans la section 2
             (python simulation_annuelle.py H3 [Th-BC|Th-D] ; "MACON" = CSV prospectif +2 °C)

Graphiques produits  (suffixe _<ZONE> ajouté à chaque nom)
-------------------
  annuel_serie.pdf       — séries T et HR pièce vs extérieur (année entière)
  annuel_profils.pdf      — profils parois Nord/Sud  (jour le + chaud et le + froid)
  annuel_confort.pdf      — confort adaptatif EN 15251 (été) + chauffage (hiver)
  annuel_degres_heures.pdf— degrés-heures été RE2020 + degrés-jours hiver
  annuel_climat.pdf       — données climatiques annuelles
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from datetime import date, timedelta
import locale

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
# from re2020        import RE2020Evaluator   # calcul RE2020 en standby
from room_simulation import RoomSimulation
from weather         import load_weather_csv, load_re2020_weather

try:
    locale.setlocale(locale.LC_TIME, "French_France.1252")
except Exception:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# 1. PARAMÈTRES — identiques à simulation_chanvre.py
# ══════════════════════════════════════════════════════════════════════════════

Lx = 4.0
Ly = 4.0
H  = 2.5
EPAISSEUR_M   = 0.30
SURFACE_SOL   = Lx * Ly          # 16 m²
VOLUME_AIR    = Lx * Ly * H      # 40 m³
MASSE_INTERNE = 110e3 * SURFACE_SOL

MESH_SIZE  = 0.01
LAMBDA_REF = 0.067

R_mat = EPAISSEUR_M / LAMBDA_REF
R_tot = R_mat + 1/8.0 + 1/25.0
U_ref = 1.0 / R_tot

EFFET_HYGRO = True
HM_EXT = 25e-9 if EFFET_HYGRO else 1e-30
HM_INT = HM_EXT * (8.0 / 25.0)

# Ventilation RE2020 : 0.5 vol/h (hygiène) + 4 vol/h surventilation nocturne (été)
# En hiver la surventilation nocturne est désactivée : ACH_NUIT = ACH_JOUR
ACH_JOUR = 0.50
ACH_NUIT = 4.0
HEURES_NUIT = list(range(0, 8)) + list(range(22, 24))

# Saison de surventilation : mai → septembre (DOY 121 → 273)
DOY_SURVENT_DEBUT = 121   # 1er mai
DOY_SURVENT_FIN   = 273   # 30 septembre


# ══════════════════════════════════════════════════════════════════════════════
# 2. MÉTÉO — année complète
# ══════════════════════════════════════════════════════════════════════════════

# Zone climatique RE2020 : H1a H1b H1c H2a H2b H2c H2d H3   (ou "MACON" = CSV prospectif +2 °C)
# Mode : "Th-BC" = année type (besoins/consommations, humidité)  |  "Th-D" = confort d'été
# Usage :  python simulation_annuelle.py H3          ou      python simulation_annuelle.py H3 Th-D
ZONE = sys.argv[1] if len(sys.argv) > 1 else "H1c"
ZONE = "MACON" if ZONE.upper() == "MACON" else ZONE[:2].upper() + ZONE[2:].lower()   # h1C → H1c
MODE = sys.argv[2] if len(sys.argv) > 2 else "Th-BC"

if ZONE == "MACON":
    meteo = load_weather_csv(os.path.join(DIR,
        "donnees-climatiques-prospectives-france-2c_macon.csv"))
    LABEL_METEO = "Mâcon prospectif +2 °C"
else:
    meteo = load_re2020_weather(os.path.join(DIR,
        "scenarios_meteorologiques_th-bc_th-d_re2020.xlsx"), ZONE, MODE)
    LABEL_METEO = f"RE2020 {ZONE} ({MODE})"
SUF = f"_{ZONE}"    # suffixe des PDF : une zone n'écrase pas l'autre

def _pdf(nom):
    return os.path.join(DIR, f"{nom}{SUF}.pdf")

print(f"Météo : {LABEL_METEO}  "
      f"T {meteo.T_ext.min():.1f}..{meteo.T_ext.max():.1f} C")

# Année complète : 0 → 365×86400
T_DEBUT = 0.0
T_FIN   = 365 * 86400.0
DOY_DEBUT = 1   # 1er janvier

DT = 3600.0

mask     = (meteo.time_s >= T_DEBUT) & (meteo.time_s < T_FIN)
time_bc  = meteo.time_s[mask]
Text_bc  = meteo.T_ext[mask]
RHext_bc = meteo.RH_ext[mask]
n_steps  = len(time_bc) - 1
steps_j  = int(86400 / DT)
n_j      = n_steps // steps_j

print(f"  {n_steps} pas de temps  ({n_j} jours)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. BÂTIMENT — identique à simulation_chanvre.py
# ══════════════════════════════════════════════════════════════════════════════

vitrages = [
    WindowConfig("Vitrage Sud",  area=1.5, orientation="S",
                 U_value=1.1, g_value=0.60, shading=0.30),
    WindowConfig("Vitrage Nord", area=0.5, orientation="N",
                 U_value=1.1, g_value=0.60, shading=0.0),
]
ponts_thermiques = [ThermalBridge("Encadrements", psi=0.04, length=6.0)]
infiltrations    = [OpeningConfig("Infiltrations", ach_contribution=0.1)]


def creer_parois():
    kw = dict(layers=[("Hempcrete", EPAISSEUR_M)], mesh_size=MESH_SIZE,
              h_ext=25.0, hm_ext=HM_EXT, h_int=8.0, hm_int=HM_INT)
    return [
        make_wall("Sud",   area=Lx*H - 1.5, orientation="S", **kw),
        make_wall("Nord",  area=Lx*H - 0.5, orientation="N", **kw),
        make_wall("Est",   area=Ly*H,        orientation="E", **kw),
        make_wall("Ouest", area=Ly*H,        orientation="W", **kw),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# 4. PLANNINGS annuels
# ══════════════════════════════════════════════════════════════════════════════

sched_occ = build_daily_schedule(list(range(8, 22)),  n_steps, int(DT))
sched_lum = build_daily_schedule(list(range(20, 24)), n_steps, int(DT))

# Ventilation : surventilation nocturne active seulement mai–septembre
ach_total = ACH_NUIT + ACH_JOUR
ratio_jour = ACH_JOUR / ach_total

sched_vent = []
for step in range(n_steps):
    doy = int(step // steps_j) + 1   # jour julien (1-365)
    heure = step % steps_j
    nuit  = heure in HEURES_NUIT
    ete   = (DOY_SURVENT_DEBUT <= doy <= DOY_SURVENT_FIN)
    if nuit and ete:
        sched_vent.append(1.0)       # plein débit nocturne estival
    else:
        sched_vent.append(ratio_jour)  # VMC hygiène uniquement

vent = VentilationConfig(n_ach=ach_total, hrv_efficiency=0.0,
                         moisture_recovery=0.0, schedule=sched_vent)

# RE2020 en standby : pas d'évaluateur connecté à la simulation
# re_ev = RE2020Evaluator(floor_area=SURFACE_SOL, climate_zone="H1b")
# re_ev.set_carrier(heating="electricity", cooling="electricity")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SIMULATION annuelle
# ══════════════════════════════════════════════════════════════════════════════

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
                                 timezone_offset=1.0, weather=meteo),
    volume        = VOLUME_AIR,
    internal_mass = MASSE_INTERNE,
    T_room_init   = float(Text_bc[0]),
    RH_room_init  = float(RHext_bc[0]),
    re2020        = None,   # RE2020 en standby
)

print(f"\nSimulation annuelle ({n_steps} pas)  "
      f"ACH nuit={ACH_NUIT} vol/h (mai–sep)  hygro={'ON' if EFFET_HYGRO else 'OFF'}")
T_arr, RH_arr = sim.run(time_bc=time_bc, Text_bc=Text_bc,
                         RHext_bc=RHext_bc, dt=DT, verbose=True)

T_arr    = np.array(T_arr)
RH_arr   = np.array(RH_arr) * 100
T_ext_f  = np.interp(np.arange(len(T_arr)) * DT, time_bc, Text_bc)
RH_ext_f = np.interp(np.arange(len(T_arr)) * DT, time_bc, RHext_bc) * 100

# Tronquer aux jours complets (n_j × steps_j) pour cohérence des indices
n_j     = len(T_arr) // steps_j
N_sync  = n_j * steps_j
T_arr    = T_arr[:N_sync]
RH_arr   = RH_arr[:N_sync]
T_ext_f  = T_ext_f[:N_sync]
RH_ext_f = RH_ext_f[:N_sync]

x_pos = sim.walls[0].layer.x_pos
x_cm  = x_pos * 100

# Jours extrêmes
T_max_j  = [T_ext_f[d*steps_j:(d+1)*steps_j].max() for d in range(n_j)]
T_min_j  = [T_ext_f[d*steps_j:(d+1)*steps_j].min() for d in range(n_j)]
JOUR_C   = int(np.argmax(T_max_j))   # jour le plus chaud
JOUR_F   = int(np.argmin(T_min_j))   # jour le plus froid

date_c = (date(2000, 1, 1) + timedelta(days=JOUR_C)).strftime("%d %B").lstrip("0")
date_f = (date(2000, 1, 1) + timedelta(days=JOUR_F)).strftime("%d %B").lstrip("0")

# Indicateurs
DH_total = float(np.maximum(T_arr - 28.0, 0.0).sum())   # [°C·h] été
DJ_total = float(np.maximum(18.0 - T_arr, 0.0).sum() / 24.0)  # [°C·j] hiver

print(f"\n  Jour le + chaud : {date_c} (jour {JOUR_C})  T_ext={T_max_j[JOUR_C]:.1f}°C")
print(f"  Jour le + froid : {date_f} (jour {JOUR_F})  T_ext min={T_min_j[JOUR_F]:.1f}°C")
print(f"  T max ext={T_ext_f.max():.1f}°C  T max pièce={T_arr.max():.1f}°C")
print(f"  T min ext={T_ext_f.min():.1f}°C  T min pièce={T_arr.min():.1f}°C")
print(f"  DH RE2020 = {DH_total:.0f} °C·h  ({DH_total/1250*100:.0f}% du seuil 1 250 °C·h)")
print(f"  DJ hiver  = {DJ_total:.0f} °C·j")


# ══════════════════════════════════════════════════════════════════════════════
# Utilitaire : étiquettes mensuelles
# ══════════════════════════════════════════════════════════════════════════════

jours    = np.arange(n_j)
date0    = date(2000, 1, 1)
tick_j, tick_lb = [], []
for d in range(n_j):
    dd = date0 + timedelta(days=d)
    if dd.day == 1:
        tick_j.append(d)
        tick_lb.append(dd.strftime("%b"))


# ══════════════════════════════════════════════════════════════════════════════
# 6. GRAPHIQUE 1 — Séries annuelles T et HR
# ══════════════════════════════════════════════════════════════════════════════

print("\n  Tracé : annuel_serie.pdf ...")

T_moy_j  = np.array([T_arr[d*steps_j:(d+1)*steps_j].mean()    for d in range(n_j)])
T_max_p  = np.array([T_arr[d*steps_j:(d+1)*steps_j].max()     for d in range(n_j)])
T_min_p  = np.array([T_arr[d*steps_j:(d+1)*steps_j].min()     for d in range(n_j)])
T_moy_e  = np.array([T_ext_f[d*steps_j:(d+1)*steps_j].mean()  for d in range(n_j)])
HR_moy_p = np.array([RH_arr[d*steps_j:(d+1)*steps_j].mean()   for d in range(n_j)])
HR_moy_e = np.array([RH_ext_f[d*steps_j:(d+1)*steps_j].mean() for d in range(n_j)])

fig1, (ax1T, ax1H) = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)
fig1.suptitle(
    f"Simulation annuelle — Béton de chanvre {EPAISSEUR_M*100:.0f} cm | "
    f"Pièce {Lx:.0f}×{Ly:.0f} m | {LABEL_METEO}",
    fontsize=12, fontweight="bold"
)

# Température
ax1T.fill_between(jours, T_min_p, T_max_p, color="#E07020", alpha=0.18,
                  label="Amplitude journalière pièce (min–max)")
ax1T.plot(jours, T_moy_j, color="#CC5500", lw=2.0, label="T moyenne pièce")
ax1T.plot(jours, T_moy_e, color="#888888", lw=1.2, ls="--", label="T moyenne extérieure")
ax1T.axhline(28, color="#CC3030", lw=1.0, ls=":", alpha=0.75, label="Seuil RE2020 = 28 °C")
ax1T.axhline(20, color="#1F5F99", lw=1.0, ls=":", alpha=0.75, label="Seuil chauffage = 20 °C")
ax1T.set_ylabel("Température [°C]", fontsize=11)
ax1T.set_xlim(0, n_j-1)
ax1T.set_xticks(tick_j); ax1T.set_xticklabels(tick_lb, fontsize=10)
ax1T.legend(fontsize=10, loc="upper left", framealpha=0.92)
ax1T.grid(True, alpha=0.18)
ax1T.text(0.99, 0.96,
    f"T max pièce : {T_arr.max():.1f} °C\nT min pièce : {T_arr.min():.1f} °C",
    transform=ax1T.transAxes, fontsize=9, ha="right", va="top",
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#BBBBBB", alpha=0.92))

# Humidité relative
ax1H.plot(jours, HR_moy_p, color="#334499", lw=2.0, label="HR moyenne pièce")
ax1H.plot(jours, HR_moy_e, color="#AAAAAA", lw=1.2, ls="--", label="HR moyenne extérieure")
ax1H.axhline(30, color="#E07020", lw=1.0, ls=":", alpha=0.75, label="Limite basse confort = 30 %")
ax1H.axhline(70, color="#CC3030", lw=1.0, ls=":", alpha=0.75, label="Limite haute confort = 70 %")
ax1H.set_ylim(0, 105)
ax1H.set_ylabel("Humidité relative [%]", fontsize=11)
ax1H.set_xlabel("Mois", fontsize=11)
ax1H.set_xlim(0, n_j-1)
ax1H.set_xticks(tick_j); ax1H.set_xticklabels(tick_lb, fontsize=10)
ax1H.legend(fontsize=10, loc="upper left", framealpha=0.92)
ax1H.grid(True, alpha=0.18)

_out = _pdf("annuel_serie")
try:    fig1.savefig(_out, dpi=150, bbox_inches="tight"); print(f"  [OK] {os.path.basename(_out)}")
except PermissionError:
    fig1.savefig(_out.replace(".pdf","_new.pdf"), dpi=150, bbox_inches="tight")
plt.close(fig1)


# ══════════════════════════════════════════════════════════════════════════════
# 7. GRAPHIQUE 2 — Profils parois Nord/Sud  (jour chaud et jour froid)
# ══════════════════════════════════════════════════════════════════════════════

print("  Tracé : annuel_profils.pdf ...")

HEURES_3 = [6, 14, 22]
C_CHAUD  = ["#1F5F99", "#E07020", "#CC3030"]   # 6h, 14h, 22h — jour chaud
C_FROID  = ["#6699CC", "#FF9944", "#EE6655"]   # idem — jour froid
PAROIS_NS = [("Sud", 0), ("Nord", 1)]

fig2, axes2 = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
fig2.suptitle(
    f"Profils de température et d'HR — parois Sud et Nord\n"
    f"Jour le + chaud : {date_c}  |  Jour le + froid : {date_f}  |  "
    f"Béton de chanvre {EPAISSEUR_M*100:.0f} cm",
    fontsize=11, fontweight="bold"
)

leg_chaud = [Line2D([0],[0], color=c, lw=2.5, label=f"{h:02d}h  {date_c}")
             for h, c in zip(HEURES_3, C_CHAUD)]
leg_froid = [Line2D([0],[0], color=c, lw=2.0, ls="--", label=f"{h:02d}h  {date_f}")
             for h, c in zip(HEURES_3, C_FROID)]

for col, (nom, idx) in enumerate(PAROIS_NS):
    wall = sim.walls[idx]
    ax_T = axes2[0, col]
    ax_H = axes2[1, col]

    for h, c_ch, c_fr in zip(HEURES_3, C_CHAUD, C_FROID):
        i_c = JOUR_C * steps_j + h
        i_f = JOUR_F * steps_j + h
        ax_T.plot(x_cm, wall.StockT[i_c].flatten()  - 273.15, color=c_ch, lw=2.5)
        ax_T.plot(x_cm, wall.StockT[i_f].flatten()  - 273.15, color=c_fr, lw=2.0, ls="--")
        ax_H.plot(x_cm, wall.StockRH[i_c].flatten() * 100,    color=c_ch, lw=2.5)
        ax_H.plot(x_cm, wall.StockRH[i_f].flatten() * 100,    color=c_fr, lw=2.0, ls="--")

    for ax in [ax_T, ax_H]:
        ax.axvline(0,               color="#CCCCCC", lw=1.2, ls="--")
        ax.axvline(EPAISSEUR_M*100, color="#CCCCCC", lw=1.2, ls="--")
        ax.set_xlim(0, EPAISSEUR_M*100)
        ax.set_xticks(np.linspace(0, EPAISSEUR_M*100, 7))
        ax.grid(True, alpha=0.18)

    ax_T.legend(handles=leg_chaud + leg_froid, fontsize=8.5, loc="best",
                framealpha=0.90, ncol=2)
    ax_T.set_title(f"Paroi {nom} — Température [°C]", fontsize=11, fontweight="bold")
    ax_T.set_ylabel("T [°C]", fontsize=11)
    ax_T.tick_params(labelbottom=False)

    ax_H.set_title(f"Paroi {nom} — Humidité relative [%]", fontsize=11, fontweight="bold")
    ax_H.set_ylabel("HR [%]", fontsize=11)
    ax_H.set_xlabel("Position dans la paroi [cm]   (ext → int)", fontsize=10)
    ax_H.set_ylim(0, 105)

    y_top = ax_T.get_ylim()[1]
    ax_T.text(0.8,   y_top, "EXT", fontsize=9, color="#888888", va="top")
    ax_T.text(EPAISSEUR_M*100-0.8, y_top, "INT", fontsize=9, color="#888888", va="top", ha="right")

_out = _pdf("annuel_profils")
try:    fig2.savefig(_out, dpi=150); print(f"  [OK] {os.path.basename(_out)}")
except PermissionError:
    fig2.savefig(_out.replace(".pdf","_new.pdf"), dpi=150)
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# 8. GRAPHIQUE 3 — Confort adaptatif EN 15251 (été) + besoins hiver
# ══════════════════════════════════════════════════════════════════════════════

print("  Tracé : annuel_confort.pdf ...")

# T_rm annuel (initialisation avec les derniers jours de décembre précédent
# — approximé ici par les premiers jours de janvier du fichier météo)
alpha    = 0.8
T_moy_j_ext = np.array([T_ext_f[d*steps_j:(d+1)*steps_j].mean() for d in range(n_j)])

T_rm_jour = np.zeros(n_j)
T_rm_jour[0] = T_moy_j_ext[0]
for d in range(1, n_j):
    T_rm_jour[d] = (1-alpha)*T_moy_j_ext[d-1] + alpha*T_rm_jour[d-1]
T_rm_h = np.repeat(T_rm_jour, steps_j)[:len(T_arr)]

T_comf = 0.33*T_rm_h + 18.8
T_sup  = T_comf + 3.0
T_inf  = T_comf - 3.0

# Filtre été : mai–septembre (DOY 121–273)
doy_h = np.array([(d // steps_j) + 1 for d in range(len(T_arr))])
masq_ete  = (doy_h >= DOY_SURVENT_DEBUT) & (doy_h <= DOY_SURVENT_FIN)
masq_hiver= ~masq_ete

dans_zone  = masq_ete & (T_arr >= T_inf) & (T_arr <= T_sup)
trop_chaud = masq_ete & (T_arr > T_sup)
trop_froid = masq_ete & (T_arr < T_inf)
hiver_froid= masq_hiver & (T_arr < 20.0)

n_ete = masq_ete.sum()
pct_c  = 100*dans_zone.sum()/n_ete if n_ete > 0 else 0
pct_ch = 100*trop_chaud.sum()/n_ete if n_ete > 0 else 0
pct_fr = 100*trop_froid.sum()/n_ete if n_ete > 0 else 0

fig3, ax3 = plt.subplots(figsize=(10, 8), constrained_layout=True)
T_rm_range = np.linspace(T_rm_h.min()-1, T_rm_h.max()+1, 200)
ax3.fill_between(T_rm_range,
                 0.33*T_rm_range+18.8-3, 0.33*T_rm_range+18.8+3,
                 color="#B8E0B0", alpha=0.55,
                 label="Zone de confort EN 15251 Cat. II (±3 °C)")
ax3.plot(T_rm_range, 0.33*T_rm_range+18.8,
         color="#2E8B57", lw=1.5, ls="--", label="T confort optimale")
ax3.scatter(T_rm_h[dans_zone],  T_arr[dans_zone],  c="#2E8B57", s=4,
            alpha=0.30, label=f"Été — en confort  {pct_c:.0f} %")
ax3.scatter(T_rm_h[trop_chaud], T_arr[trop_chaud], c="#CC3030", s=4,
            alpha=0.40, label=f"Été — trop chaud  {pct_ch:.0f} %")
ax3.scatter(T_rm_h[trop_froid], T_arr[trop_froid], c="#1F5F99", s=4,
            alpha=0.30, label=f"Été — trop froid  {pct_fr:.0f} %")
ax3.scatter(T_rm_h[hiver_froid], T_arr[hiver_froid], c="#AACCEE", s=3,
            alpha=0.20, label="Hiver — sous seuil chauffage 20 °C")
ax3.set_xlabel("T extérieure de référence T_rm [°C] (α = 0.8, EN 15251)", fontsize=11)
ax3.set_ylabel("Température de la pièce [°C]", fontsize=11)
ax3.set_title(
    f"Confort adaptatif EN 15251 Cat. II — Béton de chanvre {EPAISSEUR_M*100:.0f} cm\n"
    f"Année complète | Surventilation {ACH_NUIT} vol/h mai–sep (RE2020) | "
    f"Pièce {Lx:.0f}×{Ly:.0f} m",
    fontsize=11, fontweight="bold")
ax3.legend(fontsize=9.5, loc="upper left", framealpha=0.92, markerscale=3)
ax3.grid(True, alpha=0.20)
ax3.text(0.98, 0.03,
    f"Résultats été (mai–sep) :\n"
    f"  En confort : {pct_c:.0f} %\n"
    f"  Trop chaud : {pct_ch:.0f} %\n"
    f"  Trop froid : {pct_fr:.0f} %\n\n"
    f"DH RE2020 = {DH_total:.0f} °C·h\n"
    f"DJ hiver  = {DJ_total:.0f} °C·j",
    transform=ax3.transAxes, fontsize=9.5, va="bottom", ha="right",
    family="monospace",
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#BBBBBB", alpha=0.92))

_out = _pdf("annuel_confort")
try:    fig3.savefig(_out, dpi=150); print(f"  [OK] {os.path.basename(_out)}")
except PermissionError:
    fig3.savefig(_out.replace(".pdf","_new.pdf"), dpi=150)
plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# 9. GRAPHIQUE 4 — Degrés-heures été + degrés-jours hiver
# ══════════════════════════════════════════════════════════════════════════════

print("  Tracé : annuel_degres_heures.pdf ...")

DH_jour  = np.array([float(np.maximum(
    T_arr[d*steps_j:min((d+1)*steps_j, len(T_arr))]-28., 0.).sum())
    for d in range(n_j)])
DJ_jour  = np.array([float(np.maximum(
    18. - T_arr[d*steps_j:min((d+1)*steps_j, len(T_arr))], 0.).mean())
    for d in range(n_j)])   # [°C] par jour

DH_cumul = np.cumsum(DH_jour)

fig4, (ax4a, ax4b, ax4c) = plt.subplots(3, 1, figsize=(15, 11), constrained_layout=True)
fig4.suptitle(
    f"Indicateurs thermiques annuels — Béton de chanvre {EPAISSEUR_M*100:.0f} cm\n"
    f"DH été = {DH_total:.0f} °C·h ({DH_total/1250*100:.0f}% du seuil 1 250 °C·h)  |  "
    f"DJ hiver = {DJ_total:.0f} °C·j",
    fontsize=12, fontweight="bold"
)

# Barres DH estivaux
colors_dh = ["#CC3030" if v > 0 else "#AED6F1" for v in DH_jour]
ax4a.bar(jours, DH_jour, color=colors_dh, alpha=0.85, width=0.9)
ax4a.set_ylabel("DH journalier [°C·h]", fontsize=11)
ax4a.set_xlim(-0.5, n_j-0.5)
ax4a.set_xticks(tick_j); ax4a.set_xticklabels(tick_lb, fontsize=10)
ax4a.grid(True, alpha=0.18, axis="y")
ax4a.set_title("Degrés-heures d'inconfort estival > 28 °C  (rouge = inconfort)", fontsize=10)

# DH cumulés
ax4b.plot(jours, DH_cumul, color="#CC3030", lw=2.2, label="DH cumulés")
ax4b.axhline(1250, color="#333333", lw=1.5, ls="--",
             label="Seuil RE2020 = 1 250 °C·h/an")
ax4b.fill_between(jours, DH_cumul, 1250,
                  where=(DH_cumul >= 1250),
                  color="#CC3030", alpha=0.15, label="Dépassement")
ax4b.set_ylabel("DH cumulés [°C·h]", fontsize=11)
ax4b.set_xlim(-0.5, n_j-0.5)
ax4b.set_xticks(tick_j); ax4b.set_xticklabels(tick_lb, fontsize=10)
ax4b.legend(fontsize=10, framealpha=0.92)
ax4b.grid(True, alpha=0.18)

# Besoins de chauffage (degrés-jours)
ax4c.fill_between(jours, DJ_jour, 0, color="#1F5F99", alpha=0.55,
                  label=f"Besoin chauffage journalier [°C·j]")
ax4c.set_ylabel("Besoin chauffage [°C·j]", fontsize=11)
ax4c.set_xlabel("Mois", fontsize=11)
ax4c.set_xlim(-0.5, n_j-0.5)
ax4c.set_xticks(tick_j); ax4c.set_xticklabels(tick_lb, fontsize=10)
ax4c.legend(fontsize=10, framealpha=0.92)
ax4c.grid(True, alpha=0.18)
ax4c.set_title(f"Besoins de chauffage journaliers  (base 18 °C) — total {DJ_total:.0f} °C·j/an", fontsize=10)

_out = _pdf("annuel_degres_heures")
try:    fig4.savefig(_out, dpi=150); print(f"  [OK] {os.path.basename(_out)}")
except PermissionError:
    fig4.savefig(_out.replace(".pdf","_new.pdf"), dpi=150)
plt.close(fig4)


# ══════════════════════════════════════════════════════════════════════════════
# 10. GRAPHIQUE 5 — Données climatiques annuelles
# ══════════════════════════════════════════════════════════════════════════════

print("  Tracé : annuel_climat.pdf ...")

T_min_e  = np.array([T_ext_f[d*steps_j:(d+1)*steps_j].min()  for d in range(n_j)])
T_max_e  = np.array([T_ext_f[d*steps_j:(d+1)*steps_j].max()  for d in range(n_j)])
HR_min_e = np.array([RH_ext_f[d*steps_j:(d+1)*steps_j].min() for d in range(n_j)])
HR_max_e = np.array([RH_ext_f[d*steps_j:(d+1)*steps_j].max() for d in range(n_j)])

fig5, (ax5T, ax5H) = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True)
fig5.suptitle(
    f"Données climatiques — {LABEL_METEO}  |  Année complète ({n_j} jours)",
    fontsize=12, fontweight="bold"
)

ax5T.fill_between(jours, T_min_e, T_max_e, color="#E07020", alpha=0.20,
                  label=f"Plage journalière  amplitude moy. {(T_max_e-T_min_e).mean():.1f} °C")
ax5T.plot(jours, T_moy_j_ext, color="#CC5500", lw=2.0, label="Moyenne journalière")
ax5T.axhline(28, color="#CC3030", lw=1.0, ls=":", alpha=0.7, label="Seuil DH = 28 °C")
ax5T.axhline(0,  color="#1F5F99", lw=1.0, ls=":", alpha=0.7, label="0 °C")
ax5T.set_ylabel("Température extérieure [°C]", fontsize=11)
ax5T.set_xlim(0, n_j-1)
ax5T.set_xticks(tick_j); ax5T.set_xticklabels(tick_lb, fontsize=10)
ax5T.legend(fontsize=10, loc="upper right", framealpha=0.92)
ax5T.grid(True, alpha=0.18)

ax5H.fill_between(jours, HR_min_e, HR_max_e, color="#5577CC", alpha=0.20,
                  label="Plage journalière")
ax5H.plot(jours, HR_moy_e, color="#334499", lw=2.0, label="Moyenne journalière")
ax5H.set_ylim(0, 105)
ax5H.set_ylabel("Humidité relative extérieure [%]", fontsize=11)
ax5H.set_xlabel("Mois", fontsize=11)
ax5H.set_xlim(0, n_j-1)
ax5H.set_xticks(tick_j); ax5H.set_xticklabels(tick_lb, fontsize=10)
ax5H.legend(fontsize=10, loc="upper right", framealpha=0.92)
ax5H.grid(True, alpha=0.18)

_out = _pdf("annuel_climat")
try:    fig5.savefig(_out, dpi=150); print(f"  [OK] {os.path.basename(_out)}")
except PermissionError:
    fig5.savefig(_out.replace(".pdf","_new.pdf"), dpi=150)
plt.close(fig5)


# ══════════════════════════════════════════════════════════════════════════════
# 11. RÉSUMÉ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  RÉSULTATS ANNUELS")
print("-"*65)
print(f"  Pièce : {Lx:.0f}×{Ly:.0f} m  h={H:.1f} m  "
      f"S={SURFACE_SOL:.0f} m²  V={VOLUME_AIR:.0f} m³")
print(f"  λ = {LAMBDA_REF}  R = {R_mat:.2f} m²K/W  U = {U_ref:.3f} W/(m²K)")
print(f"  Ventilation : {ACH_JOUR} vol/h (hygiène)  +  "
      f"{ACH_NUIT} vol/h (nuit mai–sep RE2020)")
print(f"  Hygroscopique : {'ACTIF' if EFFET_HYGRO else 'DESACTIVE'}")
print(f"  Jour le + chaud : {date_c}  T_ext max = {T_ext_f.max():.1f} °C")
print(f"  Jour le + froid : {date_f}  T_ext min = {T_ext_f.min():.1f} °C")
print(f"  T max pièce : {T_arr.max():.1f} °C  |  T min pièce : {T_arr.min():.1f} °C")
print(f"  DH RE2020   : {DH_total:.0f} °C·h  ({DH_total/1250*100:.0f}% du seuil 1 250 °C·h)")
print(f"  DJ hiver    : {DJ_total:.0f} °C·j/an  (base 18 °C)")
print(f"  Confort été : {pct_c:.0f}% OK  |  {pct_ch:.0f}% trop chaud  |  {pct_fr:.0f}% trop froid")
print("-"*65)
print(f"  Météo : {LABEL_METEO}")
print("  FICHIERS")
for f in ["annuel_serie", "annuel_profils", "annuel_confort",
          "annuel_degres_heures", "annuel_climat"]:
    print(f"  {f}{SUF}.pdf")
print("="*65)
