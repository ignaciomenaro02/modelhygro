# -*- coding: utf-8 -*-
"""
sim_ete_libre.py
================
Simulation estivale sans climatisation — béton de chanvre 30 cm.
Période : juin–juillet (climat Mâcon prospectif +2°C).

Objectif
--------
Analyser le bénéfice de l'inertie thermique et hygrique du béton de chanvre
en mode libre (pas de chauffage, pas de climatisation) :

  1. Température de la pièce vs extérieur → atténuation + déphasage
  2. Profils de flux de chaleur dans la paroi à différents instants
  3. Profils d'humidité relative dans la paroi
  4. DH (degrés-heures d'inconfort, RE2020) sur la période simulée

Lancer dans Spyder :  Run → F5
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# ── Path setup ────────────────────────────────────────────────────────────────
data = os.path.dirname(os.path.abspath(__file__))
os.chdir(data)
if data not in sys.path:
    sys.path.insert(0, data)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from wall_config      import make_wall, WindowConfig, ThermalBridge, OpeningConfig
from sources          import (OccupantConfig, EquipmentConfig, LightingConfig,
                              VentilationConfig, build_daily_schedule)
from solar            import SolarCalculator
from re2020           import RE2020Evaluator
from room_simulation  import RoomSimulation
from weather          import load_weather_csv
import library as lib

sns.set_style("whitegrid")
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']


# ══════════════════════════════════════════════════════════════════════════════
# 0. Paramètres globaux
# ══════════════════════════════════════════════════════════════════════════════

DT           = 3600       # pas de temps [s] — 1 h suffit pour cette analyse
FLOOR_AREA   = 80.0       # [m²]
ROOM_VOLUME  = 200.0      # [m³]
INTERNAL_MASS= 110e3 * 80 # [J/K] — masse thermique des meubles/dalle (ISO 13790)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Données climatiques — filtrage juin–juillet
# ══════════════════════════════════════════════════════════════════════════════

WEATHER_FILE = "donnees-climatiques-prospectives-france-2c_macon.csv"
weather = load_weather_csv(WEATHER_FILE)

print(f"Météo chargée : {weather.station}  "
      f"T {weather.T_ext.min():.1f}…{weather.T_ext.max():.1f} °C")

# Début juin = 31+28+31+30+31 = 151 jours complets
# Fin juillet = 151 + 30 + 31 = 212 jours complets
T_JUNE1   = 151 * 86400.0   # [s] depuis le 1er janvier
T_AUG1    = 212 * 86400.0   # [s] depuis le 1er janvier

mask = (weather.time_s >= T_JUNE1) & (weather.time_s < T_AUG1)
time_bc  = weather.time_s[mask] - T_JUNE1   # rebase à 0 au 1er juin
Text_bc  = weather.T_ext[mask]
RHext_bc = weather.RH_ext[mask]

N_DAYS = int(time_bc[-1] / 86400)
print(f"Période simulée : {N_DAYS} jours (juin–juillet)  "
      f"T_ext {Text_bc.min():.1f}…{Text_bc.max():.1f} °C")


# ══════════════════════════════════════════════════════════════════════════════
# 2. Géométrie — 4 parois béton de chanvre 30 cm
# ══════════════════════════════════════════════════════════════════════════════
# Toutes les parois ont la même composition pour l'analyse comparative.
# La seule différence entre orientations est l'apport solaire.

walls = [
    make_wall("Sud — béton de chanvre 30 cm",  area=18.0, orientation='S',
              layers=[("Hempcrete", 0.30)], mesh_size=0.01),
    make_wall("Nord — béton de chanvre 30 cm", area=18.0, orientation='N',
              layers=[("Hempcrete", 0.30)], mesh_size=0.01),
    make_wall("Est — béton de chanvre 30 cm",  area=12.0, orientation='E',
              layers=[("Hempcrete", 0.30)], mesh_size=0.01),
    make_wall("Ouest — béton de chanvre 30 cm",area=12.0, orientation='W',
              layers=[("Hempcrete", 0.30)], mesh_size=0.01),
]

windows = [
    WindowConfig(name="Vitrage Sud", area=8.0, orientation='S',
                 U_value=1.1, g_value=0.60, shading=0.30),   # protection solaire
    WindowConfig(name="Vitrage Nord", area=2.0, orientation='N',
                 U_value=1.1, g_value=0.60, shading=0.0),
]

bridges = [
    ThermalBridge(name="Jonction sol/paroi", psi=0.15, length=40.0),
    ThermalBridge(name="Encadrements vitrages", psi=0.04, length=20.0),
]

openings = [OpeningConfig(name="Infiltrations", ach_contribution=0.1)]


# ══════════════════════════════════════════════════════════════════════════════
# 3. Sources internes  (occupants + ventilation — PAS DE CLIMATISATION)
# ══════════════════════════════════════════════════════════════════════════════

n_steps = int(time_bc[-1] / DT)

occ_sched = build_daily_schedule(hours_on=list(range(8, 22)),
                                  n_steps=n_steps, dt=DT)

occupants = OccupantConfig(n_occupants=2.0, sensible_per_person=80.0,
                           latent_per_person=60.0, schedule=occ_sched)

equipment = EquipmentConfig(power=200.0, schedule=occ_sched)   # usage réduit été

lighting  = LightingConfig(power_density=5.0, floor_area=FLOOR_AREA,
                           schedule=build_daily_schedule(hours_on=list(range(20, 24)),
                                                          n_steps=n_steps, dt=DT))

ventilation = VentilationConfig(
    n_ach=0.5, hrv_efficiency=0.0,   # VMC simple flux en été
    moisture_recovery=0.0, air_velocity=0.1,
)

# HVAC = None → simulation en mode libre (free-floating)

solar = SolarCalculator(
    latitude       = weather.lat,
    longitude      = weather.lon,
    start_doy      = 152,      # 1er juin = jour julien 152
    cloud_factor   = 0.30,
    ground_albedo  = 0.20,
    timezone_offset= 1.0,
)

re2020 = RE2020Evaluator(floor_area=FLOOR_AREA, climate_zone='H1b')


# ══════════════════════════════════════════════════════════════════════════════
# 4. Lancement de la simulation
# ══════════════════════════════════════════════════════════════════════════════

sim = RoomSimulation(
    wall_configs   = walls,
    window_configs = windows,
    bridge_configs = bridges,
    opening_configs= openings,
    occupants      = occupants,
    equipment      = equipment,
    lighting       = lighting,
    ventilation    = ventilation,
    hvac           = None,           # ← PAS DE CLIMATISATION
    solar_calc     = solar,
    volume         = ROOM_VOLUME,
    internal_mass  = INTERNAL_MASS,
    T_room_init    = Text_bc[0],     # initialisation à la T extérieure du 1er juin
    RH_room_init   = RHext_bc[0],
    re2020         = re2020,
)

for layer in sim.layers:
    print(layer)

T_room_arr, RH_room_arr = sim.run(
    time_bc  = time_bc,
    Text_bc  = Text_bc,
    RHext_bc = RHext_bc,
    dt       = DT,
    verbose  = True,
)

sim.print_summary()


# ══════════════════════════════════════════════════════════════════════════════
# 5. DH — degrés-heures d'inconfort (RE2020)
# ══════════════════════════════════════════════════════════════════════════════

T_arr  = np.array(sim.StockT_room)
RH_arr = np.array(sim.StockRH_room)
dt_h   = DT / 3600.0

DH_juin_juillet = float(np.maximum(T_arr - 28.0, 0.0).sum()) * dt_h  # [°C·h]

# Heures au-dessus de 28 °C
h_above_28 = float(np.sum(T_arr > 28.0)) * dt_h

print("\n" + "═"*55)
print("  DEGRÉS-HEURES D'INCONFORT  —  RE2020")
print("═"*55)
print(f"  Période            : juin–juillet ({N_DAYS} jours)")
print(f"  T_room max         : {T_arr.max():.1f} °C")
print(f"  T_room moyenne     : {T_arr.mean():.1f} °C")
print(f"  Heures > 28°C      : {h_above_28:.0f} h  "
      f"({h_above_28/(N_DAYS*24)*100:.1f}% de la période)")
print(f"  DH (juin-juillet)  : {DH_juin_juillet:.0f} °C·h")
print(f"  Seuil RE2020 annuel: 1250 °C·h  (zone H1b)")
print(f"  Ratio contribution : {DH_juin_juillet/1250*100:.0f}%  "
      f"du seuil annuel atteint sur 2 mois")
print("═"*55 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Axe temps + résumé T_ext interpolé
# ══════════════════════════════════════════════════════════════════════════════

n_out    = len(T_arr)
t_days   = np.arange(n_out) * DT / 86400.0            # axe en jours depuis le 1er juin
t_steps  = np.arange(n_out) * DT
Text_out = np.interp(t_steps, time_bc, Text_bc)        # T_ext sur les pas de simul.


# ══════════════════════════════════════════════════════════════════════════════
# 7. Figure 1 — Température de la pièce vs extérieur + seuil 28°C
# ══════════════════════════════════════════════════════════════════════════════

fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

ax1.plot(t_days, Text_out, lw=0.7, color=COLORS[1], alpha=0.7, label='T extérieure')
ax1.plot(t_days, T_arr,    lw=1.0, color=COLORS[0],             label='T pièce (libre)')
ax1.axhline(28, ls='--', lw=1.0, color='red', label='Seuil inconf. RE2020 = 28°C')
ax1.axhline(26, ls=':',  lw=0.8, color='orange', label='Setpoint clim. typique 26°C')
ax1.fill_between(t_days, T_arr, 28,
                 where=(T_arr > 28), color='red', alpha=0.15, label='Zone inconfort')
ax1.set_ylabel('Température [°C]', fontsize=12)
ax1.legend(fontsize=9, ncol=2)
ax1.set_title(
    f'Simulation libre — béton de chanvre 30 cm — juin–juillet (Mâcon)\n'
    f'DH = {DH_juin_juillet:.0f} °C·h  |  T_max pièce = {T_arr.max():.1f}°C  |  '
    f'T_max ext = {Text_out.max():.1f}°C  |  '
    f'Atténuation ΔT = {Text_out.max()-T_arr.max():.1f}°C',
    fontsize=11)

ax2.plot(t_days, RH_arr*100, lw=0.9, color=COLORS[4], label='HR pièce')
ax2.axhline(30, ls='--', lw=0.7, color='gray')
ax2.axhline(70, ls='--', lw=0.7, color='red')
ax2.set_ylim(0, 100)
ax2.set_ylabel('Humidité relative [%]', fontsize=12)
ax2.set_xlabel('Jours depuis le 1er juin', fontsize=12)
ax2.legend(fontsize=9)

plt.tight_layout()
fig1.savefig('ete_libre_climat_piece.pdf', dpi=150)
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 8. Analyse de l'inertie thermique — déphasage et atténuation (paroi Sud)
# ══════════════════════════════════════════════════════════════════════════════
# On compare l'amplitude de T_ext, T_surf_ext, T_surf_int et T_room sur
# une semaine représentative (semaine la plus chaude).
# ══════════════════════════════════════════════════════════════════════════════

# Identifier la semaine la plus chaude (T_room moyenne maximale sur 7 jours)
steps_per_day  = int(86400 / DT)
steps_per_week = 7 * steps_per_day
n_weeks        = max(1, (n_out - steps_per_week) // steps_per_day)

best_start = 0
best_mean  = -999.0
for i in range(n_weeks):
    s = i * steps_per_day
    m = T_arr[s:s+steps_per_week].mean()
    if m > best_mean:
        best_mean  = m
        best_start = s

s_w = best_start
e_w = min(best_start + steps_per_week, n_out)
t_week = t_days[s_w:e_w]

# Surfaces de la paroi Sud (premier mur, index 0)
wall_S   = sim.walls[0]
layer_S  = sim.layers[0]
cfg_S    = sim.wall_configs[0]

T_surf_int_S = np.array(sim.StockT_walls[cfg_S.name])[s_w:e_w]
T_surf_ext_S = np.array([wall_S.StockT[i][0, 0] - 273.15
                          for i in range(s_w, e_w)])

T_ext_week   = Text_out[s_w:e_w]
T_room_week  = T_arr[s_w:e_w]

# Amplitude : sur toute la semaine
amp_ext      = T_ext_week.max()      - T_ext_week.min()
amp_surf_int = T_surf_int_S.max()    - T_surf_int_S.min()
amp_room     = T_room_week.max()     - T_room_week.min()
attenuation  = amp_surf_int / amp_ext if amp_ext > 0 else float('nan')

# Déphasage : corrélation croisée sur un cycle de 24h
#   On soustrait la tendance pour ne garder que la composante oscillante journalière.
steps_day = int(86400 / DT)   # 24 si DT=3600
n_corr    = min(len(T_ext_week), len(T_surf_int_S))
ref  = T_ext_week[:n_corr]    - T_ext_week[:n_corr].mean()
sig  = T_surf_int_S[:n_corr]  - T_surf_int_S[:n_corr].mean()
corr = np.correlate(sig, ref, mode='full')
lags = np.arange(-(n_corr - 1), n_corr)
# Cherche le déphasage dans la plage ±0.5 jour
half = steps_day // 2
mid  = n_corr - 1
best_lag = lags[mid - half + int(np.argmax(corr[mid - half: mid + half + 1]))]
dephasage_h = float(best_lag) * DT / 3600.0

print(f"Paroi Sud — semaine la plus chaude (début jour {best_start//steps_per_day:.0f})")
print(f"  Amplitude T_ext          : {amp_ext:.1f} °C")
print(f"  Amplitude T_surf_int     : {amp_surf_int:.1f} °C")
print(f"  Atténuation              : {attenuation:.2f}  "
      f"(= {amp_surf_int:.1f}/{amp_ext:.1f})")
print(f"  Déphasage approx.        : {dephasage_h:.1f} h\n")

fig2, ax = plt.subplots(figsize=(13, 5))
ax.plot(t_week, T_ext_week,      lw=1.0, color=COLORS[1], ls='--', label='T extérieure')
ax.plot(t_week, T_surf_ext_S,    lw=1.0, color=COLORS[2], ls=':',  label='T surf. ext. paroi Sud')
ax.plot(t_week, T_surf_int_S,    lw=1.2, color=COLORS[0],          label='T surf. int. paroi Sud')
ax.plot(t_week, T_room_week,     lw=1.4, color=COLORS[3],          label='T pièce')
ax.axhline(28, ls='--', lw=0.8, color='red', alpha=0.6, label='28°C (seuil DH)')
ax.fill_between(t_week, T_room_week, 28,
                where=(T_room_week > 28), color='red', alpha=0.12)
ax.set_ylabel('Température [°C]', fontsize=12)
ax.set_xlabel('Jours depuis le 1er juin', fontsize=12)
ax.set_title(
    f'Inertie thermique — béton de chanvre 30 cm — semaine la plus chaude\n'
    f'Atténuation surface intérieure : ×{attenuation:.2f}  |  '
    f'Déphasage ≈ {dephasage_h:.1f} h',
    fontsize=11)
ax.legend(fontsize=9, ncol=2)
plt.tight_layout()
fig2.savefig('ete_libre_inertie.pdf', dpi=150)
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 9. Fonction utilitaire : profils de flux dans la paroi
# ══════════════════════════════════════════════════════════════════════════════

def _wall_flux_profiles(wall_obj, layer, step):
    """
    Calcule les profils de flux thermique et de flux de vapeur à un instant donné.

    Returns
    -------
    x_mid  : (N-1,) array   positions des interfaces entre nœuds [cm]
    q_sens : (N-1,) array   flux de chaleur sensible [W/m²]   (+ = vers int.)
    q_lat  : (N-1,) array   flux latent  Lv·g           [W/m²]
    q_tot  : (N-1,) array   flux total = sensible + latent  [W/m²]
    RH_nod : (N,)   array   HR aux nœuds [%]
    x_nod  : (N,)   array   positions des nœuds [cm]
    """
    idx = min(step, len(wall_obj.StockT) - 1)
    T   = wall_obj.StockT[idx].flatten()      # [K]
    RH  = wall_obj.StockRH[idx].flatten()     # [-]
    dx  = layer.dx.flatten()                # [m]  dx[0]=0 (surface node)

    k_f   = layer.k(T.reshape(-1,1), RH.reshape(-1,1)).flatten()
    dp_f  = layer.delta_p(T.reshape(-1,1), RH.reshape(-1,1)).flatten()
    Pv_f  = lib.Pv(T.reshape(-1,1), RH.reshape(-1,1)).flatten()

    N = layer.N_tot
    x_nod = layer.x_pos * 100.0   # [cm]

    q_sens = np.zeros(N - 1)
    q_lat  = np.zeros(N - 1)
    x_mid  = np.zeros(N - 1)

    for i in range(N - 1):
        delta_x = dx[i + 1] if dx[i + 1] > 0 else 1e-4
        k_eff   = 0.5 * (k_f[i] + k_f[i + 1])
        dp_eff  = 0.5 * (dp_f[i] + dp_f[i + 1])

        q_sens[i] = -k_eff  * (T[i+1]   - T[i])   / delta_x         # [W/m²]
        g_v       = -dp_eff * (Pv_f[i+1] - Pv_f[i]) / delta_x       # [kg/(m²·s)]
        q_lat[i]  = lib.Lv * g_v                                      # [W/m²]
        x_mid[i]  = 0.5 * (x_nod[i] + x_nod[i + 1])

    return x_mid, q_sens, q_lat, q_sens + q_lat, RH * 100.0, x_nod


# ══════════════════════════════════════════════════════════════════════════════
# 10. Figure 3 — Profils de flux thermique et HR dans la paroi Sud
#     Instantanés : 14h (pic), 3h (nuit) sur 3 jours représentatifs
# ══════════════════════════════════════════════════════════════════════════════

# Repérer le jour le plus chaud de la simulation
i_max_room = int(np.argmax(T_arr))
day_hot    = int(i_max_room // steps_per_day)

# 3 jours d'intérêt : 1 semaine avant pic, pic, et semaine après
# (bornés au domaine disponible)
days_of_interest = sorted({
    max(0, day_hot - 7),
    day_hot,
    min(N_DAYS - 2, day_hot + 7)
})

# Pour chaque jour : snapshot 14h (forte chaleur) et 3h (nuit)
hour_mid  = 14   # heure du maximum diurne
hour_night= 3    # heure du minimum nocturne

SNAPSHOTS = []
for d in days_of_interest:
    step_mid   = d * steps_per_day + hour_mid
    step_night = d * steps_per_day + hour_night
    SNAPSHOTS.append((f"J{d+1:02d} — 14h", step_mid,   'solid'))
    SNAPSHOTS.append((f"J{d+1:02d} — 03h", step_night, 'dashed'))

fig3, axes3 = plt.subplots(3, 2, figsize=(14, 12))

for row, (label, step, ls) in enumerate(SNAPSHOTS):
    ax_q  = axes3[row // 2, 0]   # flux thermique
    ax_RH = axes3[row // 2, 1]   # HR
    color = COLORS[row % len(COLORS)]

    x_mid, q_sens, q_lat, q_tot, RH_nod, x_nod = _wall_flux_profiles(
        sim.walls[0], sim.layers[0], step)

    T_room_snap = T_arr[min(step, n_out-1)]
    T_ext_snap  = Text_out[min(step, n_out-1)]

    # Flux
    ax_q.plot(x_mid, q_sens, lw=1.4, color=color, ls=ls,
              label=f'{label}  |  T_ext={T_ext_snap:.1f}°C  T_pièce={T_room_snap:.1f}°C')
    ax_q.plot(x_mid, q_lat,  lw=0.9, color=color, ls=ls, alpha=0.4)
    ax_q.fill_between(x_mid, q_sens, q_tot, alpha=0.15, color=color,
                      label='_nolegend_')

    # HR
    ax_RH.plot(x_nod, RH_nod, lw=1.4, color=color, ls=ls, label=label)

# Mise en forme des 3 lignes de graphes
for row in range(3):
    ax_q  = axes3[row, 0]
    ax_RH = axes3[row, 1]

    for xb in sim.layers[0].interface_pos[:-1] * 100:
        ax_q.axvline(xb,  ls=':', lw=0.7, color='gray', alpha=0.6)
        ax_RH.axvline(xb, ls=':', lw=0.7, color='gray', alpha=0.6)

    ax_q.axhline(0, lw=0.5, color='black', alpha=0.4)
    ax_q.set_xlabel('Position [cm]', fontsize=11)
    ax_q.set_ylabel('Flux [W/m²]', fontsize=11)
    ax_q.legend(fontsize=8)
    title_day = days_of_interest[row] + 1
    ax_q.set_title(f'Flux sensible (trait plein) + latent (zone) — jour {title_day}',
                   fontsize=10)

    ax_RH.set_xlabel('Position [cm]', fontsize=11)
    ax_RH.set_ylabel('HR [%]', fontsize=11)
    ax_RH.set_ylim(0, 100)
    ax_RH.legend(fontsize=8)
    ax_RH.set_title(f'Humidité relative dans la paroi — jour {title_day}',
                    fontsize=10)

    # Annotation ext/int (coordonnées en axes : 0–1)
    for ax in (ax_q, ax_RH):
        ax.text(0.02, 0.95, 'EXT', transform=ax.transAxes,
                ha='left',  va='top', fontsize=8, color='gray')
        ax.text(0.98, 0.95, 'INT', transform=ax.transAxes,
                ha='right', va='top', fontsize=8, color='gray')

fig3.suptitle(
    'Profils de flux thermique et HR dans la paroi Sud — béton de chanvre 30 cm\n'
    f'(extérieur → intérieur  |  flux + = vers l\'intérieur)',
    fontsize=12, y=1.01)
plt.tight_layout()
fig3.savefig('ete_libre_profils_paroi.pdf', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 11. Figure 4 — Bilan DH : bar chart journalier + courbe cumulée
# ══════════════════════════════════════════════════════════════════════════════

# DH par jour
DH_daily  = np.zeros(N_DAYS)
for d in range(N_DAYS):
    s = d * steps_per_day
    e = min(s + steps_per_day, n_out)
    DH_daily[d] = float(np.maximum(T_arr[s:e] - 28.0, 0.0).sum()) * dt_h

DH_cumul = np.cumsum(DH_daily)
days_axis = np.arange(1, N_DAYS + 1)

fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

colors_bar = ['#d62728' if d > 0 else '#aec7e8' for d in DH_daily]
ax4a.bar(days_axis, DH_daily, color=colors_bar, alpha=0.85, width=0.9,
         label='DH quotidien [°C·h]')
ax4a.set_ylabel('DH journalier [°C·h]', fontsize=12)
ax4a.set_title(
    f'Degrés-heures d\'inconfort RE2020  —  DH total (juin–juil.) = '
    f'{DH_juin_juillet:.0f} °C·h\n'
    f'(seuil annuel RE2020 zone H1b = 1250 °C·h)', fontsize=11)
ax4a.legend(fontsize=9)

ax4b.plot(days_axis, DH_cumul, lw=1.5, color=COLORS[3], label='DH cumulés [°C·h]')
ax4b.axhline(1250, ls='--', lw=1.2, color='black',
             label='Seuil annuel RE2020 = 1250 °C·h')
ax4b.fill_between(days_axis, DH_cumul,
                  where=(DH_cumul >= 1250), color='red', alpha=0.15,
                  label='Dépassement seuil')
ax4b.set_ylabel('DH cumulés [°C·h]', fontsize=12)
ax4b.set_xlabel('Jours depuis le 1er juin', fontsize=12)
ax4b.legend(fontsize=9)

plt.tight_layout()
fig4.savefig('ete_libre_DH.pdf', dpi=150)
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# 12. Résumé console
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "═"*60)
print("  RÉSUMÉ — SIMULATION ÉTÉ LIBRE — BÉTON DE CHANVRE 30 cm")
print("═"*60)
for cfg, layer in zip(sim.wall_configs, sim.layers):
    print(f"  {cfg.name:<40}  U = {layer.U_value():.3f} W/(m²K)")
print(f"\n  T_ext (juin–juillet)  : min {Text_bc.min():.1f}  "
      f"max {Text_bc.max():.1f}  moy {Text_bc.mean():.1f} °C")
print(f"  T_pièce (libre)       : min {T_arr.min():.1f}  "
      f"max {T_arr.max():.1f}  moy {T_arr.mean():.1f} °C")
print(f"  Atténuation max ΔT    : {Text_bc.max() - T_arr.max():.1f} °C  "
      f"(= T_max_ext − T_max_pièce)")
print(f"  Déphasage (paroi Sud) : ≈ {dephasage_h:.1f} h")
print(f"  DH (juin–juillet)     : {DH_juin_juillet:.0f} °C·h  "
      f"/ 1250 seuil annuel RE2020")
print(f"  HR pièce              : moy {RH_arr.mean()*100:.1f}%  "
      f"max {RH_arr.max()*100:.1f}%  min {RH_arr.min()*100:.1f}%")
print("═"*60)
print("\nFichiers sauvegardés :")
print("  ete_libre_climat_piece.pdf  — T pièce vs T ext + HR")
print("  ete_libre_inertie.pdf       — Semaine la plus chaude (déphasage)")
print("  ete_libre_profils_paroi.pdf — Flux thermique + HR dans la paroi")
print("  ete_libre_DH.pdf            — Degrés-heures journaliers + cumulés")
