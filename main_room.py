# -*- coding: utf-8 -*-
"""
main_room.py
============
Simulation hygro-thermique d'une pièce — béton de chanvre 30 cm
Période : juin–juillet (libre, sans climatisation)
Pas de temps : 1 heure (DT = 3600 s, calé sur le fichier météo horaire)

COMMENT MODIFIER LA CONFIGURATION
----------------------------------
Section 1  → parois, vitrages, ponts thermiques
Section 2  → occupants, équipements, ventilation
Section 3  → réglages simulation (volume, masse thermique, solaire)
Section 4  → RE2020 (zone climatique, surface)
Sections 5–7 → lancées automatiquement (simulation + sorties)

AJOUTER UNE PAROI
-----------------
    make_wall("Ma paroi", area=15.0, orientation='E',
              layers=[("Rock_Wool", 0.12), ("BA13", 0.013)])

MATÉRIAUX DISPONIBLES
---------------------
    Hempcrete  Rammed_Earth  Rock_Wool   Wood_Fiber
    Concrete   Wood          Vapor_Barrier  Earth_Plaster
    Gypsum_Plaster  Lime_Plaster  Fermacell  BA13
"""

import os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless — enlever si vous lancez depuis Spyder
import matplotlib.pyplot as plt

data = os.path.dirname(os.path.abspath(__file__))
os.chdir(data)
if data not in sys.path:
    sys.path.insert(0, data)

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from wall_config      import make_wall, WindowConfig, ThermalBridge, OpeningConfig, hm_lewis
from sources          import (OccupantConfig, EquipmentConfig, LightingConfig,
                              VentilationConfig, build_daily_schedule)
from solar            import SolarCalculator
from re2020           import RE2020Evaluator
from room_simulation  import RoomSimulation
from outputs          import (plot_wall_profiles, plot_re2020_comparison,
                               plot_comfort_map, table_energy_balance,
                               export_csv, plot_walls_3d,
                               plot_wall_hygrothermal_profiles)
from weather          import load_weather_csv


###############################################################################
# ──────────────────────  0. Paramètres globaux + Météo  ──────────────────────
###############################################################################

DT          = 3600      # [s] — pas de temps = 1 h (calé sur les données météo)
FLOOR_AREA  = 80.0      # [m²] surface habitable SHON_RT
ROOM_VOLUME = 200.0     # [m³] volume d'air de la zone
# Masse thermique interne effective (mobilier, dalle, cloisons)
# ISO 13790 bâtiment 'moyen' ≈ 110 kJ/(m²·K) × surface
INTERNAL_MASS = 110e3 * FLOOR_AREA   # [J/K]

# ── Fichier météo horaire (Mâcon, prospectif +2°C) ───────────────────────────
WEATHER_FILE = "donnees-climatiques-prospectives-france-2c_macon.csv"
weather  = load_weather_csv(WEATHER_FILE)

print(f"Météo : {weather.station}  ({weather.lat:.2f}N, {weather.lon:.2f}E)")
print(f"        T_ext {weather.T_ext.min():.1f}...{weather.T_ext.max():.1f} degC  "
      f"sur {weather.n_hours} h")

# ── Extraction juin–juillet ───────────────────────────────────────────────────
# Jan31 + Fev28 + Mar31 + Avr30 + Mai31 = 151 jours -> 1er juin = 151*86400 s
# Juin30 + Juil31 = 61 jours -> 1er aout = 212*86400 s
T_JUNE1 = 151 * 86400.0
T_AUG1  = 212 * 86400.0

mask     = (weather.time_s >= T_JUNE1) & (weather.time_s < T_AUG1)
time_bc  = weather.time_s[mask] - T_JUNE1   # rebase a 0 au 1er juin [s]
Text_bc  = weather.T_ext[mask]               # [degC]
RHext_bc = weather.RH_ext[mask]              # [-]
N_DAYS   = int(time_bc[-1] / 86400)

print(f"Periode simulee : {N_DAYS} jours (juin-juillet)")
print(f"  T_ext {Text_bc.min():.1f}...{Text_bc.max():.1f} degC  "
      f"HR_ext {RHext_bc.min()*100:.0f}...{RHext_bc.max()*100:.0f} %\n")


###############################################################################
# ────────────────────────  1. Géométrie du bâtiment  ─────────────────────────
###############################################################################
# Toutes les parois : béton de chanvre (Hempcrete) 30 cm
# Sens : extérieur -> intérieur

# Coefficients de transfert convectif (Lewis)
# h_ext=25 W/(m²K) ISO 6946 vent ~4m/s  →  hm_ext = 25 × Lewis
# h_int=8  W/(m²K) ISO 6946 air calme   →  hm_int = 8  × Lewis
HM_EXT = hm_lewis(25.0)   # ≈ 1.53e-7 kg/(m²·s·Pa)
HM_INT = hm_lewis(8.0)    # ≈ 4.91e-8 kg/(m²·s·Pa)
print(f"Coefficients massiques (Lewis) :  hm_ext = {HM_EXT:.2e}  hm_int = {HM_INT:.2e}  kg/(m2.s.Pa)")

walls = [
    make_wall("Sud",   area=18.0, orientation='S',
              layers=[("Hempcrete", 0.30)], mesh_size=0.01,
              h_ext=25.0, hm_ext=HM_EXT, h_int=8.0, hm_int=HM_INT),
    make_wall("Nord",  area=18.0, orientation='N',
              layers=[("Hempcrete", 0.30)], mesh_size=0.01,
              h_ext=25.0, hm_ext=HM_EXT, h_int=8.0, hm_int=HM_INT),
    make_wall("Est",   area=12.0, orientation='E',
              layers=[("Hempcrete", 0.30)], mesh_size=0.01,
              h_ext=25.0, hm_ext=HM_EXT, h_int=8.0, hm_int=HM_INT),
    make_wall("Ouest", area=12.0, orientation='W',
              layers=[("Hempcrete", 0.30)], mesh_size=0.01,
              h_ext=25.0, hm_ext=HM_EXT, h_int=8.0, hm_int=HM_INT),
]

windows = [
    WindowConfig(name="Vitrage Sud",  area=8.0, orientation='S',
                 U_value=1.1, g_value=0.60, shading=0.30),
    WindowConfig(name="Vitrage Nord", area=2.0, orientation='N',
                 U_value=1.1, g_value=0.60, shading=0.0),
]

bridges = [
    ThermalBridge(name="Jonction sol/paroi",    psi=0.15, length=40.0),
    ThermalBridge(name="Encadrements vitrages", psi=0.04, length=20.0),
]

openings = [OpeningConfig(name="Infiltrations", ach_contribution=0.1)]


###############################################################################
# ────────────────────────  2. Sources internes  ───────────────────────────────
###############################################################################

n_steps = int(time_bc[-1] / DT)   # nombre de pas sur la periode simulee

occ_sched = build_daily_schedule(hours_on=list(range(8, 22)),
                                  n_steps=n_steps, dt=DT)
lum_sched = build_daily_schedule(hours_on=list(range(20, 24)),
                                  n_steps=n_steps, dt=DT)

occupants = OccupantConfig(n_occupants=2.0,
                           sensible_per_person=80.0,
                           latent_per_person=60.0,
                           schedule=occ_sched)

equipment = EquipmentConfig(power=200.0, schedule=occ_sched)

lighting  = LightingConfig(power_density=5.0,
                           floor_area=FLOOR_AREA,
                           schedule=lum_sched)

ventilation = VentilationConfig(
    n_ach=0.5,            # [1/h] debit minimal RE2020
    hrv_efficiency=0.0,   # VMC simple flux en ete
    moisture_recovery=0.0,
    air_velocity=0.1,
)

# HVAC = None -> simulation en mode LIBRE (pas de chauffage, pas de clim)
hvac = None


###############################################################################
# ────────────────────────  3. Réglages simulation  ───────────────────────────
###############################################################################

solar = SolarCalculator(
    latitude       = weather.lat,
    longitude      = weather.lon,
    start_doy      = 152,    # 1er juin = jour julien 152
    cloud_factor   = 0.30,
    ground_albedo  = 0.20,
    timezone_offset= 1.0,
)


###############################################################################
# ────────────────────────  4. RE2020  ────────────────────────────────────────
###############################################################################
# En simulation libre (HVAC=None), les indicateurs RE2020 calcules ici sont :
#   DH  : degres-heures d'inconfort au-dessus de 28 degC (seuil 1250 degC.h)
#         -> mesure le confort passif de l'enveloppe en ete
#   Cep : = 0 car aucun systeme actif n'est utilise
#   BBio: = B_light uniquement (pas de besoin chaud/froid injecte)
#
# Pour un calcul RE2020 complet (annee entiere avec HVAC), utiliser
# hvac=HVACConfig(...) et simuler sur 8760 h.

CLIMATE_ZONE = 'H1b'
re2020 = RE2020Evaluator(floor_area=FLOOR_AREA, climate_zone=CLIMATE_ZONE)
re2020.set_carrier(heating='electricity', cooling='electricity')


###############################################################################
# ────────────────────────  5. Lancement de la simulation  ────────────────────
###############################################################################

sim = RoomSimulation(
    wall_configs    = walls,
    window_configs  = windows,
    bridge_configs  = bridges,
    opening_configs = openings,
    occupants       = occupants,
    equipment       = equipment,
    lighting        = lighting,
    ventilation     = ventilation,
    hvac            = hvac,
    solar_calc      = solar,
    volume          = ROOM_VOLUME,
    internal_mass   = INTERNAL_MASS,
    T_room_init     = float(Text_bc[0]),
    RH_room_init    = float(RHext_bc[0]),
    re2020          = re2020,
)

print("Parois construites :")
for layer in sim.layers:
    print(f"  {layer}")

T_room_arr, RH_room_arr = sim.run(
    time_bc  = time_bc,
    Text_bc  = Text_bc,
    RHext_bc = RHext_bc,
    dt       = DT,
    verbose  = True,
)

sim.print_summary()


###############################################################################
# ────────────────────────  6. RE2020 — rapport  ──────────────────────────────
###############################################################################

hours_light = float(sum(lighting.schedule)) * DT / 3600.0
E_light_kWh = lighting.power_density * FLOOR_AREA * hours_light / 1000.0
re2020.add_lighting_kWh(E_light_kWh)

report = re2020.report()
re2020.print_report(report)

T_arr = np.array(sim.StockT_room)
dt_h  = DT / 3600.0
DH    = float(np.maximum(T_arr - 28.0, 0.0).sum()) * dt_h
h_28  = float(np.sum(T_arr > 28.0)) * dt_h

print("=" * 56)
print(f"  DH (juin-juillet) = {DH:.0f} degC.h  "
      f"-> {DH/1250*100:.1f}% du seuil annuel RE2020 (1250 degC.h, H1b)")
print(f"  Heures > 28 degC  = {h_28:.0f} h  ({h_28/(N_DAYS*24)*100:.1f}% de la periode)")
print(f"  T_piece max       = {T_arr.max():.1f} degC  "
      f"(attenuation {Text_bc.max()-T_arr.max():.1f} degC vs T_ext {Text_bc.max():.1f} degC)")
print("=" * 56 + "\n")


###############################################################################
# ────────────────────────  7. Sorties graphiques  ────────────────────────────
###############################################################################

n_out    = len(T_arr)
t_days   = np.arange(n_out) * DT / 86400.0
t_steps  = np.arange(n_out) * DT
Text_out = np.interp(t_steps, time_bc, Text_bc)
steps_day = int(86400 / DT)

# ── 7a. Vue 3D des parois ─────────────────────────────────────────────────────
plot_walls_3d(sim, save='walls_3d.pdf')

# ── 7b. Climat de la pièce — T + HR avec T_ext superposé ─────────────────────
fig_c, (ax_c1, ax_c2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

ax_c1.plot(t_days, Text_out, lw=0.7, color='#ff7f0e', alpha=0.6, label='T exterieure')
ax_c1.plot(t_days, T_arr,    lw=1.0, color='#1f77b4',            label='T piece (libre)')
ax_c1.axhline(28, ls='--', lw=0.9, color='red',    label='Seuil DH 28C')
ax_c1.axhline(26, ls=':',  lw=0.7, color='orange', label='Confort max 26C')
ax_c1.fill_between(t_days, T_arr, 28,
                   where=(T_arr > 28), color='red', alpha=0.12, label='Inconfort')
ax_c1.set_ylabel('Temperature [degC]', fontsize=12)
ax_c1.legend(fontsize=9, ncol=3)
ax_c1.set_title(
    f'Beton de chanvre 30 cm — Simulation libre juin-juillet (Macon +2C)\n'
    f'T_max ext={Text_bc.max():.1f}C   T_max piece={T_arr.max():.1f}C   '
    f'Attenuation={Text_bc.max()-T_arr.max():.1f}C   DH={DH:.0f} degC.h',
    fontsize=11)

ax_c2.plot(t_days, RH_room_arr * 100, lw=0.9, color='#9467bd', label='HR piece')
ax_c2.axhline(30, ls='--', lw=0.7, color='gray', label='30% min confort')
ax_c2.axhline(70, ls='--', lw=0.7, color='red',  label='70% max confort')
ax_c2.set_ylim(0, 100)
ax_c2.set_ylabel('Humidite relative [%]', fontsize=12)
ax_c2.set_xlabel('Jours depuis le 1er juin', fontsize=12)
ax_c2.legend(fontsize=9)
plt.tight_layout()
fig_c.savefig('room_climate.pdf', dpi=150)
plt.show()

# ── 7c. Inertie thermique — semaine la plus chaude ───────────────────────────
steps_week = 7 * steps_day
n_weeks    = max(1, (n_out - steps_week) // steps_day)

best_s, best_m = 0, -999.0
for i in range(n_weeks):
    m = T_arr[i*steps_day : i*steps_day + steps_week].mean()
    if m > best_m:
        best_m = m; best_s = i * steps_day

e_w    = min(best_s + steps_week, n_out)
t_week = t_days[best_s:e_w]
cfg0   = sim.wall_configs[0]
wall0  = sim.walls[0]
T_si   = np.array(sim.StockT_walls[cfg0.name])[best_s:e_w]
T_se   = np.array([wall0.StockT[i][0, 0] - 273.15 for i in range(best_s, e_w)])
T_e_w  = Text_out[best_s:e_w]
T_r_w  = T_arr[best_s:e_w]

amp_ext = T_e_w.max() - T_e_w.min()
amp_si  = T_si.max()  - T_si.min()
atten   = amp_si / amp_ext if amp_ext > 0 else float('nan')

n_c  = len(T_e_w)
half = min(12, n_c // 4)
ref  = T_e_w - T_e_w.mean()
sig  = T_si  - T_si.mean()
corr = np.correlate(sig, ref, mode='full')
lags = np.arange(-(n_c - 1), n_c)
mid  = n_c - 1
best_lag  = lags[mid - half + int(np.argmax(corr[mid - half: mid + half + 1]))]
dephasage = float(best_lag) * DT / 3600.0

fig_i, ax_i = plt.subplots(figsize=(13, 5))
ax_i.plot(t_week, T_e_w, lw=0.9, color='#ff7f0e', ls='--', label='T exterieure')
ax_i.plot(t_week, T_se,  lw=0.9, color='#2ca02c', ls=':',  label='T surf. ext.')
ax_i.plot(t_week, T_si,  lw=1.2, color='#1f77b4',          label='T surf. int.')
ax_i.plot(t_week, T_r_w, lw=1.4, color='#d62728',          label='T piece')
ax_i.axhline(28, ls='--', lw=0.8, color='red', alpha=0.5, label='28C seuil DH')
ax_i.fill_between(t_week, T_r_w, 28, where=(T_r_w > 28), color='red', alpha=0.10)
ax_i.set_ylabel('Temperature [degC]', fontsize=12)
ax_i.set_xlabel('Jours depuis le 1er juin', fontsize=12)
ax_i.set_title(
    f'Inertie thermique — paroi Sud — semaine la plus chaude\n'
    f'Amplitude T_ext={amp_ext:.1f}C -> T_surf_int={amp_si:.1f}C '
    f'(x{atten:.2f})   Dephasage ~ {dephasage:+.0f} h',
    fontsize=11)
ax_i.legend(fontsize=9, ncol=3)
plt.tight_layout()
fig_i.savefig('room_inertie.pdf', dpi=150)
plt.show()

# ── 7d. Profils hygro-thermiques détaillés dans chaque paroi ─────────────────
# Snapshots : 3 jours (avant/pendant/après le pic chaud) x 2 heures (14h/3h)
i_hot     = int(np.argmax(T_arr))
day_hot   = i_hot // steps_day
days_snap = sorted({max(0, day_hot - 7), day_hot, min(N_DAYS - 2, day_hot + 7)})

for wi, cfg in enumerate(sim.wall_configs):
    snapshots = []
    for d in days_snap:
        for h, ls in [(14, 'solid'), (3, 'dashed')]:
            snapshots.append((f'J{d+1:02d} {h:02d}h', d * steps_day + h, ls))
    safe_name = cfg.name.replace(' ', '_').replace('/', '_')
    plot_wall_hygrothermal_profiles(
        sim, snapshots=snapshots, wall_index=wi,
        save=f'profils_{safe_name}.pdf')

# ── 7e. Profils T + HR snapshot simple (jour le plus chaud) ──────────────────
plot_wall_profiles(sim, day=day_hot, save='wall_profiles.pdf')

# ── 7f. DH journalier + cumulé ───────────────────────────────────────────────
DH_daily = np.array([
    float(np.maximum(
        T_arr[d*steps_day : min((d+1)*steps_day, n_out)] - 28.0, 0.0
    ).sum()) * dt_h
    for d in range(N_DAYS)
])
DH_cumul = np.cumsum(DH_daily)
days_ax  = np.arange(1, N_DAYS + 1)

fig_d, (ax_d1, ax_d2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
ax_d1.bar(days_ax, DH_daily,
          color=['#d62728' if v > 0 else '#aec7e8' for v in DH_daily],
          alpha=0.85, width=0.9, label='DH quotidien [degC.h]')
ax_d1.set_ylabel('DH journalier [degC.h]', fontsize=12)
ax_d1.legend(fontsize=9)
ax_d1.set_title(
    f'Degres-heures d inconfort RE2020 — beton de chanvre 30 cm\n'
    f'DH total (juin-juil.) = {DH:.0f} degC.h  '
    f'({DH/1250*100:.1f}% du seuil annuel H1b = 1250 degC.h)', fontsize=11)

ax_d2.plot(days_ax, DH_cumul, lw=1.5, color='#d62728', label='DH cumules')
ax_d2.axhline(1250, ls='--', lw=1.2, color='black', label='Seuil RE2020 = 1250 degC.h')
ax_d2.fill_between(days_ax, DH_cumul, where=(DH_cumul >= 1250),
                   color='red', alpha=0.15, label='Depassement')
ax_d2.set_ylabel('DH cumules [degC.h]', fontsize=12)
ax_d2.set_xlabel('Jours depuis le 1er juin', fontsize=12)
ax_d2.legend(fontsize=9)
plt.tight_layout()
fig_d.savefig('DH_re2020.pdf', dpi=150)
plt.show()

# ── 7g. Carte de confort adaptatif (EN 15251) ────────────────────────────────
plot_comfort_map(sim, Text_out, t_days=t_days, save='comfort_map.pdf')

# ── 7h. Tableau bilan énergétique ────────────────────────────────────────────
table_energy_balance(sim, floor_area=FLOOR_AREA)

# ── 7i. Export CSV ───────────────────────────────────────────────────────────
export_csv(sim, filepath='results_room.csv', dt=DT)


###############################################################################
# ────────────────────────  8. Résumé console  ────────────────────────────────
###############################################################################

print("\n" + "=" * 60)
print("  FICHIERS SAUVEGARDES")
print("-" * 60)
fichiers = ['walls_3d.pdf', 'room_climate.pdf', 'room_inertie.pdf',
            'wall_profiles.pdf', 'DH_re2020.pdf', 'comfort_map.pdf',
            'results_room.csv']
for cfg in sim.wall_configs:
    fichiers.append(f'profils_{cfg.name.replace(" ","_").replace("/","_")}.pdf')
for f in fichiers:
    print(f"  {f}")
print("=" * 60)
