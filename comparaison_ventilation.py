# -*- coding: utf-8 -*-
"""
comparaison_ventilation.py
==========================
Compare l'effet de différents débits de surventilation nocturne
sur le confort d'été en béton de chanvre 30 cm.

Modifier les scénarios dans la section PARAMÈTRES.
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DIR = r"C:\Users\IMena\Desktop\modelhygro"
if DIR not in sys.path:
    sys.path.insert(0, DIR)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from wall_config     import make_wall, WindowConfig, ThermalBridge, OpeningConfig
from sources         import OccupantConfig, EquipmentConfig, LightingConfig, VentilationConfig, build_daily_schedule
from solar           import SolarCalculator
from room_simulation import RoomSimulation
from weather         import load_weather_csv


# ══════════════════════════════════════════════════════════════════════════════
# PARAMÈTRES — modifier ici
# ══════════════════════════════════════════════════════════════════════════════

# Scénarios à comparer : (étiquette, ACH_nuit, couleur)
# ACH_nuit = débit de surventilation nocturne [vol/h]
# La ventilation de jour reste fixe à 0.3 vol/h dans tous les scénarios.
SCENARIOS = [
    ("Sans ventilation nocturne  (0 vol/h)",  0.0,  "#888888"),
    ("RE2020  (4 vol/h)",                     4.0,  "#1F5F99"),
    ("Intensif  (8 vol/h)",                   8.0,  "#CC3030"),
]

# Plage horaire de la surventilation nocturne (identique pour tous)
HEURES_NUIT  = list(range(0, 8)) + list(range(22, 24))   # 22h → 8h
ACH_JOUR     = 0.3    # [vol/h]  ventilation jour (fixe)

# Géométrie / paroi
EPAISSEUR_M   = 0.30
SURFACE_SOL   = 80.0
VOLUME_AIR    = 200.0
MASSE_INTERNE = 110e3 * SURFACE_SOL
HM_EXT        = 25e-9
HM_INT        = HM_EXT * 8.0 / 25.0

# Période simulée
FICHIER_METEO = os.path.join(DIR, "donnees-climatiques-prospectives-france-2c_macon.csv")
DT            = 3600.0


# ══════════════════════════════════════════════════════════════════════════════
# MÉTÉO
# ══════════════════════════════════════════════════════════════════════════════

meteo   = load_weather_csv(FICHIER_METEO)
T_JUIN1 = 151 * 86400.0
T_SEP1  = 243 * 86400.0
mask    = (meteo.time_s >= T_JUIN1) & (meteo.time_s < T_SEP1)
time_bc = meteo.time_s[mask] - T_JUIN1
Text_bc = meteo.T_ext[mask]
RHext_bc= meteo.RH_ext[mask]
n_steps = len(time_bc) - 1
steps_j = int(86400 / DT)

T_ext_full  = np.interp(np.arange(n_steps + 1) * DT, time_bc, Text_bc)
RH_ext_full = np.interp(np.arange(n_steps + 1) * DT, time_bc, RHext_bc) * 100

print(f"Météo : {meteo.station}  ({len(time_bc)} pas horaires, juin–août)")


# ══════════════════════════════════════════════════════════════════════════════
# FONCTION DE SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

vitrages = [
    WindowConfig("Sud",  area=8.0, orientation="S", U_value=1.1, g_value=0.60, shading=0.30),
    WindowConfig("Nord", area=2.0, orientation="N", U_value=1.1, g_value=0.60, shading=0.0),
]
ponts        = [ThermalBridge("PT", psi=0.04, length=20.0)]
infiltrations= [OpeningConfig("Inf", ach_contribution=0.1)]


def creer_parois():
    kw = dict(layers=[("Hempcrete", EPAISSEUR_M)], mesh_size=0.02,
              h_ext=25.0, hm_ext=HM_EXT, h_int=8.0, hm_int=HM_INT)
    return [
        make_wall("Sud",   area=18.0, orientation="S", **kw),
        make_wall("Nord",  area=18.0, orientation="N", **kw),
        make_wall("Est",   area=12.0, orientation="E", **kw),
        make_wall("Ouest", area=12.0, orientation="W", **kw),
    ]


def simuler_scenario(ach_nuit):
    """Lance une simulation pour un débit nocturne donné."""
    if ach_nuit > 0:
        ach_total = ach_nuit + ACH_JOUR
        sched_v   = build_daily_schedule(
            hours_on=HEURES_NUIT, n_steps=n_steps, dt=int(DT),
            value_on=1.0, value_off=ACH_JOUR / ach_total)
        vent = VentilationConfig(n_ach=ach_total, hrv_efficiency=0.0,
                                 moisture_recovery=0.0, schedule=sched_v)
    else:
        vent = VentilationConfig(n_ach=ACH_JOUR + 0.1,   # infiltrations seules
                                 hrv_efficiency=0.0)

    sched_occ = build_daily_schedule(list(range(8, 22)), n_steps, int(DT))
    sched_lum = build_daily_schedule(list(range(20, 24)), n_steps, int(DT))

    sim = RoomSimulation(
        wall_configs    = creer_parois(),
        window_configs  = vitrages,
        bridge_configs  = ponts,
        opening_configs = infiltrations,
        occupants  = OccupantConfig(2.0, 80.0, 60.0, sched_occ),
        equipment  = EquipmentConfig(200.0, schedule=sched_occ),
        lighting   = LightingConfig(5.0, SURFACE_SOL, schedule=sched_lum),
        ventilation= vent,
        hvac       = None,
        solar_calc = SolarCalculator(meteo.lat, meteo.lon, start_doy=152,
                                     cloud_factor=0.30, ground_albedo=0.20,
                                     timezone_offset=1.0),
        volume        = VOLUME_AIR,
        internal_mass = MASSE_INTERNE,
        T_room_init   = float(Text_bc[0]),
        RH_room_init  = float(RHext_bc[0]),
    )

    T_arr, RH_arr = sim.run(time_bc=time_bc, Text_bc=Text_bc,
                             RHext_bc=RHext_bc, dt=DT, verbose=False)
    return np.array(T_arr), np.array(RH_arr) * 100


# ══════════════════════════════════════════════════════════════════════════════
# LANCEMENT
# ══════════════════════════════════════════════════════════════════════════════

resultats = []
for label, ach, couleur in SCENARIOS:
    print(f"  Simulation : {label} ...")
    T, RH = simuler_scenario(ach)
    resultats.append(dict(label=label, ach=ach, couleur=couleur,
                          T=T, RH=RH))
    DH = float(np.maximum(T - 28.0, 0.0).sum())
    pct_28 = 100.0 * (T > 28).sum() / len(T)
    print(f"    T max={T.max():.1f}°C  DH={DH:.0f}°C·h  "
          f"Heures>28°C={pct_28:.0f}%")


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUE 1 — Série temporelle complète (T et HR)
# ══════════════════════════════════════════════════════════════════════════════

print("\n  Tracé : comparaison_serie.pdf ...")

n_j   = (n_steps + 1) // steps_j
jours = np.arange(n_j)

fig1, (ax_T, ax_HR) = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
fig1.suptitle(
    "Comparaison des scénarios de ventilation nocturne\n"
    "Béton de chanvre 30 cm  |  Météo Mâcon +2°C  |  juin–août  |  Sans climatisation",
    fontsize=12, fontweight="bold"
)

# Extérieur en fond
T_moy_e  = np.array([T_ext_full[d*steps_j:(d+1)*steps_j].mean() for d in range(n_j)])
T_min_e  = np.array([T_ext_full[d*steps_j:(d+1)*steps_j].min()  for d in range(n_j)])
T_max_e  = np.array([T_ext_full[d*steps_j:(d+1)*steps_j].max()  for d in range(n_j)])
ax_T.fill_between(jours, T_min_e, T_max_e, color="#CCCCCC", alpha=0.45,
                  label="Plage extérieure (min–max)")
ax_T.plot(jours, T_moy_e, color="#888888", lw=1.5, ls="--",
          label="T ext. moyenne journalière")
ax_T.axhline(28, color="#CC3030", lw=1.2, ls=":", alpha=0.7,
             label="Seuil RE2020 = 28 °C")

HR_moy_e = np.array([RH_ext_full[d*steps_j:(d+1)*steps_j].mean() for d in range(n_j)])
ax_HR.plot(jours, HR_moy_e, color="#888888", lw=1.5, ls="--",
           label="HR ext. moyenne journalière")
ax_HR.axhspan(30, 70, color="#2E8B57", alpha=0.07, label="Confort 30–70 %")

# Scénarios
for res in resultats:
    T_moy = np.array([res["T"][d*steps_j:(d+1)*steps_j].mean() for d in range(n_j)])
    T_max = np.array([res["T"][d*steps_j:(d+1)*steps_j].max()  for d in range(n_j)])
    HR_moy= np.array([res["RH"][d*steps_j:(d+1)*steps_j].mean()for d in range(n_j)])

    ax_T.plot(jours, T_moy, color=res["couleur"], lw=2.2, label=res["label"])
    ax_T.plot(jours, T_max, color=res["couleur"], lw=1.0, ls="--", alpha=0.5)
    ax_HR.plot(jours, HR_moy, color=res["couleur"], lw=2.2, label=res["label"])

ax_T.set_ylabel("Température [°C]", fontsize=12)
ax_T.set_xlim(0, n_j - 1)
ax_T.set_xticks(range(0, n_j, 7))
ax_T.set_xticklabels([f"J+{d}" for d in range(0, n_j, 7)], fontsize=9)
ax_T.legend(fontsize=10, loc="upper left", framealpha=0.92)
ax_T.grid(True, alpha=0.18)
ax_T.set_title("Température moyenne journalière (trait plein) et maximum journalier (tirets)", fontsize=10)

ax_HR.set_ylabel("Humidité relative [%]", fontsize=12)
ax_HR.set_xlabel("Jours depuis le 1er juin", fontsize=11)
ax_HR.set_xlim(0, n_j - 1)
ax_HR.set_ylim(15, 100)
ax_HR.set_xticks(range(0, n_j, 7))
ax_HR.set_xticklabels([f"J+{d}" for d in range(0, n_j, 7)], fontsize=9)
ax_HR.legend(fontsize=10, loc="upper right", framealpha=0.92)
ax_HR.grid(True, alpha=0.18)
ax_HR.set_title("Humidité relative moyenne journalière", fontsize=10)

_out = os.path.join(DIR, "comparaison_serie.pdf")
try:    fig1.savefig(_out, dpi=150); print("  [OK] comparaison_serie.pdf")
except PermissionError:
    fig1.savefig(_out.replace(".pdf","_new.pdf"), dpi=150)
plt.close(fig1)


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUE 2 — Jour le plus chaud : profil horaire T et HR
# ══════════════════════════════════════════════════════════════════════════════

print("  Tracé : comparaison_jour_chaud.pdf ...")

# Jour le plus chaud sur le scénario RE2020 (référence commune)
res_ref  = resultats[1]   # RE2020
T_max_j  = np.array([res_ref["T"][d*steps_j:(d+1)*steps_j].max() for d in range(n_j)])
JOUR_C   = int(np.argmax(T_max_j))
i0, i1   = JOUR_C * steps_j, (JOUR_C + 1) * steps_j
h_ax     = np.arange(steps_j)

from datetime import date, timedelta
import locale
try:
    locale.setlocale(locale.LC_TIME, "French_France.1252")
except Exception:
    pass
date_c = (date(2000, 6, 1) + timedelta(days=JOUR_C)).strftime("%d %B").lstrip("0")  # année fictive

fig2, (ax_T2, ax_HR2) = plt.subplots(2, 1, figsize=(13, 9), constrained_layout=True)
fig2.suptitle(
    f"Profil horaire du jour le plus chaud : {date_c}\n"
    "Comparaison des scénarios de ventilation nocturne",
    fontsize=12, fontweight="bold"
)

ax_T2.plot(h_ax, T_ext_full[i0:i1], color="#888888", lw=2.0, ls="--",
           label=f"T extérieure  (max {T_ext_full[i0:i1].max():.1f} °C)")
ax_T2.axhline(28, color="#CC3030", lw=1.0, ls=":", alpha=0.7,
              label="Seuil RE2020 = 28 °C")
ax_HR2.plot(h_ax, RH_ext_full[i0:i1], color="#888888", lw=2.0, ls="--",
            label="HR extérieure")
ax_HR2.axhspan(30, 70, color="#2E8B57", alpha=0.07)

for res in resultats:
    T_j  = res["T"][i0:i1]
    HR_j = res["RH"][i0:i1]
    ax_T2.plot(h_ax, T_j,  color=res["couleur"], lw=2.5,
               label=f"{res['label']}  (max {T_j.max():.1f} °C)")
    ax_HR2.plot(h_ax, HR_j, color=res["couleur"], lw=2.5,
                label=f"{res['label']}  (moy {HR_j.mean():.0f} %)")

# Annotation plage de ventilation nocturne
for ax in [ax_T2, ax_HR2]:
    ax.axvspan(0, 8,  color="#1F5F99", alpha=0.06, label="Ventilation nocturne active")
    ax.axvspan(22, 23.9, color="#1F5F99", alpha=0.06)

ax_T2.set_ylabel("Température [°C]", fontsize=12)
ax_T2.set_xlim(0, 23); ax_T2.set_xticks(range(0, 24, 2))
ax_T2.tick_params(labelbottom=False)
ax_T2.legend(fontsize=10, loc="upper left", framealpha=0.92)
ax_T2.grid(True, alpha=0.18)

ax_HR2.set_ylabel("Humidité relative [%]", fontsize=12)
ax_HR2.set_xlabel(f"Heure de la journée [h]  —  {date_c}", fontsize=11)
ax_HR2.set_xlim(0, 23); ax_HR2.set_xticks(range(0, 24, 2))
ax_HR2.set_ylim(10, 100)
ax_HR2.legend(fontsize=10, loc="upper right", framealpha=0.92)
ax_HR2.grid(True, alpha=0.18)

_out = os.path.join(DIR, "comparaison_jour_chaud.pdf")
try:    fig2.savefig(_out, dpi=150); print("  [OK] comparaison_jour_chaud.pdf")
except PermissionError:
    fig2.savefig(_out.replace(".pdf","_new.pdf"), dpi=150)
plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUE 3 — Indicateurs synthétiques
# ══════════════════════════════════════════════════════════════════════════════

print("  Tracé : comparaison_indicateurs.pdf ...")

labels  = [r["label"] for r in resultats]
couleurs= [r["couleur"] for r in resultats]
DH_vals = [float(np.maximum(r["T"] - 28.0, 0.0).sum()) for r in resultats]
Tmax_v  = [r["T"].max() for r in resultats]
Tmoy_v  = [r["T"].mean() for r in resultats]
pct_28  = [100.0 * (r["T"] > 28).sum() / len(r["T"]) for r in resultats]
HR_moy_v= [r["RH"].mean() for r in resultats]

x = np.arange(len(resultats))
w = 0.55

fig3, axes3 = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
fig3.suptitle(
    "Indicateurs de confort par scénario — Béton de chanvre 30 cm  |  juin–août",
    fontsize=12, fontweight="bold"
)

def bar_plot(ax, vals, titre, unite, ref_line=None, ref_label=None, fmt=".0f"):
    bars = ax.bar(x, vals, width=w, color=couleurs, alpha=0.85, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                f"{v:{fmt}}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    if ref_line is not None:
        ax.axhline(ref_line, color="#333333", lw=1.5, ls="--",
                   label=ref_label, alpha=0.8)
        ax.legend(fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([r["label"].split("(")[0].strip() for r in resultats],
                       fontsize=9.5, ha="center")
    ax.set_title(titre, fontsize=11, fontweight="bold")
    ax.set_ylabel(unite, fontsize=11)
    ax.set_ylim(0, max(vals) * 1.20)
    ax.grid(True, alpha=0.15, axis="y")

bar_plot(axes3[0,0], DH_vals,  "Degrés-heures d'inconfort (DH)",
         "°C·h", ref_line=1250, ref_label="Seuil RE2020 H1b = 1 250 °C·h")
bar_plot(axes3[0,1], Tmax_v,   "Température maximale atteinte",
         "°C",   ref_line=28,   ref_label="Seuil RE2020 = 28 °C", fmt=".1f")
bar_plot(axes3[1,0], pct_28,   "Heures avec T > 28 °C",
         "% des heures simulées")
bar_plot(axes3[1,1], HR_moy_v, "Humidité relative moyenne",
         "%",    ref_line=30,   ref_label="Min confort = 30 %",
         fmt=".0f")

_out = os.path.join(DIR, "comparaison_indicateurs.pdf")
try:    fig3.savefig(_out, dpi=150); print("  [OK] comparaison_indicateurs.pdf")
except PermissionError:
    fig3.savefig(_out.replace(".pdf","_new.pdf"), dpi=150)
plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*65)
print(f"  {'Scénario':<38} {'DH':>6} {'Tmax':>6} {'%>28':>6} {'HR moy':>7}")
print("-"*65)
for res, DH, Tm, p28, HR in zip(resultats, DH_vals, Tmax_v, pct_28, HR_moy_v):
    print(f"  {res['label']:<38} {DH:>6.0f} {Tm:>6.1f} {p28:>5.0f}% {HR:>6.0f}%")
print("="*65)
print("  DH = degrés-heures d'inconfort [°C·h]  |  seuil RE2020 = 1 250")
print("  Fichiers : comparaison_serie.pdf  comparaison_jour_chaud.pdf  comparaison_indicateurs.pdf")
