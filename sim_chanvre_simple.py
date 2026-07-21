# -*- coding: utf-8 -*-
"""
sim_chanvre_simple.py
=====================
4 graphiques simples — piece en beton de chanvre 30 cm
  G1 : Temperatures ete  (T_ext / T_surf_ext / T_surf_int / T_piece)  + dephasage
  G2 : Temperatures hiver (idem)
  G3 : Flux de chaleur a la surface interieure (ete et hiver)
  G4 : Humidite relative (piece + surf. int.) ete et hiver

Ventilation RE2020 :
  Ete   — surventilation nocturne 4.0 vol/h (22h-8h) + 0.3 vol/h diurne
  Hiver — VMC double flux 0.3 vol/h, HRV=0.75
"""

import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DIR = r"C:\Users\IMena\Desktop\modelhygro"
if DIR not in sys.path:
    sys.path.insert(0, DIR)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from wall_config      import make_wall, WindowConfig, ThermalBridge, OpeningConfig
from sources          import (OccupantConfig, EquipmentConfig, LightingConfig,
                               VentilationConfig, build_daily_schedule)
from solar            import SolarCalculator
from re2020           import RE2020Evaluator
from room_simulation  import RoomSimulation
from weather          import load_weather_csv

# ── Parametres communs ─────────────────────────────────────────────────────────
DT           = 3600.0
FLOOR_AREA   = 80.0
ROOM_VOLUME  = 200.0
INTERNAL_MASS = 110e3 * FLOOR_AREA

HM_EXT = 25e-9
HM_INT = HM_EXT * (8.0 / 25.0)

WEATHER_FILE = os.path.join(DIR, "donnees-climatiques-prospectives-france-2c_macon.csv")
weather = load_weather_csv(WEATHER_FILE)

# ── Geometrie (identique ete et hiver) ────────────────────────────────────────
def make_walls():
    return [
        make_wall("Sud",   area=18.0, orientation="S",
                  layers=[("Hempcrete", 0.30)], mesh_size=0.01,
                  h_ext=25.0, hm_ext=HM_EXT, h_int=8.0, hm_int=HM_INT),
        make_wall("Nord",  area=18.0, orientation="N",
                  layers=[("Hempcrete", 0.30)], mesh_size=0.01,
                  h_ext=25.0, hm_ext=HM_EXT, h_int=8.0, hm_int=HM_INT),
        make_wall("Est",   area=12.0, orientation="E",
                  layers=[("Hempcrete", 0.30)], mesh_size=0.01,
                  h_ext=25.0, hm_ext=HM_EXT, h_int=8.0, hm_int=HM_INT),
        make_wall("Ouest", area=12.0, orientation="W",
                  layers=[("Hempcrete", 0.30)], mesh_size=0.01,
                  h_ext=25.0, hm_ext=HM_EXT, h_int=8.0, hm_int=HM_INT),
    ]

windows = [
    WindowConfig(name="Vitrage Sud",  area=8.0, orientation="S",
                 U_value=1.1, g_value=0.60, shading=0.30),
    WindowConfig(name="Vitrage Nord", area=2.0, orientation="N",
                 U_value=1.1, g_value=0.60, shading=0.0),
]
bridges  = [ThermalBridge(name="Encadrements", psi=0.04, length=20.0)]
openings = [OpeningConfig(name="Infiltrations", ach_contribution=0.1)]


def run_season(label, t0_s, t1_s, ach_night, hrv, is_summer):
    """Run one RoomSimulation for a seasonal period."""
    mask     = (weather.time_s >= t0_s) & (weather.time_s < t1_s)
    time_bc  = weather.time_s[mask] - t0_s
    Text_bc  = weather.T_ext[mask]
    RHext_bc = weather.RH_ext[mask]
    n_steps  = len(time_bc) - 1

    occ_sched = build_daily_schedule(hours_on=list(range(8, 22)),
                                     n_steps=n_steps, dt=DT)
    lum_sched = build_daily_schedule(hours_on=list(range(20, 24)),
                                     n_steps=n_steps, dt=DT)

    if is_summer:
        # Surventilation nocturne RE2020 : 4 vol/h la nuit + 0.3 vol/h le jour
        ACH_BASE  = 0.3
        ACH_NUIT  = 4.3   # 0.3 + 4.0
        night_h   = list(range(0, 8)) + list(range(22, 24))
        sched     = build_daily_schedule(hours_on=night_h, n_steps=n_steps, dt=DT,
                                         value_on=1.0, value_off=ACH_BASE / ACH_NUIT)
        vent = VentilationConfig(n_ach=ACH_NUIT, hrv_efficiency=0.0,
                                  moisture_recovery=0.0, schedule=sched)
        start_doy = 152   # 1er juin
    else:
        # VMC double flux hiver RE2020 : 0.3 vol/h, HRV=0.75
        vent = VentilationConfig(n_ach=0.3, hrv_efficiency=hrv,
                                  moisture_recovery=0.0)
        start_doy = 1     # 1er janvier

    solar = SolarCalculator(
        latitude=weather.lat, longitude=weather.lon,
        start_doy=start_doy, cloud_factor=0.30,
        ground_albedo=0.20, timezone_offset=1.0,
    )
    re2020_ev = RE2020Evaluator(floor_area=FLOOR_AREA, climate_zone="H1b")
    re2020_ev.set_carrier(heating="electricity", cooling="electricity")

    sim = RoomSimulation(
        wall_configs    = make_walls(),
        window_configs  = windows,
        bridge_configs  = bridges,
        opening_configs = openings,
        occupants       = OccupantConfig(n_occupants=2.0,
                                         sensible_per_person=80.0,
                                         latent_per_person=60.0,
                                         schedule=occ_sched),
        equipment       = EquipmentConfig(power=200.0, schedule=occ_sched),
        lighting        = LightingConfig(power_density=5.0, floor_area=FLOOR_AREA,
                                          schedule=lum_sched),
        ventilation     = vent,
        hvac            = None,
        solar_calc      = solar,
        volume          = ROOM_VOLUME,
        internal_mass   = INTERNAL_MASS,
        T_room_init     = float(Text_bc[0]),
        RH_room_init    = float(RHext_bc[0]),
        re2020          = re2020_ev,
    )

    print(f"\n[{label}] Lancement simulation ({n_steps} pas)...")
    T_room_arr, RH_room_arr = sim.run(
        time_bc=time_bc, Text_bc=Text_bc, RHext_bc=RHext_bc, dt=DT, verbose=True)

    # Extraire T_surf_ext (noeud 0) et T_surf_int (noeud -1) de la paroi Sud
    wall_sud  = sim.walls[0]
    cfg_sud   = sim.wall_configs[0]
    n_out     = len(T_room_arr)

    T_surf_ext = np.array([wall_sud.StockT[i][0,  0] - 273.15 for i in range(n_out)])
    T_surf_int = np.array([wall_sud.StockT[i][-1, 0] - 273.15 for i in range(n_out)])
    RH_surf    = np.array([float(wall_sud.StockRH[i][-1, 0])  for i in range(n_out)])

    # Flux de chaleur (conduction a la surface interieure paroi Sud) [W/m^2]
    # Approximation via gradient des deux derniers noeuds
    N_lay = wall_sud.layer.N_tot
    dx    = wall_sud.layer.dx.flatten()
    q_int = []
    for i in range(n_out):
        T_f = wall_sud.StockT[i].flatten()
        k   = float(wall_sud.layer.k(wall_sud.StockT[i], wall_sud.StockRH[i]).flatten()[-1])
        q   = -k * (T_f[-1] - T_f[-2]) / dx[-1]
        q_int.append(q)
    q_int = np.array(q_int)

    T_ext_full = np.interp(np.arange(n_out) * DT, time_bc, Text_bc)

    return dict(
        T_arr      = np.array(T_room_arr),
        RH_arr     = np.array(RH_room_arr),
        T_ext      = T_ext_full,
        T_surf_ext = T_surf_ext,
        T_surf_int = T_surf_int,
        RH_surf    = RH_surf,
        q_int      = q_int,
        n_out      = n_out,
    )


def best_week_start(T, n_days, steps_day, hottest=True):
    """Return the step index of the hottest/coldest 7-day window."""
    best_s, best_m = 0, (-999 if hottest else 999)
    for i in range(max(1, n_days - 7)):
        m = T[i*steps_day : i*steps_day + 7*steps_day].mean()
        if (hottest and m > best_m) or (not hottest and m < best_m):
            best_m = m; best_s = i * steps_day
    return best_s


def dephasage_from_corr(sig_in, sig_out, dt_h):
    """Estimate phase lag [h] via cross-correlation (restricted to +-12 h)."""
    n    = len(sig_in)
    half = min(int(12 / dt_h), n // 4)
    ref  = sig_in  - sig_in.mean()
    sig  = sig_out - sig_out.mean()
    corr = np.correlate(sig, ref, mode="full")
    lags = np.arange(-(n - 1), n)
    mid  = n - 1
    best = int(np.argmax(corr[mid - half : mid + half + 1]))
    lag  = lags[mid - half + best]
    return float(lag) * dt_h


# ── Extraction periodes ────────────────────────────────────────────────────────
# Hiver  : 1er janvier -> 31 janvier (31 jours)
T_JAN1 = 0.0
T_FEB1 = 31 * 86400.0

# Ete    : 1er juin -> 31 juillet (61 jours)
T_JUN1 = 151 * 86400.0
T_AUG1 = 212 * 86400.0

print("=" * 60)
print("Simulation beton de chanvre 30 cm — ETE + HIVER")
print("=" * 60)

ete    = run_season("ETE",   T_JUN1, T_AUG1, ach_night=4.3, hrv=0.0,  is_summer=True)
hiver  = run_season("HIVER", T_JAN1, T_FEB1, ach_night=0.3, hrv=0.75, is_summer=False)

print("\n[OK] Simulations terminees\n")

# ── Semaines representantes ────────────────────────────────────────────────────
steps_day = int(86400 / DT)
n_ete   = ete["n_out"]
n_hiver = hiver["n_out"]
n_ete_d = n_ete // steps_day
n_hiv_d = n_hiver // steps_day

s_ete   = best_week_start(ete["T_arr"],   n_ete_d,   steps_day, hottest=True)
s_hiv   = best_week_start(hiver["T_arr"], n_hiv_d,   steps_day, hottest=False)

e_ete   = min(s_ete + 7*steps_day, n_ete)
e_hiv   = min(s_hiv + 7*steps_day, n_hiver)

t_ete_h  = np.arange(e_ete  - s_ete)  * (DT/3600)   # heures depuis debut semaine
t_hiv_h  = np.arange(e_hiv  - s_hiv)  * (DT/3600)

dep_ete  = dephasage_from_corr(ete["T_surf_ext"][s_ete:e_ete],
                                ete["T_surf_int"][s_ete:e_ete], DT/3600)
dep_hiv  = dephasage_from_corr(hiver["T_surf_ext"][s_hiv:e_hiv],
                                hiver["T_surf_int"][s_hiv:e_hiv], DT/3600)

atten_ete = ((ete["T_surf_int"][s_ete:e_ete].max() - ete["T_surf_int"][s_ete:e_ete].min()) /
             max(ete["T_surf_ext"][s_ete:e_ete].max() - ete["T_surf_ext"][s_ete:e_ete].min(), 0.1))
atten_hiv = ((hiver["T_surf_int"][s_hiv:e_hiv].max() - hiver["T_surf_int"][s_hiv:e_hiv].min()) /
             max(hiver["T_surf_ext"][s_hiv:e_hiv].max() - hiver["T_surf_ext"][s_hiv:e_hiv].min(), 0.1))

print(f"Dephasage ete  (surf_ext->surf_int) = {dep_ete:.1f} h,  amortissement = {(1-atten_ete)*100:.0f}%")
print(f"Dephasage hiver (surf_ext->surf_int)= {dep_hiv:.1f} h,  amortissement = {(1-atten_hiv)*100:.0f}%")

# ── Figures ────────────────────────────────────────────────────────────────────
C_EXT = "#E07020"   # orange — T_ext
C_SE  = "#9DBF40"   # vert clair — T_surf_ext
C_SI  = "#2E6B9E"   # bleu — T_surf_int
C_RM  = "#CC3030"   # rouge — T_piece
C_HUM = "#7B4EA0"   # violet — humidite

fig, axes = plt.subplots(4, 1, figsize=(14, 20))
fig.suptitle("Beton de chanvre 30 cm — Piece libre (sans climatisation)\n"
             "Ventilation RE2020 : ete 4 vol/h nuit | hiver VMC 0.3 vol/h HRV75%",
             fontsize=13, fontweight="bold", y=0.995)


def annotate_dephasage(ax, t, sig_in, sig_out, dep, color_in, color_out):
    """Mark peak of sig_in and sig_out, draw a horizontal arrow for dephasing."""
    # Find peaks in last 4 days (to avoid edge effects)
    n_show = len(t)
    search = slice(n_show//2, n_show)  # second half of week
    i_peak_in  = search.start + int(np.argmax(sig_in[search]))
    i_peak_out = search.start + int(np.argmax(sig_out[search]))
    t_pin  = t[i_peak_in]
    t_pout = t[i_peak_out]
    y_in   = sig_in[i_peak_in]
    y_out  = sig_out[i_peak_out]
    y_mid  = (y_in + y_out) / 2
    ax.annotate("", xy=(t_pout, y_mid), xytext=(t_pin, y_mid),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.8))
    ax.text((t_pin + t_pout) / 2, y_mid + 0.5,
            f" {abs(dep):.0f} h", ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
    ax.axvline(t_pin,  color=color_in,  lw=1.0, ls=":", alpha=0.6)
    ax.axvline(t_pout, color=color_out, lw=1.0, ls=":", alpha=0.6)


# ── G1 : Temperatures ete ─────────────────────────────────────────────────────
ax = axes[0]
se = ete["T_surf_ext"][s_ete:e_ete]
si = ete["T_surf_int"][s_ete:e_ete]
te = ete["T_ext"][s_ete:e_ete]
tr = ete["T_arr"][s_ete:e_ete]

ax.plot(t_ete_h, te, color=C_EXT, lw=1.5, ls="--", label="T ext. (air dehors)", alpha=0.85)
ax.plot(t_ete_h, se, color=C_SE,  lw=1.5, ls=":",  label="T surf. ext. (noeud 0)")
ax.plot(t_ete_h, si, color=C_SI,  lw=2.5,           label="T surf. int. (noeud N)")
ax.plot(t_ete_h, tr, color=C_RM,  lw=2.0,           label="T piece")
ax.axhline(28, color="red", lw=1.0, ls="--", alpha=0.6, label="28 degC seuil DH")
ax.fill_between(t_ete_h, tr, 28, where=(tr > 28), color="red", alpha=0.12)
annotate_dephasage(ax, t_ete_h, se, si, dep_ete, C_SE, C_SI)
ax.set_ylabel("Temperature [degC]", fontsize=11)
ax.set_title(
    f"G1 — ETE (semaine la plus chaude)   |   "
    f"Dephasage surf_ext -> surf_int = {dep_ete:.0f} h   |   "
    f"Amortissement = {(1-atten_ete)*100:.0f}%",
    fontsize=10, loc="left")
ax.legend(fontsize=9, ncol=4, loc="upper left")
ax.grid(True, alpha=0.25)
ax.set_xlim(0, len(t_ete_h) - 1)
for d in range(0, 8*24, 24):
    ax.axvline(d, color="lightgray", lw=0.7, ls=":")

# ── G2 : Temperatures hiver ───────────────────────────────────────────────────
ax = axes[1]
se_h = hiver["T_surf_ext"][s_hiv:e_hiv]
si_h = hiver["T_surf_int"][s_hiv:e_hiv]
te_h = hiver["T_ext"][s_hiv:e_hiv]
tr_h = hiver["T_arr"][s_hiv:e_hiv]

ax.plot(t_hiv_h, te_h, color=C_EXT, lw=1.5, ls="--", label="T ext. (air dehors)", alpha=0.85)
ax.plot(t_hiv_h, se_h, color=C_SE,  lw=1.5, ls=":",  label="T surf. ext. (noeud 0)")
ax.plot(t_hiv_h, si_h, color=C_SI,  lw=2.5,           label="T surf. int. (noeud N)")
ax.plot(t_hiv_h, tr_h, color=C_RM,  lw=2.0,           label="T piece")
ax.axhline(19, color="steelblue", lw=1.0, ls="--", alpha=0.6, label="19 degC confort min")
annotate_dephasage(ax, t_hiv_h, se_h, si_h, dep_hiv, C_SE, C_SI)
ax.set_ylabel("Temperature [degC]", fontsize=11)
ax.set_title(
    f"G2 — HIVER (semaine la plus froide)   |   "
    f"Dephasage surf_ext -> surf_int = {dep_hiv:.0f} h   |   "
    f"Amortissement = {(1-atten_hiv)*100:.0f}%",
    fontsize=10, loc="left")
ax.legend(fontsize=9, ncol=4, loc="upper left")
ax.grid(True, alpha=0.25)
ax.set_xlim(0, len(t_hiv_h) - 1)
for d in range(0, 8*24, 24):
    ax.axvline(d, color="lightgray", lw=0.7, ls=":")

# ── G3 : Flux de chaleur surface interieure (paroi Sud) ───────────────────────
ax = axes[2]
q_ete  = ete["q_int"][s_ete:e_ete]
q_hiv  = hiver["q_int"][s_hiv:e_hiv]
# normalise: ete en rouge, hiver en bleu, sur axes separes (twin)
ax2 = ax.twinx()
ax.plot(t_ete_h,  q_ete, color=C_RM,      lw=2.0, label="Flux ete [W/m2]")
ax2.plot(t_hiv_h, q_hiv, color="steelblue", lw=2.0, ls="--", label="Flux hiver [W/m2]")
ax.axhline(0, color="gray", lw=0.8)
ax.fill_between(t_ete_h, q_ete, 0, where=(q_ete > 0), color=C_RM,      alpha=0.15,
                label="Chaleur entrant dans la piece (ete)")
ax.fill_between(t_ete_h, q_ete, 0, where=(q_ete < 0), color="steelblue", alpha=0.15,
                label="Chaleur sortant vers ext. (ete)")
ax.set_ylabel("Flux chaleur ete [W/m2]", fontsize=11, color=C_RM)
ax2.set_ylabel("Flux chaleur hiver [W/m2]", fontsize=11, color="steelblue")
ax.tick_params(axis="y", labelcolor=C_RM)
ax2.tick_params(axis="y", labelcolor="steelblue")
ax.set_title("G3 — Flux de chaleur a la surface interieure — paroi Sud (+ = vers piece)",
             fontsize=10, loc="left")
lines1, labs1 = ax.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc="upper right")
ax.grid(True, alpha=0.25)
ax.set_xlim(0, max(len(t_ete_h), len(t_hiv_h)) - 1)
for d in range(0, 8*24, 24):
    ax.axvline(d, color="lightgray", lw=0.7, ls=":")

# ── G4 : Humidite relative piece + surf. int. ─────────────────────────────────
ax = axes[3]
rh_piece_ete  = ete["RH_arr"][s_ete:e_ete]  * 100
rh_surf_ete   = ete["RH_surf"][s_ete:e_ete] * 100
rh_piece_hiv  = hiver["RH_arr"][s_hiv:e_hiv]  * 100
rh_surf_hiv   = hiver["RH_surf"][s_hiv:e_hiv] * 100

ax.plot(t_ete_h, rh_piece_ete, color=C_HUM,   lw=2.0, label="HR piece — ete")
ax.plot(t_ete_h, rh_surf_ete,  color=C_HUM,   lw=1.5, ls=":", label="HR surf. int. — ete")
ax.plot(t_hiv_h, rh_piece_hiv, color="navy",  lw=2.0, ls="--", label="HR piece — hiver")
ax.plot(t_hiv_h, rh_surf_hiv,  color="navy",  lw=1.5, ls="-.", label="HR surf. int. — hiver")
ax.axhline(30, color="gray", lw=0.8, ls="--", label="30% min confort")
ax.axhline(70, color="red",  lw=0.8, ls="--", label="70% max confort")
ax.fill_between(t_ete_h, 30, 70, color="green", alpha=0.05, label="Zone confort 30-70%")
ax.set_ylim(10, 100)
ax.set_ylabel("Humidite relative [%]", fontsize=11)
ax.set_xlabel("Heures depuis le debut de la semaine representative", fontsize=11)
ax.set_title("G4 — Humidite relative — piece et surface interieure de la paroi Sud",
             fontsize=10, loc="left")
ax.legend(fontsize=9, ncol=3, loc="upper right")
ax.grid(True, alpha=0.25)
ax.set_xlim(0, max(len(t_ete_h), len(t_hiv_h)) - 1)
for d in range(0, 8*24, 24):
    ax.axvline(d, color="lightgray", lw=0.7, ls=":")

# Etiquette jours sur G4
for d in range(8):
    axes[3].text(d*24 + 12, 11, f"J{d+1}", ha="center", fontsize=8, color="gray")

plt.tight_layout(rect=[0, 0, 1, 0.995])
out = os.path.join(DIR, "chanvre_4graphiques.pdf")
fig.savefig(out, dpi=150)
plt.close()

print(f"\n[OK] chanvre_4graphiques.pdf sauvegarde dans {DIR}")
