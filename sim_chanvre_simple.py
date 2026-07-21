# -*- coding: utf-8 -*-
"""
sim_chanvre_simple.py
=====================
4 graphiques simples + vue 3D — piece beton de chanvre 30 cm
Ventilation RE2020 : ete 4 vol/h nuit | hiver VMC 0.3 vol/h HRV 75%
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

from wall_config     import make_wall, WindowConfig, ThermalBridge, OpeningConfig
from sources         import (OccupantConfig, EquipmentConfig, LightingConfig,
                              VentilationConfig, build_daily_schedule)
from solar           import SolarCalculator
from re2020          import RE2020Evaluator
from room_simulation import RoomSimulation
from weather         import load_weather_csv
from outputs         import plot_walls_3d

# ── Parametres ─────────────────────────────────────────────────────────────────
DT            = 3600.0
FLOOR_AREA    = 80.0
ROOM_VOLUME   = 200.0
INTERNAL_MASS = 110e3 * FLOOR_AREA
HM_EXT = 25e-9
HM_INT = HM_EXT * (8.0 / 25.0)

weather = load_weather_csv(
    os.path.join(DIR, "donnees-climatiques-prospectives-france-2c_macon.csv"))

windows  = [
    WindowConfig("Vitrage Sud",  area=8.0, orientation="S",
                 U_value=1.1, g_value=0.60, shading=0.30),
    WindowConfig("Vitrage Nord", area=2.0, orientation="N",
                 U_value=1.1, g_value=0.60, shading=0.0),
]
bridges  = [ThermalBridge("Encadrements", psi=0.04, length=20.0)]
openings = [OpeningConfig("Infiltrations", ach_contribution=0.1)]


def make_walls():
    kw = dict(layers=[("Hempcrete", 0.30)], mesh_size=0.01,
              h_ext=25.0, hm_ext=HM_EXT, h_int=8.0, hm_int=HM_INT)
    return [
        make_wall("Sud",   area=18.0, orientation="S", **kw),
        make_wall("Nord",  area=18.0, orientation="N", **kw),
        make_wall("Est",   area=12.0, orientation="E", **kw),
        make_wall("Ouest", area=12.0, orientation="W", **kw),
    ]


def run_season(label, t0_s, t1_s, is_summer, start_doy):
    mask     = (weather.time_s >= t0_s) & (weather.time_s < t1_s)
    time_bc  = weather.time_s[mask] - t0_s
    Text_bc  = weather.T_ext[mask]
    RHext_bc = weather.RH_ext[mask]
    n_steps  = len(time_bc) - 1

    sched_occ = build_daily_schedule(hours_on=list(range(8, 22)),
                                     n_steps=n_steps, dt=DT)
    sched_lum = build_daily_schedule(hours_on=list(range(20, 24)),
                                     n_steps=n_steps, dt=DT)

    if is_summer:
        night_h = list(range(0, 8)) + list(range(22, 24))
        sched_v = build_daily_schedule(hours_on=night_h, n_steps=n_steps,
                                       dt=DT, value_on=1.0, value_off=0.3/4.3)
        vent = VentilationConfig(n_ach=4.3, hrv_efficiency=0.0,
                                 moisture_recovery=0.0, schedule=sched_v)
    else:
        vent = VentilationConfig(n_ach=0.3, hrv_efficiency=0.75,
                                 moisture_recovery=0.0)

    solar = SolarCalculator(latitude=weather.lat, longitude=weather.lon,
                            start_doy=start_doy, cloud_factor=0.30,
                            ground_albedo=0.20, timezone_offset=1.0)
    re_ev = RE2020Evaluator(floor_area=FLOOR_AREA, climate_zone="H1b")
    re_ev.set_carrier(heating="electricity", cooling="electricity")

    sim = RoomSimulation(
        wall_configs=make_walls(), window_configs=windows,
        bridge_configs=bridges, opening_configs=openings,
        occupants  = OccupantConfig(n_occupants=2.0, sensible_per_person=80.0,
                                    latent_per_person=60.0, schedule=sched_occ),
        equipment  = EquipmentConfig(power=200.0, schedule=sched_occ),
        lighting   = LightingConfig(power_density=5.0, floor_area=FLOOR_AREA,
                                    schedule=sched_lum),
        ventilation=vent, hvac=None, solar_calc=solar,
        volume=ROOM_VOLUME, internal_mass=INTERNAL_MASS,
        T_room_init=float(Text_bc[0]), RH_room_init=float(RHext_bc[0]),
        re2020=re_ev,
    )

    print(f"  [{label}] simulation...")
    T_arr, RH_arr = sim.run(time_bc=time_bc, Text_bc=Text_bc,
                             RHext_bc=RHext_bc, dt=DT, verbose=True)
    T_arr  = np.array(T_arr)
    RH_arr = np.array(RH_arr)
    n_out  = len(T_arr)

    wall_sud = sim.walls[0]
    T_surf_int = np.array([wall_sud.StockT[i][-1, 0] - 273.15 for i in range(n_out)])
    RH_surf    = np.array([float(wall_sud.StockRH[i][-1, 0])   for i in range(n_out)])

    dx_last = float(wall_sud.layer.dx.flatten()[-1])
    q_int = np.array([
        -float(wall_sud.layer.k(wall_sud.StockT[i], wall_sud.StockRH[i]).flatten()[-1])
        * (wall_sud.StockT[i].flatten()[-1] - wall_sud.StockT[i].flatten()[-2]) / dx_last
        for i in range(n_out)
    ])

    T_ext_full = np.interp(np.arange(n_out) * DT, time_bc, Text_bc)

    return dict(sim=sim, T_arr=T_arr, RH_arr=RH_arr, T_ext=T_ext_full,
                T_surf_int=T_surf_int, RH_surf=RH_surf, q_int=q_int, n_out=n_out)


def hottest_week(T, steps_day):
    n = len(T) // steps_day
    best_s, best_m = 0, -999.0
    for i in range(max(1, n - 7)):
        m = T[i*steps_day:(i+1)*steps_day].mean()
        if m > best_m:
            best_m = m; best_s = i * steps_day
    return best_s, min(best_s + 7*steps_day, len(T))

def coldest_week(T, steps_day):
    n = len(T) // steps_day
    best_s, best_m = 0, 999.0
    for i in range(max(1, n - 7)):
        m = T[i*steps_day:(i+1)*steps_day].mean()
        if m < best_m:
            best_m = m; best_s = i * steps_day
    return best_s, min(best_s + 7*steps_day, len(T))


def draw_dephasage_arrow(ax, t_h, T_ext, T_room, steps_day, pick_day=3):
    """
    Marque le pic de T_ext et le pic de T_room sur le jour pick_day
    et trace une fleche bidirectionnelle entre les deux pics.
    """
    s = pick_day * steps_day
    e = min(s + steps_day, len(T_ext))
    if e <= s:
        return 0.0

    seg_e = T_ext[s:e]
    seg_r = T_room[s:e]
    i_e   = int(np.argmax(seg_e))
    i_r   = int(np.argmax(seg_r))
    t_e   = t_h[s + i_e]
    t_r   = t_h[s + i_r]
    v_e   = seg_e[i_e]
    v_r   = seg_r[i_r]

    dep_h = t_r - t_e
    if dep_h <= 0:
        dep_h += 24.0

    # Lignes verticales sur les deux pics
    ax.axvline(t_e, color="#E07020", lw=1.4, ls="--", alpha=0.7)
    ax.axvline(t_r, color="#CC3030", lw=1.4, ls="--", alpha=0.7)

    # Fleche entre les pics
    y_top = max(v_e, v_r) + 1.2
    ax.annotate("", xy=(t_r, y_top), xytext=(t_e, y_top),
                arrowprops=dict(arrowstyle="<->", color="black",
                                lw=2.0, mutation_scale=16))
    ax.text((t_e + t_r) / 2.0, y_top + 0.6,
            f"Dephasage\n{dep_h:.0f} h",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", alpha=0.9))
    return dep_h


# ── Simulations ────────────────────────────────────────────────────────────────
print("Beton de chanvre 30 cm — ETE + HIVER")
ete   = run_season("ETE",   151*86400.0, 212*86400.0, True,  152)
hiver = run_season("HIVER", 0.0,          31*86400.0, False,   1)
print("  Simulations terminees.")

steps_day = int(86400 / DT)

se, ee = hottest_week(ete["T_arr"],   steps_day)
sh, eh = coldest_week(hiver["T_arr"], steps_day)

t_e = np.arange(ee - se) * (DT / 3600.0)
t_h = np.arange(eh - sh) * (DT / 3600.0)

Te  = ete["T_ext"][se:ee]
Tr  = ete["T_arr"][se:ee]
Ti  = ete["T_surf_int"][se:ee]
RHe = ete["RH_arr"][se:ee] * 100
q_e = ete["q_int"][se:ee]

Te_h  = hiver["T_ext"][sh:eh]
Tr_h  = hiver["T_arr"][sh:eh]
Ti_h  = hiver["T_surf_int"][sh:eh]
RHh   = hiver["RH_arr"][sh:eh] * 100
q_h   = hiver["q_int"][sh:eh]

DT_h = DT / 3600.0

# ── Couleurs et style global ───────────────────────────────────────────────────
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12,
                     "axes.titleweight": "bold", "figure.facecolor": "white"})

ORANGE = "#E07020"
ROUGE  = "#CC3030"
BLEU   = "#1F5F99"
VERT   = "#2E8B57"

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 4 graphiques simples
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(4, 1, figsize=(13, 22), constrained_layout=True)
fig.suptitle("Beton de chanvre 30 cm — Macon +2C\n"
             "Simulation libre (sans climatisation)",
             fontsize=14, fontweight="bold")

# Etiquettes jours sous chaque graphique
def add_day_labels(ax, t, y_pos):
    n_days = int(t[-1] / 24) + 1
    for d in range(n_days + 1):
        ax.axvline(d * 24, color="#CCCCCC", lw=0.8, zorder=0)
    for d in range(n_days):
        ax.text(d * 24 + 12, y_pos, f"Jour {d+1}",
                ha="center", va="bottom", fontsize=9, color="#888888")


# ── G1 : Temperatures ete ─────────────────────────────────────────────────────
ax = axes[0]
ax.plot(t_e, Te, color=ORANGE, lw=2.0, ls="--", label="Temperature exterieure")
ax.plot(t_e, Tr, color=ROUGE,  lw=2.5,           label="Temperature piece")
ax.fill_between(t_e, Tr, 28,
                where=(Tr > 28), color=ROUGE, alpha=0.15,
                label="Inconfort (> 28 C)")
ax.axhline(28, color=ROUGE,   lw=1.2, ls="-.", alpha=0.7, label="Seuil DH 28 C")

dep_ete = draw_dephasage_arrow(ax, t_e, Te, Tr, steps_day, pick_day=3)

y_lim = (Te.min() - 2, max(Tr.max(), Te.max()) + 4)
ax.set_ylim(y_lim)
ax.set_ylabel("Temperature [C]", fontsize=12)
ax.set_title(f"ETE — semaine la plus chaude"
             f"   |   T ext max {Te.max():.0f} C"
             f"   |   T piece max {Tr.max():.0f} C"
             f"   |   Dephasage {dep_ete:.0f} h")
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.15)
ax.set_xlim(0, t_e[-1])
add_day_labels(ax, t_e, y_lim[0] + 0.3)

# ── G2 : Temperatures hiver ───────────────────────────────────────────────────
ax = axes[1]
ax.plot(t_h, Te_h, color=BLEU,  lw=2.0, ls="--", label="Temperature exterieure")
ax.plot(t_h, Tr_h, color=ROUGE, lw=2.5,           label="Temperature piece")
ax.fill_between(t_h, Tr_h, 19,
                where=(Tr_h < 19), color=BLEU, alpha=0.15,
                label="Sous confort (< 19 C)")
ax.axhline(19, color=BLEU, lw=1.2, ls="-.", alpha=0.7, label="Confort min 19 C")

dep_hiv = draw_dephasage_arrow(ax, t_h, Te_h, Tr_h, steps_day, pick_day=3)

y_lim_h = (Te_h.min() - 2, Tr_h.max() + 4)
ax.set_ylim(y_lim_h)
ax.set_ylabel("Temperature [C]", fontsize=12)
ax.set_title(f"HIVER — semaine la plus froide"
             f"   |   T ext min {Te_h.min():.0f} C"
             f"   |   T piece min {Tr_h.min():.0f} C"
             f"   |   Dephasage {dep_hiv:.0f} h")
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.15)
ax.set_xlim(0, t_h[-1])
add_day_labels(ax, t_h, y_lim_h[0] + 0.3)

# ── G3 : Flux de chaleur (ete) ────────────────────────────────────────────────
ax = axes[2]
ax.plot(t_e, q_e, color=ROUGE, lw=2.2, label="Flux conductif paroi Sud")
ax.axhline(0, color="black", lw=1.0)
ax.fill_between(t_e, q_e, 0,
                where=(q_e > 0), color=ROUGE, alpha=0.20,
                label="Chaleur entrant dans la piece")
ax.fill_between(t_e, q_e, 0,
                where=(q_e < 0), color=BLEU, alpha=0.20,
                label="Chaleur stockee dans le mur (recharge nocturne)")
ax.set_ylabel("Flux [W/m2]", fontsize=12)
ax.set_title("ETE — Flux de chaleur a la surface interieure de la paroi Sud\n"
             "(> 0 : chaleur vers la piece  |  < 0 : mur absorbe / restitue la nuit)")
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.15)
ax.set_xlim(0, t_e[-1])
add_day_labels(ax, t_e, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]))

# ── G4 : Humidite relative ────────────────────────────────────────────────────
ax = axes[3]
ax.plot(t_e, RHe,  color=ORANGE, lw=2.5, label="Humidite piece — ete")
ax.plot(t_h, RHh,  color=BLEU,   lw=2.5, ls="--", label="Humidite piece — hiver")
ax.axhline(30, color="#999999", lw=1.2, ls=":", label="30% min confort")
ax.axhline(70, color="#999999", lw=1.2, ls=":", label="70% max confort")
ax.fill_between(t_e, 30, 70, color=VERT, alpha=0.07, label="Zone de confort 30-70%")
ax.set_ylim(10, 100)
ax.set_ylabel("Humidite relative [%]", fontsize=12)
ax.set_xlabel("Heures depuis le debut de la semaine", fontsize=12)
ax.set_title("Humidite relative de la piece — ete et hiver\n"
             "(effet tampon hygroscopique du chanvre)")
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.15)
ax.set_xlim(0, max(t_e[-1], t_h[-1]))
add_day_labels(ax, t_e if len(t_e) > len(t_h) else t_h, 11)

fig.savefig(os.path.join(DIR, "chanvre_4graphiques.pdf"), dpi=150)
plt.close(fig)
print("  [OK] chanvre_4graphiques.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Vue 3D "Beton de chanvre" — parois jointes, echelle reelle
# ══════════════════════════════════════════════════════════════════════════════
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sim_ete = ete["sim"]

# Dimensions interieures de la piece [m]
Lx, Ly, H = 10.0, 8.0, 2.7
t = 0.30   # epaisseur reelle du mur (pas de magnification)

# Couleurs
C_CHANVRE  = "#8BB04A"   # vert chanvre — murs
C_INTERIEUR = "#F5F0E8"  # creme pale  — air interieur
C_SOL      = "#D9CDB5"   # beige       — dalle sol
C_VITRE    = "#AED6F1"   # bleu ciel   — vitrage


def box_faces(x0, x1, y0, y1, z0, z1):
    """Retourne les 6 faces d'un parallelepipede."""
    v = np.array([
        [x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],  # bas
        [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1],  # haut
    ])
    return [v[f] for f in [
        [0,1,2,3],[4,5,6,7],   # dessus/dessous
        [0,1,5,4],[1,2,6,5],   # faces laterales
        [2,3,7,6],[3,0,4,7],
    ]]


def add_box(ax, x0,x1,y0,y1,z0,z1, color, alpha=0.90, ec="#333333", lw=0.4):
    faces = box_faces(x0,x1,y0,y1,z0,z1)
    poly  = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                             edgecolor=ec, linewidths=lw, zsort="min")
    ax.add_collection3d(poly)


fig3d = plt.figure(figsize=(12, 8), facecolor="white")
ax3d  = fig3d.add_subplot(111, projection="3d")
ax3d.set_facecolor("#FAFAFA")

# ── Dalle sol (interieur) ──────────────────────────────────────────────────────
add_box(ax3d, -t, Lx+t, -t, Ly+t, -0.05, 0,
        color=C_SOL if False else "#C8B99A", alpha=0.85, ec="#999999")

# ── 4 murs joincts aux coins — epaisseur reelle 30 cm ─────────────────────────
# Convention : S/N couvrent les angles complets (x de -t a Lx+t)
#              E/W remplissent entre les murs S et N (y de 0 a Ly)
walls_3d = [
    # (nom,      x0,    x1,    y0,   y1,    label_x, label_y, label_z)
    ("Sud",   -t,   Lx+t,  -t,   0,    Lx/2,   -t/2,   H/2),
    ("Nord",  -t,   Lx+t,  Ly,   Ly+t, Lx/2,   Ly+t/2, H/2),
    ("Ouest", -t,   0,     0,    Ly,   -t/2,   Ly/2,   H/2),
    ("Est",   Lx,   Lx+t,  0,    Ly,   Lx+t/2, Ly/2,   H/2),
]

for nom, x0, x1, y0, y1, lx, ly, lz in walls_3d:
    add_box(ax3d, x0, x1, y0, y1, 0, H, color=C_CHANVRE, alpha=0.92)
    ax3d.text(lx, ly, lz,
              f"{nom}\n30 cm",
              ha="center", va="center", fontsize=10, fontweight="bold",
              color="white",
              bbox=dict(boxstyle="round,pad=0.15", fc=C_CHANVRE,
                        ec="none", alpha=0.0))

# ── Volume interieur (transparent pour voir l'interieur) ─────────────────────
add_box(ax3d, 0, Lx, 0, Ly, 0, H,
        color=C_INTERIEUR, alpha=0.12, ec="#BBBBBB", lw=0.5)

# ── Vitrage Sud (8 m2, represente au centre de la facade Sud) ─────────────────
w_vit = 3.0;  h_vit = 1.4   # dimensions approximatives
xv0 = Lx/2 - w_vit/2;  xv1 = Lx/2 + w_vit/2
zv0 = 0.9;               zv1 = zv0 + h_vit
vit_verts = np.array([
    [xv0, 0, zv0],[xv1, 0, zv0],[xv1, 0, zv1],[xv0, 0, zv1]
])
ax3d.add_collection3d(
    Poly3DCollection([vit_verts], alpha=0.45, facecolor=C_VITRE,
                     edgecolor="#2980B9", linewidths=1.0))

# ── Axes et mise en forme ─────────────────────────────────────────────────────
ax3d.set_xlim(-t, Lx+t)
ax3d.set_ylim(-t, Ly+t)
ax3d.set_zlim(0, H)
ax3d.set_xlabel("x [m]", fontsize=10, labelpad=6)
ax3d.set_ylabel("y [m]", fontsize=10, labelpad=6)
ax3d.set_zlabel("z [m]", fontsize=10, labelpad=4)

# Ratio visuel realiste
ax3d.set_box_aspect((Lx + 2*t, Ly + 2*t, H))
ax3d.view_init(elev=25, azim=-50)

# Legende
patch_mur = mpatches.Patch(color=C_CHANVRE,   label="Beton de chanvre 30 cm")
patch_vit = mpatches.Patch(color=C_VITRE,     label="Vitrage Sud 8 m2")
ax3d.legend(handles=[patch_mur, patch_vit], loc="upper left",
            fontsize=10, framealpha=0.9)

U_val = sim_ete.layers[0].U_value()
ax3d.set_title(
    f"Piece — Beton de chanvre 30 cm\n"
    f"Dimensions interieures {Lx:.0f} x {Ly:.0f} x {H:.1f} m   "
    f"|   U paroi = {U_val:.3f} W/(m2.K)",
    fontsize=12, fontweight="bold"
)

fig3d.savefig(os.path.join(DIR, "chanvre_3d.pdf"), dpi=150)
plt.close(fig3d)
print("  [OK] chanvre_3d.pdf")

print(f"\n  Dephasage ete  (T_ext -> T_piece) : {dep_ete:.0f} h")
print(f"  Dephasage hiver (T_ext -> T_piece): {dep_hiv:.0f} h")
