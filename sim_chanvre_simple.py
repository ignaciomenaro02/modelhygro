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
# FIGURE 2 — Vue 3D "Beton de chanvre"
# ══════════════════════════════════════════════════════════════════════════════
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sim_ete = ete["sim"]
Lx, Ly, H = 10.0, 8.0, 2.7
cx, cy     = Lx / 2.0, Ly / 2.0
mag        = 5.0    # epaisseur x5 pour la visibilite

# Couleur unique : chanvre vert naturel
MAT_COLOR = {"Hempcrete": "#8BB04A"}

def _placement(orient, width):
    o = orient.lower()
    if o == "s":  return (np.array([cx-width/2, 0,  0]), np.array([width,0,0]), np.array([0,0,H]), np.array([ 0,-1,0]))
    if o == "n":  return (np.array([cx-width/2, Ly, 0]), np.array([width,0,0]), np.array([0,0,H]), np.array([ 0, 1,0]))
    if o == "e":  return (np.array([Lx, cy-width/2, 0]), np.array([0,width,0]), np.array([0,0,H]), np.array([ 1, 0,0]))
    if o == "w":  return (np.array([0,  cy-width/2, 0]), np.array([0,width,0]), np.array([0,0,H]), np.array([-1, 0,0]))
    return (np.array([cx-width/2, 0, 0]), np.array([width,0,0]), np.array([0,0,H]), np.array([0,-1,0]))

def _slab(P, u, v, n, d0, d1):
    c = []
    for d in (d0, d1):
        for a, b in ((0,0),(1,0),(1,1),(0,1)):
            c.append(P + a*u + b*v + d*n)
    c = np.array(c)
    return [c[f] for f in [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]]

fig3d = plt.figure(figsize=(11, 8), facecolor="white")
ax3d  = fig3d.add_subplot(111, projection="3d")
ax3d.set_facecolor("#F8F8F8")

ORIENT_LABEL = {"S": "Sud", "N": "Nord", "E": "Est", "W": "Ouest"}

for cfg, layer in zip(sim_ete.wall_configs, sim_ete.layers):
    width = max(cfg.area / H, 1.0)
    P, u, v, n = _placement(cfg.orientation, width)
    thk  = float(layer.emat[0]) * mag
    faces = _slab(P, u, v, n, 0, thk)
    poly  = Poly3DCollection(faces, alpha=0.88,
                             facecolor=MAT_COLOR["Hempcrete"],
                             edgecolor="#4A6820", linewidths=0.5)
    ax3d.add_collection3d(poly)
    # Etiquette sur la face exterieure
    centre = P + 0.5*u + 0.5*v + thk*n
    label  = ORIENT_LABEL.get(cfg.orientation.upper(), cfg.orientation)
    ax3d.text(*centre,
              f"{label}\n{layer.emat[0]*100:.0f} cm\nU={layer.U_value():.2f} W/m2K",
              fontsize=9, ha="center", va="center", color="white", fontweight="bold")

# Boite de la piece (contour fin)
for xs, ys, zs in [
    ([0,Lx,Lx,0,0],[0,0,0,0,0],[0,0,H,H,0]),
    ([0,Lx,Lx,0,0],[Ly,Ly,Ly,Ly,Ly],[0,0,H,H,0]),
    ([0,0],[0,0],[0,H]), ([Lx,Lx],[0,0],[0,H]),
    ([0,0],[Ly,Ly],[0,H]), ([Lx,Lx],[Ly,Ly],[0,H]),
]:
    ax3d.plot(xs, ys, zs, color="#AAAAAA", lw=0.8, ls="--")

# Legende
patch = mpatches.Patch(color=MAT_COLOR["Hempcrete"], label="Beton de chanvre 30 cm")
ax3d.legend(handles=[patch], loc="upper left", fontsize=11,
            title="Materiau", title_fontsize=10)

ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
ax3d.set_title("Piece — Beton de chanvre 30 cm\n"
               f"4 parois  |  U = {sim_ete.layers[0].U_value():.3f} W/(m2.K)  "
               f"|  Epaisseur affichee x{mag:.0f}",
               fontsize=12, fontweight="bold")
ax3d.set_box_aspect((Lx, Ly, H))
ax3d.view_init(elev=22, azim=-50)

fig3d.savefig(os.path.join(DIR, "chanvre_3d.pdf"), dpi=150)
plt.close(fig3d)
print("  [OK] chanvre_3d.pdf")

print(f"\n  Dephasage ete  (T_ext -> T_piece) : {dep_ete:.0f} h")
print(f"  Dephasage hiver (T_ext -> T_piece): {dep_hiv:.0f} h")
