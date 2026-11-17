# -*- coding: utf-8 -*-
"""
simulation.py
=============
Heat and moisture simulation of one room (5 × 5 × 2.5 m by default) with hygroscopic walls,
driven by the RE2020 conventional weather.

Everything that is normally changed is in SECTION 1. Variants are better written as scenarios
(scenarios.py) than by editing section 1 each time.

    python simulation.py                                  # section 1 as it is
    python simulation.py --zone H3 --mode Th-D            # another climate
    python simulation.py --start 15/07 --end 31/08        # another period ("01/01" to "01/01" = one year)
    python simulation.py --scenario poster_hemp+hemp_sourced
    python run_scenarios.py name1 name2 --parallel 3      # several runs and a comparison table

Each run writes a new folder results/<date>_<time>_<scenario>_<zone>_<mode>_<start>-<end>/ with
    model_3d, climate, series_T_RH, wall_fields_hottest, wall_fields_coldest,
    comfort_map, degree_hours, heating (PDF figures, in French)
    run_config.txt   every setting used
    summary.json     key indicators
    re2020_report.txt
    series.csv       hourly room values
    wall_fields.npz  hourly temperature and RH at every node of every wall

Sections: 1 configuration, 2 command line, 3 derived quantities, 4 weather, 5 building,
6 run, 7 indicators, 8 figures, 9 saved results.
"""

import os, sys, argparse, locale, json
from datetime import date, timedelta, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

DIR = os.path.dirname(os.path.abspath(__file__))
if DIR not in sys.path:
    sys.path.insert(0, DIR)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
try:
    locale.setlocale(locale.LC_TIME, "French_France.1252")   # French month names in the figures
except Exception:
    pass

import library       as lib
import heating       as re_gains
from wall_config     import make_wall, WindowConfig, ThermalBridge, OpeningConfig
from wall_layer      import _MAT_REGISTRY, apply_material_params
from sources         import (OccupantConfig, EquipmentConfig, LightingConfig,
                             VentilationConfig, HVACConfig, build_daily_schedule)
from solar           import SolarCalculator
from heating         import RE2020Heating, occupancy_series, internal_gains_series, n_adeq
from re2020          import (RE2020Evaluator, adaptive_discomfort, hot_discomfort_limit,
                             adaptive_period, DH_MAX)
from room_simulation import RoomSimulation
from weather         import load_re2020_weather


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ── Weather (RE2020 conventional weather) ─────────────────────────────────────
ZONE = "H1c"        # climate zone: H1a H1b H1c H2a H2b H2c H2d H3
MODE = "Th-BC"      # "Th-BC" = typical year (energy) | "Th-D" = year with a heat wave (summer comfort)

# ── Simulated period ──────────────────────────────────────────────────────────
# "day/month"; END_DATE is excluded. Same start and end = one full year ("01/01" → "01/01").
# If END_DATE comes before START_DATE the period runs over the new year ("01/11" → "01/03").
START_DATE = "01/01"
END_DATE   = "12/03"

# Spin-up: the walls keep their moisture for months, so the run first simulates SPINUP_DAYS days
# ending where the period starts (the weather year is repeated), and keeps only the final state
# of the walls, the room air and the heating season. Nothing of the spin-up is in the results.
SPINUP_DAYS = 0

# ── Room [m] ──────────────────────────────────────────────────────────────────
LX = 5.0            # interior length, East–West
LY = 5.0            # interior depth, South–North
H  = 2.5            # ceiling height

# ── Walls ─────────────────────────────────────────────────────────────────────
# Layers from OUTSIDE to INSIDE, (material, thickness in m). The materials are the classes of
# materials_library.py, e.g. Hempcrete, Rammed_Earth, Earth_Plaster, Lime_Plaster, BA13, Wood_Fiber:
#   [("Hempcrete", 0.30), ("Earth_Plaster", 0.02)]     hemp concrete + earth plaster inside
#   [("Wood_Fiber", 0.10), ("Rammed_Earth", 0.30)]     rammed earth insulated outside
WALL_STACK  = [("Hempcrete", 0.30)]
WALL_LAYERS = {"South": WALL_STACK, "North": WALL_STACK, "East": WALL_STACK, "West": WALL_STACK}

# Property values of the materials with a PARAMS table (Hempcrete, Rammed_Earth): a number, or
# "min" / "max" / "nominal" of the literature range. E.g. {"Hempcrete": {"k_dry": "max"}}.
MATERIAL_PARAMS = {}

# ── Roof ──────────────────────────────────────────────────────────────────────
# ROOF = True adds a fifth, horizontal wall on top of the room; False leaves the ceiling adiabatic.
# Layers from outside (top) to inside: breathable membrane, insulation, concrete slab.
ROOF       = True
ROOF_STACK = [("Rain_Barrier", 0.002), ("Rock_Wool", 0.20), ("Concrete", 0.20)]
ROOF_HYGRO = False  # False: the roof exchanges heat only, no vapour, so the moisture comes from the walls
if ROOF:
    WALL_LAYERS["Roof"] = ROOF_STACK

# ── Windows, thermal bridges, air leaks ───────────────────────────────────────
# The opaque area of each wall is its length × height minus its windows. None for now.
WINDOWS = [
    # WindowConfig("South window", area=1.5, orientation="S", U_value=1.1, g_value=0.60, shading=0.30),
]
THERMAL_BRIDGES = [
    # ThermalBridge("Window frames", psi=0.04, length=6.0),
]
INFILTRATION = [OpeningConfig("Infiltration", ach_contribution=0.1)]   # [1/h]

# ── Ventilation [1/h] ─────────────────────────────────────────────────────────
ACH_BASE      = 0.50        # hygiene ventilation, all day
ACH_OVERVENT  = 4.0         # added on summer nights (0 = constant ventilation)
NIGHT_HOURS   = list(range(0, 8)) + list(range(22, 24))    # 22h → 8h
SUMMER_MONTHS = (5, 9)      # May → September
# True: the night overventilation is window opening by the occupants (RE2020 manual window
# management): open at 23 °C, closed below 22 °C, only when outdoors is cooler, scaled with the
# outdoor temperature. False: ACH_OVERVENT on every summer night.
OVERVENT_CONTROL = False

# ── Internal gains ────────────────────────────────────────────────────────────
# "RE2020": conventional dwelling scenario (heating.py): 1.28 equivalent adults for 25 m², 90 W
#           and 0.055 kg/h of vapour each when awake at home, 70 % asleep, 0 away; equipment
#           5.7 W/m² awake / 1.14 W/m² otherwise; lighting 1.4 W/m². The values below are ignored.
# "SIMPLE": the values below, every day.
INTERNAL_GAINS  = "SIMPLE"
N_OCCUPANTS     = 2.0
Q_SENSIBLE      = 80.0      # [W/person]
Q_LATENT        = 60.0      # [W/person] water vapour, as latent power (60 W ≈ 86 g/h)
OCCUPANCY_HOURS = list(range(8, 22))
EQUIPMENT_W     = 100.0     # on when the occupants are there
LIGHTING_W_M2   = 5.0
LIGHTING_HOURS  = list(range(20, 24))

# ── Heating (RE2020 scenario, see heating.py) ─────────────────────────────────
# Ideal heater: 19 °C when occupied, 16 °C when reduced (weekday absences, Christmas week).
HEATING            = True
HEATING_SETPOINT   = 19.0   # [°C]
HEATING_SETBACK    = 16.0   # [°C]
HEATING_EFFICIENCY = 1.0    # 1 = direct electric heater; the need does not depend on it
HEATING_MAX_W      = 3000.0
# "RE2020": the regulatory season algorithm, which opens and closes the season on different dates
# for different walls. (first_day, last_day): the same calendar for every run, to compare walls at
# equal heating service, e.g. (274, 120) = 1 October → 30 April.
HEATING_SEASON     = "RE2020"

# ── RE2020 indicators (re2020.py) ─────────────────────────────────────────────
# DH (summer comfort, regulatory in Th-D mode only) and, with heating, the heating part of Bbio
# and of the primary energy. Cooling, lighting and hot water are not modelled.
RE2020 = True

# ── Physics ───────────────────────────────────────────────────────────────────
HYGRO_EFFECT = True         # False = dry walls, heat conduction only (as in dynamic thermal simulation)
# Temperature of the comfort indicators: "operative" = (air + mean radiant)/2, as RE2020 and
# Pléiades/COMFIE; "air" = room air. The mean radiant temperature averages the inner surfaces
# of the walls by area; the floor (and the ceiling without roof) is taken at air temperature.
COMFORT_TEMP = "operative"
# Interior coating as a vapour resistance at the inner surface, Sd [m] per wall (paint 0.05-0.5,
# PE film ≈ 100). Unlike a Vapor_Barrier layer it adds no mass and needs no fine mesh.
WALL_INTERIOR_SD = {}       # e.g. {"North": 100.0}

# Surface heat transfer, convection + radiation [W/(m²·K)] (EN ISO 6946: Rse 0.04, Rsi 0.13)
H_EXT, H_INT = 25.0, 8.0
# Convective part only, which sets the vapour transfer (Lewis relation, section 3):
# EN ISO 6946 h_c,ext = 4 + 4·v with v = 4 m/s; h_c,int = 2.5 for a vertical wall
H_C_EXT, H_C_INT = 20.0, 2.5

# ── Numerics ──────────────────────────────────────────────────────────────────
DT             = 3600.0     # time step [s]: weather and schedules are hourly, do not change
WALL_SUBSTEPS  = 6          # wall steps per hour (600 s each)
MESH_OPT       = 2          # 2 = equal elements inside each layer (see wall_layer.py)
MESH_SIZE      = 0.01       # element size in the walls [m]
MESH_SIZE_ROOF = 0.01       # element size in the roof [m]

# ── Figures ───────────────────────────────────────────────────────────────────
PROFILE_HOURS = [6, 14, 22] # hours drawn in the wall_fields figures


# ══════════════════════════════════════════════════════════════════════════════
# 2. COMMAND LINE AND SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════

_ap = argparse.ArgumentParser(description="Heat and moisture simulation of a room.")
_ap.add_argument("--zone");  _ap.add_argument("--mode")
_ap.add_argument("--start"); _ap.add_argument("--end")
_ap.add_argument("--heating", choices=["on", "off"])
_ap.add_argument("--re2020",  choices=["on", "off"])
_ap.add_argument("--scenario", help="named variant(s) of scenarios.py, combined with '+'")
_ap.add_argument("--spinup", type=int, help="SPINUP_DAYS")
_args, _ = _ap.parse_known_args()

# Scenarios replace section 1 values, from left to right; the command-line flags come last
SCENARIO, SCENARIO_NOTE = "base", ""
if _args.scenario:
    from scenarios import SCENARIOS
    SCENARIO, _notes = _args.scenario, []
    for _name in SCENARIO.split("+"):
        if _name not in SCENARIOS:
            raise SystemExit(f"Unknown scenario '{_name}'. Available: {', '.join(SCENARIOS)}")
        _ov = dict(SCENARIOS[_name])
        _notes.append(_ov.pop("note", _name))
        _ov = {k: v for k, v in _ov.items() if k.isupper()}
        _bad = [k for k in _ov if k not in globals()]
        if _bad:
            raise SystemExit(f"Scenario '{_name}': unknown parameter(s) {_bad}. "
                             f"Use the upper-case variable names of section 1.")
        if "WALL_STACK" in _ov:                      # the four vertical walls, never the roof
            WALL_STACK  = _ov.pop("WALL_STACK")
            WALL_LAYERS = {**WALL_LAYERS, **{n: WALL_STACK for n in ("South", "North", "East", "West")}}
        if "WALL_LAYERS" in _ov:                     # only the walls listed
            WALL_LAYERS = {**WALL_LAYERS, **_ov.pop("WALL_LAYERS")}
        globals().update(_ov)
    SCENARIO_NOTE = " | ".join(_notes)

ZONE        = _args.zone  or ZONE
MODE        = _args.mode  or MODE
START_DATE  = _args.start or START_DATE
END_DATE    = _args.end   or END_DATE
HEATING     = HEATING if _args.heating is None else (_args.heating == "on")
RE2020      = RE2020  if _args.re2020  is None else (_args.re2020 == "on")
SPINUP_DAYS = SPINUP_DAYS if _args.spinup is None else _args.spinup
ZONE        = ZONE[:2].upper() + ZONE[2:].lower()          # h1C → H1c

# The settings of this run, written to run_config.txt
_CONFIG_NAMES = [k for k in list(globals()) if k.isupper() and k not in ("DIR", "SCENARIOS")]


# ══════════════════════════════════════════════════════════════════════════════
# 3. DERIVED QUANTITIES
# ══════════════════════════════════════════════════════════════════════════════

def _doy(txt):
    """'21/06' → day of the year (the weather year has 365 days)."""
    d, m = (int(x) for x in txt.strip().split("/"))
    return date(2001, m, d).timetuple().tm_yday

DOY_START = _doy(START_DATE)
DOY_END   = _doy(END_DATE)
N_DAYS    = (DOY_END - DOY_START) % 365 or 365

def day_date(k, start_doy=None):
    """Date of day k of a run (0 = first day) starting on start_doy (default: this run)."""
    first = DOY_START if start_doy is None else start_doy
    return date(2001, 1, 1) + timedelta(days=(first - 1 + k) % 365)

def _is_summer(month):
    a, b = SUMMER_MONTHS
    return (month >= a) & (month <= b) if a <= b else (month >= a) | (month <= b)

MATERIALS_IN_FORCE = apply_material_params(MATERIAL_PARAMS)
FLOOR_AREA    = LX * LY
VOLUME        = LX * LY * H
INTERNAL_MASS = 110e3 * FLOOR_AREA          # [J/K] furniture, floor, partitions (ISO 13790, medium)

# Surface vapour transfer coefficients from the convective heat transfer (Lewis relation, Le ≈ 1):
#   hm = h_c / (rho_air · cp_air · Rv · T)  at 20 °C   →  exterior ≈ 1.2e-7, interior ≈ 1.5e-8 kg/(m²·s·Pa)
# Dry walls get a negligible value instead of 0, which would leave the moisture equation undefined.
_LEWIS = 1.0 / (1.2 * lib.CpA * lib.Rv * 293.15)
HM_EXT = H_C_EXT * _LEWIS if HYGRO_EFFECT else 1e-30
HM_INT = H_C_INT * _LEWIS if HYGRO_EFFECT else 1e-30

ORIENT  = {"South": "S", "North": "N", "East": "E", "West": "W"}
WIDTH   = {"South": LX, "North": LX, "East": LY, "West": LY}
WALL_FR = {"South": "Sud", "North": "Nord", "East": "Est", "West": "Ouest", "Roof": "Toiture"}
if ROOF:
    ORIENT["Roof"] = "roof"
THICKNESS = {n: sum(e for _, e in WALL_LAYERS[n]) for n in ORIENT}

def describe_layers(layers):
    return " + ".join(f"{m} {e*100:g} cm" for m, e in layers)

def wall_u_value(layers):
    """U [W/(m²·K)] with surface resistances, conductivities at 20 °C / 50 %."""
    return 1.0 / (1.0 / H_INT + 1.0 / H_EXT + sum(e / _MAT_REGISTRY[m].k(293.15, 0.5) for m, e in layers))

# Short description for the figure titles (the roof is meant to differ, so it is left out)
DESCRIPTION = (describe_layers(WALL_LAYERS["South"])
               if all(WALL_LAYERS[n] == WALL_LAYERS["South"] for n in ("North", "East", "West"))
               else "parois mixtes")


# ══════════════════════════════════════════════════════════════════════════════
# 4. WEATHER AND OUTPUT FOLDER
# ══════════════════════════════════════════════════════════════════════════════

wx = load_re2020_weather(os.path.join(DIR, "scenarios_meteorologiques_th-bc_th-d_re2020.xlsx"), ZONE, MODE)
LABEL, TAG = f"RE2020 {ZONE} ({MODE})", f"{ZONE}_{MODE}"
N_YEAR = len(wx.T_ext)

# Outdoor conditions of the period: hour 0 = 00:00 of START_DATE
idx       = ((DOY_START - 1) * 24 + np.arange(N_DAYS * 24)) % N_YEAR
time_bc   = np.arange(N_DAYS * 24) * DT
T_ext_bc  = wx.T_ext[idx]
RH_ext_bc = wx.RH_ext[idx]
n_steps   = len(time_bc) - 1

# Running mean of the daily outdoor temperature, T_rm = 0.8·T_rm(previous day) + 0.2·T(previous day),
# over two repeated years so that 1 January already has a full history. Used by the summer comfort.
T_day_year = wx.T_ext[:(N_YEAR // 24) * 24].reshape(-1, 24).mean(axis=1)
_seq   = np.concatenate([T_day_year, T_day_year])
_rm    = np.full(len(_seq), _seq[0])
for d in range(1, len(_seq)):
    _rm[d] = 0.2 * _seq[d - 1] + 0.8 * _rm[d - 1]
rm_year = _rm[len(T_day_year):]                       # T_rm of each day of the year

# A new folder for every run (never overwritten)
RESULTS_ROOT = os.environ.get("SIM_RESULTS_DIR") or os.path.join(DIR, "results")
RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(RESULTS_ROOT, f"{RUN_ID}_{SCENARIO}_{TAG}_"
                       f"{START_DATE.replace('/', '')}-{END_DATE.replace('/', '')}")
_base_dir, _k = OUT_DIR, 1
while os.path.exists(OUT_DIR):                        # two runs in the same second
    _k += 1
    OUT_DIR = f"{_base_dir}_{_k}"
os.makedirs(OUT_DIR)

print(f"Weather : {LABEL}   T {wx.T_ext.min():.1f}..{wx.T_ext.max():.1f} °C")
print(f"Period  : {START_DATE} → {END_DATE}  ({N_DAYS} days, {n_steps} steps)")
for n in ORIENT:
    print(f"Wall {n:5s}: {describe_layers(WALL_LAYERS[n])}   U = {wall_u_value(WALL_LAYERS[n]):.3f} W/(m²·K)")
print(f"Scenario: {SCENARIO}" + (f"  ({SCENARIO_NOTE})" if SCENARIO_NOTE else ""))
print(f"Output  : {os.path.relpath(OUT_DIR, DIR)}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def create_walls(quiet=False):
    """WallConfig of each wall (and of the roof)."""
    walls = []
    for name, ori in ORIENT.items():
        layers = WALL_LAYERS[name]
        if name == "Roof":
            area, tilt, mesh = LX * LY, 0.0, MESH_SIZE_ROOF
        else:
            area = WIDTH[name] * H - sum(w.area for w in WINDOWS if w.orientation == ori)
            tilt, mesh = 90.0, MESH_SIZE
        if MESH_OPT == 0:
            # one element size for the whole wall: a thin layer forces it down everywhere
            mesh = min(mesh, min(e for _, e in layers) / 2)
            if mesh < MESH_SIZE and not quiet:
                print(f"  [!] Thin layer in {name}: mesh reduced to {mesh*1000:g} mm (slower run)")
        # an interior coating adds its resistance in series: 1/hm = 1/hm_int + Sd/delta_air
        sd     = WALL_INTERIOR_SD.get(name, 0.0)
        hm_int = 1.0 / (1.0 / HM_INT + sd / lib.Da(293.15)) if sd > 0 else HM_INT
        hm_ext = HM_EXT
        if name == "Roof" and not ROOF_HYGRO:
            hm_int = hm_ext = 1e-30
        walls.append(make_wall(name, area=area, orientation=ori, layers=layers, tilt=tilt,
                               Mesh_Opt=MESH_OPT, mesh_size=mesh, h_ext=H_EXT, hm_ext=hm_ext,
                               h_int=H_INT, hm_int=hm_int, hygro=HYGRO_EFFECT))
    return walls


# In Th-D mode the RE2020 forbids heating during the adaptive-comfort period
ADAPTIVE_DAYS = adaptive_period(rm_year)
adaptive = ADAPTIVE_DAYS if (HEATING and MODE == "Th-D") else None


def internal_gains(start_doy, n_steps):
    """Occupants, equipment and lighting for n_steps hours from start_doy (see INTERNAL_GAINS)."""
    if INTERNAL_GAINS == "RE2020":
        occ, equip, light = internal_gains_series(start_doy, n_steps)
        latent_w = re_gains.MOIST_ADULT_KG_H * 2.5e6 / 3600.0      # 0.055 kg/h as a latent power
        return (OccupantConfig(n_adeq(FLOOR_AREA), re_gains.GAIN_ADULT_W, latent_w, occ.tolist()),
                EquipmentConfig(re_gains.EQUIP_W_M2 * FLOOR_AREA, schedule=equip.tolist()),
                LightingConfig(re_gains.LIGHT_W_M2, FLOOR_AREA, schedule=light.tolist()))
    occ_sched   = build_daily_schedule(OCCUPANCY_HOURS, n_steps, int(DT))
    light_sched = build_daily_schedule(LIGHTING_HOURS,  n_steps, int(DT))
    return (OccupantConfig(N_OCCUPANTS, Q_SENSIBLE, Q_LATENT, occ_sched),
            EquipmentConfig(EQUIPMENT_W, schedule=occ_sched),
            LightingConfig(LIGHTING_W_M2, FLOOR_AREA, schedule=light_sched))


def build_simulation(start_doy, n_steps, T_room0, RH_room0, evaluate=True, quiet=False):
    """
    The room for n_steps hours from 00:00 of day start_doy. Called twice: for the spin-up and
    for the period itself. Returns (simulation, heating controller or None, RE2020 evaluator or None).
    """
    occupants, equipment, lighting = internal_gains(start_doy, n_steps)

    # Ventilation: ACH_BASE all the time, plus ACH_OVERVENT on summer nights
    step     = np.arange(n_steps)
    month    = np.array([day_date(int(k), start_doy).month for k in step // 24])
    vent_on  = np.isin(step % 24, NIGHT_HOURS) & _is_summer(month)
    if OVERVENT_CONTROL:
        ventilation = VentilationConfig(n_ach=ACH_BASE, overvent_ach=ACH_OVERVENT,
                                        overvent_window=vent_on.tolist())
    else:
        ach_total   = ACH_BASE + ACH_OVERVENT
        base_ratio  = ACH_BASE / ach_total if ach_total > 0 else 1.0
        ventilation = VentilationConfig(n_ach=ach_total,
                                        schedule=np.where(vent_on, 1.0, base_ratio).tolist())

    hvac, ctrl = None, None
    if HEATING:
        ctrl = RE2020Heating(start_doy, FLOOR_AREA, HEATING_SETPOINT, HEATING_SETBACK, adaptive,
                             season_window=None if HEATING_SEASON == "RE2020" else HEATING_SEASON)
        hvac = HVACConfig(T_heat_set=HEATING_SETPOINT, T_cool_set=99.0,          # no cooling
                          efficiency_heat=HEATING_EFFICIENCY, max_power_heat=HEATING_MAX_W,
                          heating_control=ctrl)

    ev = RE2020Evaluator(floor_area=FLOOR_AREA, climate_zone=ZONE) if (RE2020 and evaluate) else None

    sim = RoomSimulation(
        wall_configs    = create_walls(quiet),
        window_configs  = WINDOWS,
        bridge_configs  = THERMAL_BRIDGES,
        opening_configs = INFILTRATION,
        occupants       = occupants,
        equipment       = equipment,
        lighting        = lighting,
        ventilation     = ventilation,
        hvac            = hvac,
        solar_calc      = SolarCalculator(wx, start_doy=start_doy, ground_albedo=0.20),
        volume          = VOLUME,
        internal_mass   = INTERNAL_MASS,
        wall_substeps   = WALL_SUBSTEPS,
        T_room_init     = T_room0,
        RH_room_init    = RH_room0,
        re2020          = ev,
    )
    return sim, ctrl, ev


# ══════════════════════════════════════════════════════════════════════════════
# 6. RUN
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nRunning…  night overventilation {ACH_OVERVENT:g} 1/h "
      f"(months {SUMMER_MONTHS[0]}–{SUMMER_MONTHS[1]})   hygro={'ON' if HYGRO_EFFECT else 'OFF'}   "
      f"heating={'RE2020' if HEATING else 'OFF'}")

# Spin-up over the SPINUP_DAYS days before START_DATE; room and walls start at the outdoor state
warm_sim = warm_ctrl = None
if SPINUP_DAYS > 0:
    doy_w = (DOY_START - 1 - SPINUP_DAYS) % 365 + 1
    n_w   = SPINUP_DAYS * 24
    idx_w = ((doy_w - 1) * 24 + np.arange(n_w + 1)) % N_YEAR
    print(f"Spin-up : {SPINUP_DAYS} days before {START_DATE} (results discarded, final state kept)")
    warm_sim, warm_ctrl, _ = build_simulation(doy_w, n_w, float(wx.T_ext[idx_w[0]]),
                                              float(wx.RH_ext[idx_w[0]]), evaluate=False, quiet=True)
    warm_sim.run(np.arange(n_w + 1) * DT, wx.T_ext[idx_w], wx.RH_ext[idx_w], dt=DT)
    T_init, RH_init = warm_sim.T_room, warm_sim.RH_room
else:
    T_init, RH_init = float(T_ext_bc[0]), float(RH_ext_bc[0])

sim, heating_ctrl, re_ev = build_simulation(DOY_START, n_steps, T_init, RH_init)
if warm_sim is not None:
    sim.load_state(warm_sim)
    if heating_ctrl is not None:
        heating_ctrl.continue_from(warm_ctrl)
    del warm_sim, warm_ctrl

T_run, RH_run = sim.run(time_bc, T_ext_bc, RH_ext_bc, dt=DT)


# ══════════════════════════════════════════════════════════════════════════════
# 7. INDICATORS
# ══════════════════════════════════════════════════════════════════════════════
# Hourly arrays of length N; row k = state at hour k of the period (row 0 = initial state)

S       = 24
n_days  = len(T_run) // S
N       = n_days * S
days    = np.arange(n_days)
hour_h  = np.arange(N) % S
T_room  = np.array(T_run)[:N]
RH_room = np.array(RH_run)[:N] * 100
T_out   = np.interp(np.arange(len(T_run)) * DT, time_bc, T_ext_bc)[:N]
RH_out  = np.interp(np.arange(len(T_run)) * DT, time_bc, RH_ext_bc)[:N] * 100

def daily(x, f):
    """Apply f to each day of an hourly array."""
    return np.array([f(x[k*S:(k+1)*S]) for k in range(n_days)])

# Operative temperature = (air + mean radiant)/2, the mean radiant temperature being the
# area-weighted inner surface temperatures (unmodelled floor/ceiling at air temperature)
WALL_NAMES = [cfg.name for cfg in sim.wall_configs]
WALL_AREA  = np.array([cfg.area for cfg in sim.wall_configs])
T_si       = np.column_stack([np.array(sim.StockT_walls[n])[:N] for n in WALL_NAMES])
A_air      = FLOOR_AREA * (1 if ROOF else 2)
T_mrt      = (T_si @ WALL_AREA + A_air * T_room) / (WALL_AREA.sum() + A_air)
T_op       = 0.5 * (T_room + T_mrt)
T_comf     = T_op if COMFORT_TEMP == "operative" else T_room
COMF_FR    = "opérative" if COMFORT_TEMP == "operative" else "de l'air"

T_out_max = daily(T_out, np.max)
T_out_min = daily(T_out, np.min)
HOTTEST   = int(np.argmax(T_out_max))
COLDEST   = int(np.argmin(T_out_min))
fmt_date  = lambda k: day_date(k).strftime("%d %B").lstrip("0")

# Summer comfort. Simplified DH: degree-hours above 28 °C, all hours.
DH_hour = np.maximum(T_comf - 28.0, 0.0)
DH_28   = float(DH_hour.sum())
DH_total = DH_28
# RE2020 DH: occupied hours of the adaptive-comfort period, adaptive limit (re2020.py)
T_rm_h  = np.repeat(rm_year[(DOY_START - 1 + days) % len(rm_year)], S)
occ_h   = occupancy_series(DOY_START, N)
adapt_h = np.repeat(ADAPTIVE_DAYS[(DOY_START - 1 + days) % len(rm_year)], S)
T_lim_h = hot_discomfort_limit(T_rm_h, hour_h, T_cool_set=26.0, category=1)
if RE2020:
    DH_hour  = adaptive_discomfort(T_comf, T_rm_h, occ_h, hour_h, adapt_h, T_cool_set=26.0, category=1)
    DH_total = float(DH_hour.sum())
    re_ev.add_adaptive_discomfort_hours(T_comf, T_rm_h, occ_h, hour_h, adapt_h, T_cool_set=26.0,
                                        category=1, dt=int(DT))

# Cold side, used without heating: degree-days below 18 °C
DJ_total = float(np.maximum(18.0 - T_comf, 0.0).sum() / 24.0)

# Comfort map: occupied hours between the heating setpoint and the RE2020 hot limit
in_zone  = occ_h & (T_comf >= HEATING_SETPOINT) & (T_comf <= T_lim_h)
too_hot  = occ_h & (T_comf >  T_lim_h)
too_cold = occ_h & (T_comf <  HEATING_SETPOINT)
n_occ    = int(occ_h.sum())
pct_ok, pct_hot, pct_cold = (100 * m.sum() / n_occ if n_occ else 0.0 for m in (in_zone, too_hot, too_cold))

# Ventilation and internal gains actually applied (for series.csv)
ACH_h = np.r_[sim.StockACH[:1], sim.StockACH][:N]
_occ, _eq, _li = internal_gains(DOY_START, N)
Q_int_h = np.array([_occ.sensible_at(k) + _eq.heat_at(k) + _li.heat_at(k) for k in range(N)])
G_int_h = np.array([_occ.moisture_at(k) * 3600.0 for k in range(N)])      # [kg/h]

# Humidity: indoor RH and moisture exchanged with the walls
RH_amp      = float(np.mean(daily(RH_room, np.max) - daily(RH_room, np.min)))   # mean daily amplitude
pct_RH_hi   = float(100 * np.mean(RH_room > 70.0))
pct_RH_lo   = float(100 * np.mean(RH_room < 30.0))
Q_lat       = np.array(sim.StockQ_lat_total)[:N]                  # [W], + = the walls release moisture
LAT_REL_KWH = float(Q_lat[Q_lat > 0].sum() * DT / 3.6e6)
LAT_ABS_KWH = float(-Q_lat[Q_lat < 0].sum() * DT / 3.6e6)

if HEATING:
    Q_heat         = np.maximum(np.array(sim.StockQ_HVAC)[:N], 0.0)        # [W]
    HEAT_NEED_KWH  = float(Q_heat.sum() * DT / 3.6e6)
    HEAT_FINAL_KWH = float(sim.E_heat_kWh)
    HEAT_DAYS      = int(sum(heating_ctrl.authorized_days(n_days)))

print(f"\n  Hottest day : {fmt_date(HOTTEST)}  T_out = {T_out_max[HOTTEST]:.1f} °C")
print(f"  Coldest day : {fmt_date(COLDEST)}  T_out = {T_out_min[COLDEST]:.1f} °C")


# ══════════════════════════════════════════════════════════════════════════════
# 8. FIGURES (text in French)
# ══════════════════════════════════════════════════════════════════════════════

def _date_ticks():
    pos, lab = [], []
    if n_days >= 75:                                  # first day of each month
        for k in range(n_days):
            if day_date(k).day == 1:
                pos.append(k); lab.append(day_date(k).strftime("%b" if n_days > 200 else "%B"))
    else:
        for k in range(0, n_days, max(1, n_days // 8)):
            pos.append(k); lab.append(day_date(k).strftime("%d/%m"))
    return pos, lab

TICK_POS, TICK_LAB = _date_ticks()

def set_date_ticks(ax):
    ax.set_xticks(TICK_POS); ax.set_xticklabels(TICK_LAB, fontsize=10)
    ax.set_xlim(-0.5, n_days - 0.5)

def save_fig(fig, name, **kw):
    """Save as PDF; if the file is open in a viewer, save it as <name>_new.pdf instead."""
    path = os.path.join(OUT_DIR, name + ".pdf")
    for target in (path, path.replace(".pdf", "_new.pdf")):
        try:
            fig.savefig(target, dpi=150, **kw)
            print(f"  [OK] {os.path.relpath(target, DIR)}")
            break
        except PermissionError:
            continue
    else:
        print(f"  [!] Could not save {name}.pdf (is it open?)")
    plt.close(fig)

def info_box(ax, text, x, y, ha, va):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=9.5, ha=ha, va=va, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#BBBBBB", alpha=0.92))

TITLE = ((f"[{SCENARIO}] " if SCENARIO != "base" else "") +
         f"{DESCRIPTION} — Pièce {LX:g}×{LY:g} m  |  {LABEL}  |  {START_DATE} → {END_DATE} ({N_DAYS} jours)")


def plot_3d():
    """The room seen from the south-west, walls drawn with their real thickness."""
    tS, tN, tE, tW = (THICKNESS[n] for n in ("South", "North", "East", "West"))

    def box(x0, x1, y0, y1, z0, z1):
        return [[[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]], [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],
                [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]], [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],
                [[x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]], [[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[x1,y0,z1]]]

    def label(x, y, z, text, color):
        ax.text(x, y, z, text, ha="center", va="center", fontsize=11, fontweight="bold", color="#222222",
                zorder=1000, bbox=dict(boxstyle="round,pad=0.15", fc=color, ec="#444444", alpha=0.85, lw=0.8))

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_axes([0.05, 0.0, 0.90, 0.93], projection="3d")
    ax.set_facecolor("#F8F8F8")

    # South and North run over the corners; East and West fit between them
    walls = [("South", -tW, LX+tE, -tS, 0,     "#D4956A", (LX/2, -tS/2)),
             ("North", -tW, LX+tE,  LY, LY+tN, "#7BA7C4", (LX/2, LY+tN/2)),
             ("West",  -tW, 0,      0,  LY,    "#A8C888", (-tW/2, LY/2)),
             ("East",   LX, LX+tE,  0,  LY,    "#C8A8C8", (LX+tE/2, LY/2))]
    for name, x0, x1, y0, y1, col, (lx, ly) in walls:
        ax.add_collection3d(Poly3DCollection(box(x0, x1, y0, y1, 0, H), alpha=0.78,
                            facecolor=col, edgecolor="#444444", linewidth=0.5))
        label(lx, ly, H * 0.55, WALL_FR[name], col)
    ax.add_collection3d(Poly3DCollection([[[0,0,0],[LX,0,0],[LX,LY,0],[0,LY,0]]], alpha=0.30,
                        facecolor="#E8D8B0", edgecolor="#999999", linewidth=0.6))
    if ROOF:
        tR = THICKNESS["Roof"]
        ax.add_collection3d(Poly3DCollection(box(-tW, LX+tE, -tS, LY+tN, H, H+tR), alpha=0.78,
                            facecolor="#C9C0A8", edgecolor="#444444", linewidth=0.5))
        label(LX/2, LY/2, H + tR/2, WALL_FR["Roof"], "#C9C0A8")

    # One window per orientation, with the total glazed area of that orientation
    length = {"S": LX, "N": LX, "E": LY, "W": LY}
    for ori in "SNEW":
        A = sum(w.area for w in WINDOWS if w.orientation == ori)
        if A <= 0:
            continue
        hw = min(np.sqrt(A), H - 0.3); ww = A / hw
        if ww > 0.9 * length[ori]:
            ww = 0.9 * length[ori]; hw = A / ww
        z0 = float(np.clip(0.90 if ori in "SEW" else 1.20, 0.1, H - hw - 0.1))
        c0 = (length[ori] - ww) / 2
        a, b = {"S": ((c0, -tS + 0.01), (c0 + ww, -tS + 0.01)),
                "N": ((c0, LY + tN - 0.01), (c0 + ww, LY + tN - 0.01)),
                "W": ((-tW + 0.01, c0), (-tW + 0.01, c0 + ww)),
                "E": ((LX + tE - 0.01, c0), (LX + tE - 0.01, c0 + ww))}[ori]
        ax.add_collection3d(Poly3DCollection([[[*a, z0], [*b, z0], [*b, z0 + hw], [*a, z0 + hw]]],
                            alpha=0.55, facecolor="#AED6F1", edgecolor="#2980B9", linewidth=1.2))
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2, z0+hw+0.07, f"Vitrage {ori}\n{A:g} m²",
                ha="center", va="bottom", fontsize=7.5, color="#1A5276", zorder=1000)

    ax.quiver(LX/2, LY+tN+0.25, H+0.05, 0, 0.35, 0, color="#CC3030", lw=2.5, arrow_length_ratio=0.45)
    ax.text(LX/2, LY+tN+0.65, H+0.05, "N", ha="center", va="bottom", fontsize=13,
            fontweight="bold", color="#CC3030", zorder=1000)
    ax.plot([0, LX], [-tS-0.18]*2, [0, 0], color="#555555", lw=1.2)
    ax.text(LX/2, -tS-0.22, 0, f"{LX:g} m", ha="center", va="top", fontsize=9, color="#555555")
    ax.plot([LX+tE+0.12]*2, [0, LY], [0, 0], color="#555555", lw=1.2)
    ax.text(LX+tE+0.16, LY/2, 0, f"{LY:g} m", ha="left", va="center", fontsize=9, color="#555555", rotation=90)
    ax.plot([LX+tE+0.12]*2, [LY+tN+0.12]*2, [0, H], color="#555555", lw=1.2)
    ax.text(LX+tE+0.16, LY+tN+0.15, H/2, f"h = {H:g} m", ha="left", va="center", fontsize=9, color="#555555")

    ax.set_xlim(-tW-0.3, LX+tE+0.5); ax.set_ylim(-tS-0.3, LY+tN+0.8); ax.set_zlim(0, H+0.35)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)
    ax.set_title(f"Pièce {LX:g}×{LY:g} m  h = {H:g} m\nSurface : {FLOOR_AREA:g} m²    Volume : {VOLUME:g} m³",
                 fontsize=12, fontweight="bold", pad=10)
    ax.view_init(elev=24, azim=-48)
    save_fig(fig, "model_3d", bbox_inches="tight")


def plot_climate():
    """Outdoor temperature and RH of the period, daily mean and range."""
    fig, (axT, axH) = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(f"Données climatiques — {LABEL}  |  {START_DATE} → {END_DATE} ({N_DAYS} jours)",
                 fontsize=12, fontweight="bold")
    T_mn, T_mx = daily(T_out, np.min), daily(T_out, np.max)
    axT.fill_between(days, T_mn, T_mx, color="#E07020", alpha=0.20,
                     label=f"Plage journalière  (amplitude moy. {(T_mx - T_mn).mean():.1f} °C)")
    axT.plot(days, daily(T_out, np.mean), color="#CC5500", lw=2.0, label="Moyenne journalière")
    axT.set_ylabel("Température extérieure [°C]", fontsize=11)
    axH.fill_between(days, daily(RH_out, np.min), daily(RH_out, np.max), color="#5577CC",
                     alpha=0.20, label="Plage journalière")
    axH.plot(days, daily(RH_out, np.mean), color="#334499", lw=2.0, label="Moyenne journalière")
    axH.set_ylim(0, 105); axH.set_ylabel("Humidité relative extérieure [%]", fontsize=11)
    axH.set_xlabel("Période simulée", fontsize=11)
    for ax in (axT, axH):
        set_date_ticks(ax); ax.legend(fontsize=10, loc="upper right", framealpha=0.92); ax.grid(True, alpha=0.18)
    save_fig(fig, "climate")


def plot_series():
    """Room air temperature and RH against the outdoor values, daily mean and range."""
    fig, (axT, axH) = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(TITLE + f"\nSurventilation nocturne {ACH_OVERVENT:g} vol/h  |  "
                 f"Hygroscopique : {'actif' if HYGRO_EFFECT else 'désactivé'}", fontsize=11, fontweight="bold")

    axT.fill_between(days, daily(T_out, np.min), daily(T_out, np.max), color="#DDDDDD",
                     alpha=0.55, label="Plage extérieure (min–max)")
    axT.fill_between(days, daily(T_room, np.min), daily(T_room, np.max), color="#E07020",
                     alpha=0.22, label="Plage pièce, air (min–max)")
    axT.plot(days, daily(T_out, np.mean),  color="#888888", lw=1.5, ls="--", label="T moy. extérieure")
    axT.plot(days, daily(T_room, np.mean), color="#CC5500", lw=2.2, label="T moy. de l'air de la pièce")
    if HEATING:
        axT.axhline(HEATING_SETPOINT, color="#1F5F99", lw=1.0, ls=":", alpha=0.75,
                    label=f"Consigne de chauffage {HEATING_SETPOINT:g} °C")
    axT.set_ylabel("Température [°C]", fontsize=11)
    axT.legend(fontsize=10, loc="upper left", framealpha=0.92)
    info_box(axT, f"T max pièce : {T_room.max():.1f} °C   T min pièce : {T_room.min():.1f} °C\n"
                  f"T max ext : {T_out.max():.1f} °C   DH = {DH_total:.0f} °C·h", 0.99, 0.96, "right", "top")

    axH.plot(days, daily(RH_out, np.mean),  color="#AAAAAA", lw=1.5, ls="--", label="HR moy. extérieure")
    axH.plot(days, daily(RH_room, np.mean), color="#334499", lw=2.2, label="HR moy. pièce")
    axH.axhline(30, color="#E07020", lw=0.8, ls=":", alpha=0.6, label="Limite basse confort 30 %")
    axH.axhline(70, color="#CC3030", lw=0.8, ls=":", alpha=0.6, label="Limite haute confort 70 %")
    axH.set_ylim(0, 105); axH.set_ylabel("Humidité relative [%]", fontsize=11)
    axH.set_xlabel("Période simulée", fontsize=11)
    axH.legend(fontsize=10, loc="upper left", framealpha=0.92)
    for ax in (axT, axH):
        set_date_ticks(ax); ax.grid(True, alpha=0.18)
    save_fig(fig, "series_T_RH", bbox_inches="tight")


def wall_fields(wall, i):
    """Vapour pressure [Pa] and heat flux [W/m²] (conduction + latent, + = outside → inside) at step i."""
    T, RH = wall.StockT[i], wall.StockRH[i]
    x  = wall.layer.x_pos
    Pv = lib.Pv(T, RH).flatten()
    q  = (-wall.layer.k(T, RH).flatten() * np.gradient(T.flatten(), x)
          - lib.Lv * wall.layer.delta_p(T, RH).flatten() * np.gradient(Pv, x))
    return Pv, q


def plot_wall_fields(day, kind, file_name):
    """Temperature, RH, vapour pressure and heat flux across every wall, at PROFILE_HOURS of one day."""
    colors = ["#1F5F99", "#E07020", "#CC3030", "#2E8B57", "#8E44AD"]
    names  = list(ORIENT)
    rows   = ["Température [°C]", "Humidité relative [%]", "Pression de vapeur Pv [Pa]",
              "Flux de chaleur q [W/m²]  (+ = ext → int)"]
    fig, axes = plt.subplots(len(rows), len(names), figsize=(4.6 * len(names), 14),
                             constrained_layout=True, sharex="col", squeeze=False)
    fig.suptitle(f"Champs dans les parois — {kind} : {fmt_date(day)}  "
                 f"(T ext max {T_out_max[day]:.1f} °C, min {T_out_min[day]:.1f} °C)", fontsize=12, fontweight="bold")
    for col, name in enumerate(names):
        wall = sim.walls[col]
        x_cm = wall.layer.x_pos * 100
        for h, c in zip(PROFILE_HOURS, colors):
            i = day * S + h
            Pv, q = wall_fields(wall, i)
            for r, y in enumerate((wall.StockT[i].flatten() - 273.15, wall.StockRH[i].flatten() * 100, Pv, q)):
                axes[r, col].plot(x_cm, y, color=c, lw=2.0)
        for r in range(len(rows)):
            ax = axes[r, col]
            for b in wall.layer.layer_bounds * 100:
                ax.axvline(b, color="#CCCCCC", lw=1.1, ls="--")
            ax.set_xlim(0, THICKNESS[name] * 100); ax.grid(True, alpha=0.18)
            if col == 0:
                ax.set_ylabel(rows[r], fontsize=10)
        axes[1, col].set_ylim(0, 105)
        axes[3, col].axhline(0, color="#888888", lw=0.8)
        for i, m in enumerate(wall.layer.mat):
            cx = (wall.layer.layer_bounds[i] + wall.layer.layer_bounds[i+1]) / 2 * 100
            axes[-1, col].text(cx, 0.02, m, transform=axes[-1, col].get_xaxis_transform(),
                               ha="center", va="bottom", fontsize=8, color="#777777")
        axes[0, col].set_title(f"Paroi {WALL_FR[name]}", fontsize=11, fontweight="bold")
        axes[-1, col].set_xlabel("Position [cm]  (ext → int)", fontsize=10)
    axes[0, 0].legend(handles=[Line2D([0], [0], color=c, lw=2.0, label=f"{h:02d}h") for h, c in zip(PROFILE_HOURS, colors)],
                      title="Heure", fontsize=9, title_fontsize=9, loc="best", framealpha=0.9)
    save_fig(fig, file_name)


def plot_comfort():
    """RE2020 comfort map: occupied hours against T_rm, with the limits of the DH indicator."""
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    x = np.linspace(T_rm_h.min() - 1, T_rm_h.max() + 1, 200)
    upper = hot_discomfort_limit(x, np.full(len(x), 12))
    ax.fill_between(x, HEATING_SETPOINT, upper, color="#B8E0B0", alpha=0.55, label="Zone de confort RE2020")
    ax.plot(x, upper, color="#CC3030", lw=1.8, label="Limite haute 6h-22h : min(max(26 ; 0,33·T_rm + 20,8) ; 28)")
    ax.axhline(26.0, color="#CC3030", lw=1.4, ls="--", label="Limite haute 22h-6h (sommeil) : 26 °C")
    ax.axhline(HEATING_SETPOINT, color="#1F5F99", lw=1.6,
               label=f"Limite basse = consigne de chauffage {HEATING_SETPOINT:g} °C")
    ax.scatter(T_rm_h[~occ_h], T_comf[~occ_h], c="#999999", s=3, alpha=0.10, label="Heures inoccupées (non comptées)")
    ax.scatter(T_rm_h[in_zone],  T_comf[in_zone],  c="#2E8B57", s=5, alpha=0.35, label=f"En confort  {pct_ok:.0f} %")
    ax.scatter(T_rm_h[too_hot],  T_comf[too_hot],  c="#CC3030", s=5, alpha=0.45, label=f"Trop chaud  {pct_hot:.0f} %")
    ax.scatter(T_rm_h[too_cold], T_comf[too_cold], c="#1F5F99", s=5, alpha=0.35, label=f"Trop froid  {pct_cold:.0f} %")
    ax.set_xlabel("T extérieure de référence T_rm [°C]  (moyenne glissante, α = 0.8)", fontsize=11)
    ax.set_ylabel(f"Température {COMF_FR} de la pièce [°C]", fontsize=11)
    ax.set_title(f"Carte de confort RE2020 — confort adaptatif, catégorie 1 — {DESCRIPTION}\n"
                 f"{LABEL}  |  {START_DATE} → {END_DATE}  |  Pièce {LX:g}×{LY:g} m", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.92, markerscale=3)
    ax.grid(True, alpha=0.20)
    info_box(ax, f"Heures occupées : {n_occ} h\n  En confort : {pct_ok:.0f} %\n  Trop chaud : {pct_hot:.0f} %\n"
                 f"  Trop froid : {pct_cold:.0f} %\n\nDH RE2020 = {DH_total:.0f} °C·h ({DH_total/DH_MAX*100:.0f} % du seuil)",
             0.98, 0.03, "right", "bottom")
    save_fig(fig, "comfort_map")


def plot_degree_hours():
    """Daily and cumulated summer discomfort; degree-days below 18 °C for long free-floating runs."""
    DH_day = daily(DH_hour, np.sum)
    DH_cum = np.cumsum(DH_day)
    with_dj = N_DAYS >= 150 and not HEATING
    fig, axes = plt.subplots(3 if with_dj else 2, 1, figsize=(15, 11 if with_dj else 8), constrained_layout=True)
    fig.suptitle(f"Indicateurs thermiques — {DESCRIPTION}  |  {LABEL}\n"
                 f"DH = {DH_total:.0f} °C·h ({DH_total/DH_MAX*100:.0f}% du seuil 1 250 °C·h)"
                 + (f"  |  DJ hiver = {DJ_total:.0f} °C·j" if with_dj else ""), fontsize=12, fontweight="bold")
    axes[0].bar(days, DH_day, color="#CC3030", alpha=0.85, width=0.9)
    axes[0].set_ylabel("DH journalier [°C·h]", fontsize=11)
    axes[0].set_title("Degrés-heures d'inconfort RE2020 (confort adaptatif cat. 1, heures occupées)"
                      if RE2020 else "Degrés-heures au-dessus de 28 °C", fontsize=10)
    axes[0].grid(True, alpha=0.18, axis="y")

    axes[1].plot(days, DH_cum, color="#CC3030", lw=2.2, label="DH cumulés")
    axes[1].axhline(DH_MAX, color="#333333", lw=1.5, ls="--", label="Seuil RE2020 = 1 250 °C·h/an")
    if (DH_cum >= DH_MAX).any():
        axes[1].fill_between(days, DH_cum, DH_MAX, where=(DH_cum >= DH_MAX), color="#CC3030",
                             alpha=0.15, label="Dépassement du seuil")
    axes[1].set_ylabel("DH cumulés [°C·h]", fontsize=11)
    axes[1].legend(fontsize=10, framealpha=0.92); axes[1].grid(True, alpha=0.18)

    if with_dj:
        DJ_day = daily(T_comf, lambda x: float(np.maximum(18.0 - x, 0.0).mean()))
        axes[2].fill_between(days, DJ_day, 0, color="#1F5F99", alpha=0.55, label="Degrés-jours sous 18 °C")
        axes[2].set_ylabel("Degrés-jours [°C·j]", fontsize=11)
        axes[2].set_title(f"Inconfort froid (base 18 °C) — total {DJ_total:.0f} °C·j", fontsize=10)
        axes[2].legend(fontsize=10, framealpha=0.92); axes[2].grid(True, alpha=0.18)
    for ax in axes:
        set_date_ticks(ax)
    axes[-1].set_xlabel("Période simulée", fontsize=11)
    save_fig(fig, "degree_hours")


def plot_heating():
    """Room temperature with the heating season shaded, and the daily heating need."""
    need_day = daily(Q_heat, np.sum) * DT / 3.6e6 / FLOOR_AREA          # [kWh/m²·day]
    season   = np.array(heating_ctrl.authorized_days(n_days))
    fig, (axT, axQ) = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(TITLE + f"\nChauffage RE2020 : consigne {HEATING_SETPOINT:g} °C (réduit {HEATING_SETBACK:g} °C)"
                 f"  |  saison ouverte {HEAT_DAYS} jours sur {n_days}", fontsize=11, fontweight="bold")

    # shade the periods with the season open (one span per run of consecutive open days)
    edges = np.diff(np.r_[0, season.astype(int), 0])
    for ax in (axT, axQ):
        for k0, k1 in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            ax.axvspan(k0 - 0.5, k1 - 0.5, color="#FFE9B0", alpha=0.45, lw=0)
    axT.fill_between(days, daily(T_room, np.min), daily(T_room, np.max), color="#E07020",
                     alpha=0.22, label="Plage pièce, air (min–max)")
    axT.plot(days, daily(T_room, np.mean), color="#CC5500", lw=2.2, label="T moy. de l'air de la pièce")
    axT.plot(days, daily(T_out, np.mean),  color="#888888", lw=1.4, ls="--", label="T moy. extérieure")
    axT.axhline(HEATING_SETPOINT, color="#CC3030", lw=1.0, ls=":", label=f"Consigne {HEATING_SETPOINT:g} °C")
    axT.axhline(HEATING_SETBACK,  color="#1F5F99", lw=1.0, ls=":", label=f"Réduit {HEATING_SETBACK:g} °C")
    axT.set_ylabel("Température [°C]", fontsize=11)
    axT.legend(fontsize=10, loc="upper right", framealpha=0.92)
    axT.text(0.01, 0.04, "Zone jaune = saison de chauffe ouverte", transform=axT.transAxes,
             fontsize=9, color="#8A6D00", va="bottom")

    axQ.bar(days, need_day, color="#1F5F99", alpha=0.85, width=0.9, label="Besoin journalier [kWh/m²]")
    axQ.set_ylabel("Besoin de chauffage [kWh/m²·jour]", fontsize=11)
    axQ.set_xlabel("Période simulée", fontsize=11)
    axQc = axQ.twinx()
    axQc.plot(days, np.cumsum(need_day), color="#CC3030", lw=2.2, label="Cumul [kWh/m²]")
    axQc.set_ylabel("Besoin cumulé [kWh/m²]", fontsize=11)
    axQ.legend(handles=axQ.get_legend_handles_labels()[0] + axQc.get_legend_handles_labels()[0],
               fontsize=10, loc="upper right", framealpha=0.92)
    info_box(axQ, f"Besoin total : {HEAT_NEED_KWH:.0f} kWh  =  {HEAT_NEED_KWH/FLOOR_AREA:.1f} kWh/m²\n"
                  f"Puissance de pointe : {Q_heat.max():.0f} W  (max {HEATING_MAX_W:g} W)\n"
                  f"Énergie finale (rendement {HEATING_EFFICIENCY:g}) : {HEAT_FINAL_KWH:.0f} kWh",
             0.01, 0.96, "left", "top")
    for ax in (axT, axQ):
        set_date_ticks(ax); ax.grid(True, alpha=0.18)
    save_fig(fig, "heating")


print("\n  Figures:")
plot_3d()
plot_climate()
plot_series()
plot_wall_fields(HOTTEST, "jour le + chaud", "wall_fields_hottest")
plot_wall_fields(COLDEST, "jour le + froid", "wall_fields_coldest")
plot_comfort()
plot_degree_hours()
if HEATING:
    plot_heating()


# ══════════════════════════════════════════════════════════════════════════════
# 9. SUMMARY AND SAVED RESULTS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 66)
print("  RESULTS")
print("-" * 66)
print(f"  Weather     : {LABEL}")
print(f"  Period      : {START_DATE} → {END_DATE}  ({N_DAYS} days)")
print(f"  Room        : {LX:g}×{LY:g} m  h={H:g} m  A={FLOOR_AREA:g} m²  V={VOLUME:g} m³")
for n in ORIENT:
    print(f"  Wall {n:5s}  : {describe_layers(WALL_LAYERS[n])}   U = {wall_u_value(WALL_LAYERS[n]):.3f} W/(m²·K)")
print(f"  Ventilation : {ACH_BASE:g} 1/h base + {ACH_OVERVENT:g} 1/h at night "
      f"(months {SUMMER_MONTHS[0]}–{SUMMER_MONTHS[1]})")
print(f"  Hygroscopic : {'ON' if HYGRO_EFFECT else 'OFF'}")
print(f"  Spin-up     : {SPINUP_DAYS} days" if SPINUP_DAYS else "  Spin-up     : none (walls start at the outdoor state)")
print(f"  T max out / room (air) : {T_out.max():.1f} / {T_room.max():.1f} °C")
print(f"  T min out / room (air) : {T_out.min():.1f} / {T_room.min():.1f} °C")
print(f"  Indoor RH   : mean {RH_room.mean():.0f} %  (min {RH_room.min():.0f}, max {RH_room.max():.0f}),  "
      f"daily amplitude {RH_amp:.1f} %,  >70 %: {pct_RH_hi:.1f} % of hours,  <30 %: {pct_RH_lo:.1f} %")
print(f"  Wall moisture exchange (latent): released {LAT_REL_KWH:.0f} kWh, absorbed {LAT_ABS_KWH:.0f} kWh")
if RE2020:
    print(f"  DH RE2020   : {DH_total:.0f} °C·h  ({DH_total/DH_MAX*100:.0f}% of the 1 250 °C·h threshold)"
          f"   [above 28 °C: {DH_28:.0f} °C·h]")
else:
    print(f"  DH above 28 °C : {DH_28:.0f} °C·h")
if HEATING:
    print(f"  Heating     : need {HEAT_NEED_KWH:.0f} kWh = {HEAT_NEED_KWH/FLOOR_AREA:.1f} kWh/m²,  "
          f"final energy {HEAT_FINAL_KWH:.0f} kWh,  peak {Q_heat.max():.0f} W,  season open {HEAT_DAYS}/{n_days} days")
else:
    print(f"  Degree-days below 18 °C : {DJ_total:.0f} °C·d")
print(f"  RE2020 comfort (occupied hours, {COMFORT_TEMP} temperature): "
      f"{pct_ok:.0f}% OK  |  {pct_hot:.0f}% too hot  |  {pct_cold:.0f}% too cold")

if RE2020:
    rep = re_ev.report()
    dh_note = "" if MODE == "Th-D" else "   (regulatory in Th-D mode only)"
    re_lines = [
        f"RE2020 indicators - zone {ZONE}, mode {MODE}, period {START_DATE} -> {END_DATE}, "
        f"reference area {FLOOR_AREA:g} m2",
        f"DH (adaptive comfort, category 1, occupied hours) : {rep['DH_adaptive']:.0f} degC.h "
        f"= {rep['DH_adaptive']/rep['DH_max']*100:.0f}% of the {rep['DH_max']:.0f} threshold{dh_note}",
        f"DH simplified (fixed 28 degC, all hours)          : {rep['DH_28C']:.0f} degC.h",
    ]
    if HEATING:
        re_lines += [
            f"Heating need B_ch                                 : {rep['B_heat']:.1f} kWh/(m2.yr)",
            f"Bbio, heating part (2 x B_ch)                     : {rep['BBio']:.1f} points",
            f"Primary energy, heating only (x2.3 electricity)   : {rep['ep_heat']:.1f} kWh_ep/(m2.yr)",
        ]
    re_lines.append("Partial: no cooling, lighting, hot-water or auxiliaries needs; comfort evaluated with the "
                    + ("operative temperature (air + mean radiant)/2." if COMFORT_TEMP == "operative"
                       else "room air temperature."))
    print("-" * 66)
    for line in re_lines:
        print("  " + line)
    with open(os.path.join(OUT_DIR, "re2020_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(re_lines) + "\n")

walls_used = {mat for layers in WALL_LAYERS.values() for mat, _ in layers}
summary = {
    "scenario": SCENARIO, "note": SCENARIO_NOTE, "run_id": RUN_ID, "folder": os.path.basename(OUT_DIR),
    "zone": ZONE, "mode": MODE, "start": START_DATE, "end": END_DATE, "days": N_DAYS,
    "walls": {n: describe_layers(WALL_LAYERS[n]) for n in ORIENT},
    "ach_base": ACH_BASE, "ach_overvent": ACH_OVERVENT, "heating": HEATING, "re2020": RE2020,
    "T_max_out": float(T_out.max()), "T_max_room": float(T_room.max()), "T_min_room": float(T_room.min()),
    "comfort_temperature": COMFORT_TEMP, "T_max_op": float(T_op.max()), "T_min_op": float(T_op.min()),
    "DH_RE2020": float(DH_total), "DH_28C": float(DH_28),
    "comfort_ok_pct": float(pct_ok), "comfort_hot_pct": float(pct_hot), "comfort_cold_pct": float(pct_cold),
    "heating_need_kWh_m2": float(HEAT_NEED_KWH / FLOOR_AREA) if HEATING else None,
    "RH_mean": float(RH_room.mean()), "RH_min": float(RH_room.min()), "RH_max": float(RH_room.max()),
    "RH_daily_amplitude": RH_amp, "pct_RH_above_70": pct_RH_hi, "pct_RH_below_30": pct_RH_lo,
    "latent_release_kWh": LAT_REL_KWH, "latent_absorb_kWh": LAT_ABS_KWH,
    "interior_Sd": dict(WALL_INTERIOR_SD), "hygro_effect": HYGRO_EFFECT, "spinup_days": SPINUP_DAYS,
    "heating_season": HEATING_SEASON if HEATING else None,
    "material_params": {m: v for m, v in MATERIALS_IN_FORCE.items() if m in walls_used},
}

# Every setting of the run, then the property values of the sourced materials
with open(os.path.join(OUT_DIR, "run_config.txt"), "w", encoding="utf-8") as f:
    f.write(f"# simulation.py  run {RUN_ID}  scenario = {SCENARIO}"
            + (f"  ({SCENARIO_NOTE})" if SCENARIO_NOTE else "") + "\n")
    for k in _CONFIG_NAMES:
        f.write(f"{k} = {globals()[k]!r}\n")
    for m, vals in summary["material_params"].items():
        f.write(f"# {m}: {vals!r}\n")

with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# Hourly room values; row k = hour k of the period
np.savetxt(os.path.join(OUT_DIR, "series.csv"),
           np.column_stack([np.arange(N) // S, hour_h, T_out, RH_out, T_room, RH_room,
                            Q_heat if HEATING else np.zeros(N), Q_lat, occ_h.astype(int),
                            T_rm_h, DH_hour, T_lim_h, adapt_h.astype(int), Q_int_h, G_int_h, ACH_h,
                            T_mrt, T_op, T_si]),
           delimiter=",", fmt="%.4f", comments="",
           header="day,hour,T_out,RH_out,T_room,RH_room,Q_heat_W,Q_lat_W,occupied,T_rm,DH_h,"
                  "T_lim,adaptive_period,Q_int_W,G_int_kg_h,ACH,T_mrt,T_op,"
                  + ",".join(f"Tsi_{n}" for n in WALL_NAMES))

# Temperature [°C] and RH [-] at every node of every wall, every hour (same rows as series.csv)
np.savez_compressed(
    os.path.join(OUT_DIR, "wall_fields.npz"), walls=np.array(list(ORIENT)),
    **{f"{n}_{key}": arr for n, w in zip(ORIENT, sim.walls) for key, arr in (
        ("T",      (np.array(w.StockT)[:N, :, 0] - 273.15).astype(np.float32)),
        ("RH",     np.array(w.StockRH)[:N, :, 0].astype(np.float32)),
        ("x",      w.layer.x_pos),
        ("bounds", w.layer.layer_bounds),
        ("mat",    np.array(w.layer.mat)))})

print("-" * 66)
print(f"  Folder: {os.path.relpath(OUT_DIR, DIR)}")
print("=" * 66)
