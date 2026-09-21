# -*- coding: utf-8 -*-
"""
simulation.py
=============
Hygrothermal simulation of a four-wall room (summer, full year or any period)
with multilayer walls and RE2020 weather by climate zone.

Everything that is normally changed lives in SECTION 1 (configuration).

Usage
-----
    python simulation.py                              # uses the configuration in section 1
    python simulation.py --zone H3                    # another climate zone
    python simulation.py --zone H3 --mode Th-BC       # another weather mode
    python simulation.py --start 01/01 --end 01/01    # full year
    python simulation.py --start 15/07 --end 31/08    # any period
    python simulation.py --mode Th-BC --start 01/01 --end 01/01 --heating on   # RE2020 heating
    python simulation.py --scenario rammed_earth      # a named variant from scenarios.py
    (several variants at once + comparison table: python run_scenarios.py --all)

Results: EVERY run gets its own folder (never overwritten):
    results/<date>_<time>_<scenario>_<zone>_<mode>_<start>-<end>/
with the PDFs (model_3d, series_T_RH, wall_profiles, wall_fields_hottest,
wall_fields_coldest, climate, comfort_map, degree_hours, + heating if HEATING),
run_config.txt (every parameter used), summary.json (key indicators) and, with
RE2020, re2020_report.txt.
(The text inside the graphics is in French.)
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
    locale.setlocale(locale.LC_TIME, "French_France.1252")   # French month names in the graphics
except Exception:
    pass

import library       as lib
from wall_config     import make_wall, WindowConfig, ThermalBridge, OpeningConfig
from wall_layer      import _MAT_REGISTRY
from sources         import (OccupantConfig, EquipmentConfig, LightingConfig,
                             VentilationConfig, HVACConfig, build_daily_schedule)
from solar           import SolarCalculator
from heating         import RE2020Heating, occupancy_series
from re2020          import RE2020Evaluator, adaptive_discomfort
from room_simulation import RoomSimulation
from weather         import load_weather_csv, load_re2020_weather


# ══════════════════════════════════════════════════════════════════════════════
# 1. CONFIGURATION  —  the only section you normally need to edit
# ══════════════════════════════════════════════════════════════════════════════

# ── Weather ───────────────────────────────────────────────────────────────────
ZONE = "H1c"        # H1a H1b H1c H2a H2b H2c H2d H3     |  "MACON" = prospective CSV (+2 °C)
MODE = "Th-D"       # "Th-D" = summer comfort (with heat wave) | "Th-BC" = typical year
                    # (the mode is ignored when ZONE = "MACON")

# ── Simulated period ──────────────────────────────────────────────────────────
# Format "day/month". END_DATE is EXCLUDED (the run stops when that day begins).
#   Summer    :  "21/06" → "21/09"
#   Full year :  "01/01" → "01/01"   (start = end means 365 days)
#   If END < START the period wraps over the new year (e.g. "01/11" → "01/03").
START_DATE = "01/01"
END_DATE   = "01/01"

# Spin-up: walls keep their moisture for months, so starting them at the outdoor state of the
# first hour distorts the results (indoor RH, latent exchange, even comfort). The run therefore
# first simulates SPINUP_DAYS days that END where the period starts (same weather file, cyclic
# year), and only the final state (walls, room air, heating season) is kept: nothing of the
# spin-up appears in the results. 365 = one full year before the period; 0 = no spin-up.
SPINUP_DAYS = 365

# ── Room geometry [m] ─────────────────────────────────────────────────────────
LX = 5.0            # interior length, East–West
LY = 5.0            # interior depth, South–North
H  = 2.5            # ceiling height

# ── Wall materials ────────────────────────────────────────────────────────────
# Each wall is a list of layers [(material, thickness_m), ...] from OUTSIDE to INSIDE.
# Available materials: Hempcrete, Rammed_Earth, Earth_Plaster, Lime_Plaster,
#   Gypsum_Plaster, BA13, Fermacell, Concrete, Wood, Wood_Fiber, Rock_Wool, Vapor_Barrier
# Examples:
#   WALL_STACK = [("Rammed_Earth", 0.30)]                            # rammed earth
#   WALL_STACK = [("Hempcrete", 0.30), ("Earth_Plaster", 0.02)]      # hempcrete + earth plaster
#   WALL_STACK = [("Wood_Fiber", 0.10), ("Rammed_Earth", 0.30)]      # exterior insulation + earth
WALL_STACK = [("Hempcrete", 0.30)]
#WALL_STACK1 = [("Rammed_Earth", 0.30), ("Earth_Plaster", 0.02), ("Vapor_Barrier", 0.02)]

# Keys: South, North, East, West. Give a wall its own list to make it different.
WALL_LAYERS = {"South": WALL_STACK, "North": WALL_STACK, "East": WALL_STACK, "West": WALL_STACK}

# ── Windows (orientation "S", "N", "E" or "W") ────────────────────────────────
# The opaque area of each wall is computed automatically: length × height − windows.
WINDOWS = [
    WindowConfig("South window", area=1.5, orientation="S", U_value=1.1, g_value=0.60, shading=0.30),
    WindowConfig("North window", area=0.5, orientation="N", U_value=1.1, g_value=0.60, shading=0.0),
]
THERMAL_BRIDGES = [ThermalBridge("Window frames", psi=0.04, length=6.0)]
INFILTRATION    = [OpeningConfig("Infiltration", ach_contribution=0.1)]   # [1/h]

# ── Ventilation ───────────────────────────────────────────────────────────────
# ACH_BASE     : hygiene ventilation, all day [1/h]
# ACH_OVERVENT : air changes ADDED to ACH_BASE during night overventilation [1/h].
#                For constant ventilation equal to ACH_BASE, set ACH_OVERVENT = 0
ACH_BASE      = 0.50
ACH_OVERVENT  = 4.0
NIGHT_HOURS   = list(range(0, 8)) + list(range(22, 24))    # 22h → 8h
SUMMER_MONTHS = (5, 9)     # May–September: night overventilation active and comfort evaluated

# ── Internal loads ────────────────────────────────────────────────────────────
N_OCCUPANTS     = 1.0
Q_SENSIBLE      = 80.0       # [W/person]
Q_LATENT        = 60.0       # [W/person]  (water vapour)
OCCUPANCY_HOURS = list(range(8, 22))
EQUIPMENT_W     = 100.0      # switched on together with the occupants
LIGHTING_W_M2   = 5.0
LIGHTING_HOURS  = list(range(20, 24))

# ── Heating (RE2020 / Th-BCE 2020 conventions for dwellings, see heating.py) ─
# HEATING = True adds an ideal heating system driven by the RE2020 scenario:
#   - setpoint 19 °C when occupied, 16 °C in reduced mode (weekday absences, Christmas week)
#   - heating season: allowed until early spring, forbidden from 2 July to 9 September,
#     restarted in autumn when the room gets cold
# Meant for MODE = "Th-BC" and periods that include the cold months (e.g. the full year).
# The heating need does not depend on the efficiency; the final energy does.
HEATING            = False
HEATING_SETPOINT   = 19.0      # [°C] occupied
HEATING_SETBACK    = 16.0      # [°C] reduced
HEATING_EFFICIENCY = 1.0       # COP / efficiency (1.0 = ideal electric heater)
HEATING_MAX_W      = 3000.0    # [W] maximum heating power

# ── RE2020 indicators (see re2020.py) ─────────────────────────────────────────
# RE2020 = True connects the RE2020 evaluator and reports, in the summary and in
# re2020_report.txt:
#   - DH, the official summer-comfort indicator (adaptive comfort, occupied hours only)
#   - the heating part of Bbio (2·B_ch) and of Cep, when HEATING = True
# Regulatory use of DH: MODE = "Th-D". Bbio and Cep are only PARTIAL here (no cooling,
# lighting, hot water or auxiliaries are modelled).
RE2020 = True

# ── Physics and numerics (normally left alone) ────────────────────────────────
HYGRO_EFFECT  = True       # False = walls impermeable to vapour (heat only)

# Interior coating / vapour film of a wall, as a resistance at the INTERIOR surface:
# equivalent air-layer thickness Sd [m] (wall name → Sd; missing or 0 = bare surface).
# Thermally negligible and it does not refine the mesh, unlike a Vapor_Barrier LAYER
# (a thin film there would force a very fine mesh; a 2 cm PE layer adds unreal thermal mass).
# Typical Sd: paint 0.05–0.5 m, PE film ≈ 100 m (Bui used 0.5 m for a conventional coating).
WALL_INTERIOR_SD = {}      # e.g. {"North": 100.0, "East": 100.0}

# Latent heat of the moisture exchanged between walls and room air:
# True  = also added to the room-air heat balance (as in the original model)
# False = walls exchange only sensible heat with the air (evaporation cools the wall surface)
# This choice changes how hygroscopic walls affect the room temperature: see the study notes.
LATENT_TO_AIR = True
H_EXT, H_INT  = 25.0, 8.0  # convective coefficients [W/(m²·K)]
MESH_SIZE     = 0.01       # mesh size inside the wall [m]
DT            = 3600.0     # time step [s] — schedules and weather are hourly: do not change

# ── Graphics ──────────────────────────────────────────────────────────────────
PROFILE_HOURS = [6, 14, 22]           # hours of the extreme days used for the in-wall profiles
PROFILE_WALLS = ["South", "North"]    # walls whose T(x) and RH(x) profiles are drawn
                                      # (the wall_fields figures always show the four walls)
COMFORT_MAP   = "RE2020"              # "RE2020": zone between the heating setpoint and the RE2020
                                      #   adaptive limit, occupied hours only (as the DH indicator)
                                      # "EN15251": EN 15251 category II band, summer months only


# ══════════════════════════════════════════════════════════════════════════════
# 2. COMMAND LINE (optional: overrides the configuration above)
# ══════════════════════════════════════════════════════════════════════════════

_ap = argparse.ArgumentParser(description="Hygrothermal simulation of a room.")
_ap.add_argument("--zone");  _ap.add_argument("--mode")
_ap.add_argument("--start"); _ap.add_argument("--end")
_ap.add_argument("--heating", choices=["on", "off"])
_ap.add_argument("--re2020",  choices=["on", "off"])
_ap.add_argument("--scenario", help="named variant defined in scenarios.py")
_ap.add_argument("--latent", choices=["on", "off"], help="LATENT_TO_AIR (see section 1)")
_ap.add_argument("--spinup", type=int, help="SPINUP_DAYS (see section 1); 0 = no spin-up")
_args, _ = _ap.parse_known_args()

# Scenario: overrides of the configuration above (command-line flags below still win)
SCENARIO, SCENARIO_NOTE = "base", ""
if _args.scenario:
    from scenarios import SCENARIOS
    if _args.scenario not in SCENARIOS:
        raise SystemExit(f"Unknown scenario '{_args.scenario}'. Available: {', '.join(SCENARIOS)}")
    SCENARIO = _args.scenario
    _ov = dict(SCENARIOS[SCENARIO])
    SCENARIO_NOTE = _ov.pop("note", "")
    _ov = {k: v for k, v in _ov.items() if k.isupper()}          # lower-case keys are documentation
    _bad = [k for k in _ov if k not in globals()]
    if _bad:
        raise SystemExit(f"Scenario '{SCENARIO}': unknown parameter(s) {_bad}. "
                         f"Use the upper-case variable names of simulation.py section 1.")
    if "WALL_STACK" in _ov:                                       # one stack for all four walls
        WALL_STACK  = _ov.pop("WALL_STACK")
        WALL_LAYERS = {n: WALL_STACK for n in WALL_LAYERS}
    if "WALL_LAYERS" in _ov:                                      # only the listed walls change
        WALL_LAYERS = {**WALL_LAYERS, **_ov.pop("WALL_LAYERS")}
    globals().update(_ov)

ZONE       = _args.zone  or ZONE
MODE       = _args.mode  or MODE
START_DATE = _args.start or START_DATE
END_DATE   = _args.end   or END_DATE
HEATING    = HEATING if _args.heating is None else (_args.heating == "on")
RE2020     = RE2020  if _args.re2020  is None else (_args.re2020  == "on")
LATENT_TO_AIR = LATENT_TO_AIR if _args.latent is None else (_args.latent == "on")
SPINUP_DAYS   = SPINUP_DAYS if _args.spinup is None else _args.spinup
ZONE = "MACON" if ZONE.upper() == "MACON" else ZONE[:2].upper() + ZONE[2:].lower()   # h1C → H1c

# Every parameter in force for this run (saved in run_config.txt)
_CONFIG_NAMES = [k for k in list(globals()) if k.isupper() and k not in ("DIR", "SCENARIOS")]


# ══════════════════════════════════════════════════════════════════════════════
# 3. DERIVED QUANTITIES (calendar, geometry, material descriptions)
# ══════════════════════════════════════════════════════════════════════════════

def _doy(txt):
    """'21/06' → day of the year (non-leap year: the weather has 8760 h)."""
    d, m = (int(x) for x in txt.strip().split("/"))
    return date(2001, m, d).timetuple().tm_yday

DOY_START = _doy(START_DATE)
DOY_END   = _doy(END_DATE)
N_DAYS    = (DOY_END - DOY_START) % 365 or 365

def day_date(k, start_doy=None):
    """Calendar date of day k of the run (0 = first day), or of a run starting at start_doy."""
    first = DOY_START if start_doy is None else start_doy
    return date(2001, 1, 1) + timedelta(days=(first - 1 + k) % 365)

FLOOR_AREA    = LX * LY
VOLUME        = LX * LY * H
INTERNAL_MASS = 110e3 * FLOOR_AREA          # [J/K] ISO 13790, medium inertia
HM_EXT        = 25e-9 if HYGRO_EFFECT else 1e-30
HM_INT        = HM_EXT * (H_INT / H_EXT)

ORIENT    = {"South": "S", "North": "N", "East": "E", "West": "W"}
WIDTH     = {"South": LX, "North": LX, "East": LY, "West": LY}
THICKNESS = {n: sum(e for _, e in WALL_LAYERS[n]) for n in ORIENT}
WALL_FR   = {"South": "Sud", "North": "Nord", "East": "Est", "West": "Ouest"}   # graphics are in French

def describe_layers(layers):
    return " + ".join(f"{m} {e*100:g} cm" for m, e in layers)

def wall_u_value(layers):
    """U [W/(m²·K)] with surface resistances, library conductivity at 20 °C / 50 % RH."""
    R = 1.0 / H_INT + 1.0 / H_EXT + sum(e / _MAT_REGISTRY[m].k(293.15, 0.5) for m, e in layers)
    return 1.0 / R

DESCRIPTION = (describe_layers(WALL_LAYERS["South"])
               if all(WALL_LAYERS[n] == WALL_LAYERS["South"] for n in ORIENT)
               else "parois mixtes")


# ══════════════════════════════════════════════════════════════════════════════
# 4. WEATHER
# ══════════════════════════════════════════════════════════════════════════════

if ZONE == "MACON":
    wx = load_weather_csv(os.path.join(DIR, "donnees-climatiques-prospectives-france-2c_macon.csv"))
    LABEL, TAG = "Mâcon prospectif +2 °C", "MACON"
else:
    wx = load_re2020_weather(
        os.path.join(DIR, "scenarios_meteorologiques_th-bc_th-d_re2020.xlsx"), ZONE, MODE)
    LABEL, TAG = f"RE2020 {ZONE} ({MODE})", f"{ZONE}_{MODE}"

# One new folder per run: date_time_scenario_zone_mode_period (never overwritten)
RESULTS_ROOT = os.environ.get("SIM_RESULTS_DIR") or os.path.join(DIR, "results")
RUN_ID  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = os.path.join(RESULTS_ROOT, f"{RUN_ID}_{SCENARIO}_{TAG}_"
                       f"{START_DATE.replace('/', '')}-{END_DATE.replace('/', '')}")
_base_dir, _k = OUT_DIR, 1
while os.path.exists(OUT_DIR):                    # two runs in the same second
    _k += 1
    OUT_DIR = f"{_base_dir}_{_k}"
os.makedirs(OUT_DIR)

N_YEAR    = len(wx.T_ext)
idx       = ((DOY_START - 1) * 24 + np.arange(N_DAYS * 24)) % N_YEAR
time_bc   = np.arange(N_DAYS * 24) * DT
T_ext_bc  = wx.T_ext[idx]
RH_ext_bc = wx.RH_ext[idx]
n_steps   = len(time_bc) - 1

print(f"Weather : {LABEL}   T {wx.T_ext.min():.1f}..{wx.T_ext.max():.1f} °C")
print(f"Period  : {START_DATE} → {END_DATE}  ({N_DAYS} days, {n_steps} steps)")
for n in ORIENT:
    print(f"Wall {n:5s}: {describe_layers(WALL_LAYERS[n])}   U = {wall_u_value(WALL_LAYERS[n]):.3f} W/(m²·K)")
print(f"Scenario: {SCENARIO}" + (f"  ({SCENARIO_NOTE})" if SCENARIO_NOTE else ""))
print(f"Output  : {os.path.relpath(OUT_DIR, DIR)}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. BUILDING AND SCHEDULES
# ══════════════════════════════════════════════════════════════════════════════

def _is_summer(month):
    a, b = SUMMER_MONTHS
    return (month >= a) & (month <= b) if a <= b else (month >= a) | (month <= b)


def create_walls(quiet=False):
    walls = []
    for name, ori in ORIENT.items():
        layers = WALL_LAYERS[name]
        area   = WIDTH[name] * H - sum(w.area for w in WINDOWS if w.orientation == ori)
        # a thin layer needs a finer mesh (at least 2 nodes per layer)
        mesh = min(MESH_SIZE, min(e for _, e in layers) / 2)
        if mesh < MESH_SIZE and not quiet:
            print(f"  [!] Thin layer in {name}: mesh reduced to {mesh*1000:g} mm (slower run)")
        # interior vapour resistance in series with the surface transfer coefficient:
        #   1/hm_eff = 1/hm_int + Sd/delta_air
        sd     = WALL_INTERIOR_SD.get(name, 0.0)
        hm_int = 1.0 / (1.0 / HM_INT + sd / lib.Da(293.15)) if sd > 0 else HM_INT
        walls.append(make_wall(name, area=area, orientation=ori, layers=layers,
                               mesh_size=mesh, h_ext=H_EXT, hm_ext=HM_EXT,
                               h_int=H_INT, hm_int=hm_int))
    return walls


# Running-mean outdoor temperature T_rm (EN 15251, α = 0.8), computed over the whole
# weather year with one year of spin-up. Used for the adaptive comfort and the heating.
alpha = 0.8
T_day_year = wx.T_ext[:(N_YEAR // 24) * 24].reshape(-1, 24).mean(axis=1)
seq = np.concatenate([T_day_year, T_day_year])
r, rm_seq = seq[0], np.zeros(len(seq))
for d in range(1, len(seq)):
    r = (1 - alpha) * seq[d - 1] + alpha * r
    rm_seq[d] = r
rm_year = rm_seq[len(T_day_year):]

# RE2020 Th-D: heating is never allowed during the adaptive-comfort period, from the first
# to the last day of the year with T_rm above 16 °C
adaptive = None
if HEATING and MODE == "Th-D" and ZONE != "MACON":
    warm = np.where(rm_year > 16.0)[0]
    adaptive = np.zeros(len(rm_year), dtype=bool)
    if len(warm):
        adaptive[warm[0]:warm[-1] + 1] = True


RE2020_ZONE = ZONE if ZONE != "MACON" else "H1c"       # Mâcon is assumed to be in zone H1c


def build_simulation(start_doy, n_steps, T_room0, RH_room0, evaluate=True, quiet=False):
    """
    Room simulation of n_steps hourly steps starting at 00:00 of day-of-year start_doy.
    Returns (simulation, heating controller or None, RE2020 evaluator or None).
    It is called twice: for the spin-up run and for the period itself.
    """
    occ_sched   = build_daily_schedule(OCCUPANCY_HOURS, n_steps, int(DT))
    light_sched = build_daily_schedule(LIGHTING_HOURS,  n_steps, int(DT))

    # Ventilation: night overventilation only during the summer months
    ach_total   = ACH_BASE + ACH_OVERVENT
    base_ratio  = ACH_BASE / ach_total if ach_total > 0 else 1.0
    step        = np.arange(n_steps)
    step_month  = np.array([day_date(int(k), start_doy).month for k in step // 24])
    is_night    = np.isin(step % 24, NIGHT_HOURS)
    vent_sched  = np.where(is_night & _is_summer(step_month), 1.0, base_ratio).tolist()
    ventilation = VentilationConfig(n_ach=ach_total, hrv_efficiency=0.0,
                                    moisture_recovery=0.0, schedule=vent_sched)

    # Heating: RE2020 scenario and season (see heating.py); no heating = free-floating room
    hvac, ctrl = None, None
    if HEATING:
        ctrl = RE2020Heating(start_doy, FLOOR_AREA, HEATING_SETPOINT, HEATING_SETBACK, adaptive)
        hvac = HVACConfig(T_heat_set=HEATING_SETPOINT, T_cool_set=99.0,        # no cooling
                          efficiency_heat=HEATING_EFFICIENCY, max_power_heat=HEATING_MAX_W,
                          heating_control=ctrl)

    # RE2020 evaluator: RoomSimulation feeds it the heating need / energy during the run
    ev = None
    if RE2020 and evaluate:
        ev = RE2020Evaluator(floor_area=FLOOR_AREA, climate_zone=RE2020_ZONE)
        ev.set_carrier(heating="electricity", cooling="electricity")

    sim = RoomSimulation(
        wall_configs   = create_walls(quiet),
        window_configs = WINDOWS,
        bridge_configs = THERMAL_BRIDGES,
        opening_configs= INFILTRATION,
        occupants  = OccupantConfig(N_OCCUPANTS, Q_SENSIBLE, Q_LATENT, occ_sched),
        equipment  = EquipmentConfig(EQUIPMENT_W, schedule=occ_sched),
        lighting   = LightingConfig(LIGHTING_W_M2, FLOOR_AREA, schedule=light_sched),
        ventilation= ventilation,
        hvac       = hvac,
        solar_calc = SolarCalculator(wx.lat, wx.lon, start_doy=start_doy,
                                     cloud_factor=0.30, ground_albedo=0.20,
                                     timezone_offset=1.0, weather=wx),
        volume        = VOLUME,
        internal_mass = INTERNAL_MASS,
        T_room_init   = T_room0,
        RH_room_init  = RH_room0,
        re2020        = ev,
        latent_to_air = LATENT_TO_AIR,
    )
    return sim, ctrl, ev


# ══════════════════════════════════════════════════════════════════════════════
# 6. SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

print(f"\nRunning…  night overventilation {ACH_OVERVENT:g} 1/h "
      f"(months {SUMMER_MONTHS[0]}–{SUMMER_MONTHS[1]})   hygro={'ON' if HYGRO_EFFECT else 'OFF'}   "
      f"heating={'RE2020' if HEATING else 'OFF'}")

# Spin-up: SPINUP_DAYS days ending at 00:00 of the first day of the period (weather is cyclic)
warm_sim = warm_ctrl = None
if SPINUP_DAYS > 0:
    doy_w = (DOY_START - 1 - SPINUP_DAYS) % 365 + 1
    n_w   = SPINUP_DAYS * 24
    idx_w = ((doy_w - 1) * 24 + np.arange(n_w + 1)) % N_YEAR
    print(f"Spin-up : {SPINUP_DAYS} days before {START_DATE} (results discarded, final state kept)")
    warm_sim, warm_ctrl, _ = build_simulation(doy_w, n_w, float(wx.T_ext[idx_w[0]]),
                                              float(wx.RH_ext[idx_w[0]]), evaluate=False, quiet=True)
    warm_sim.run(time_bc=np.arange(n_w + 1) * DT, Text_bc=wx.T_ext[idx_w], RHext_bc=wx.RH_ext[idx_w],
                 dt=DT, verbose=True)
    T_init, RH_init = warm_sim.T_room, warm_sim.RH_room
else:
    T_init, RH_init = float(T_ext_bc[0]), float(RH_ext_bc[0])

sim, heating_ctrl, re_ev = build_simulation(DOY_START, n_steps, T_init, RH_init)
if warm_sim is not None:
    sim.load_state(warm_sim)
    if heating_ctrl is not None:
        heating_ctrl.continue_from(warm_ctrl)
    del warm_sim, warm_ctrl

T_run, RH_run = sim.run(time_bc=time_bc, Text_bc=T_ext_bc, RHext_bc=RH_ext_bc, dt=DT, verbose=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7. RESULTS AND INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

S      = 24                                        # steps per day
n_days = len(T_run) // S
N      = n_days * S
T_room  = np.array(T_run)[:N]
RH_room = np.array(RH_run)[:N] * 100
T_out   = np.interp(np.arange(len(T_run)) * DT, time_bc, T_ext_bc)[:N]
RH_out  = np.interp(np.arange(len(T_run)) * DT, time_bc, RH_ext_bc)[:N] * 100

def daily(x, f):
    return np.array([f(x[k*S:(k+1)*S]) for k in range(n_days)])

T_out_max = daily(T_out, np.max)
T_out_min = daily(T_out, np.min)
HOTTEST   = int(np.argmax(T_out_max))              # hottest day (index)
COLDEST   = int(np.argmin(T_out_min))              # coldest day (index)
fmt_date  = lambda k: day_date(k).strftime("%d %B").lstrip("0")

DH_hour  = np.maximum(T_room - 28.0, 0.0)                       # simplified DH: fixed 28 °C, all hours
DH_28    = float(DH_hour.sum())                                 # [°C·h]
DH_total = DH_28                                                # replaced by the official DH below when RE2020 is on
DJ_total = float(np.maximum(18.0 - T_room, 0.0).sum() / 24.0)   # [°C·d] below 18 °C

# T_rm for the simulated days (rm_year is computed in section 5)
T_rm_day = np.array([rm_year[(DOY_START - 1 + k) % len(rm_year)] for k in range(n_days)])
T_rm_h   = np.repeat(T_rm_day, S)

# RE2020 occupancy scenario for dwellings (hourly, 1 = occupied)
occ_h = occupancy_series(DOY_START, N)

# Official RE2020 summer comfort (DH): adaptive comfort, occupied hours only.
# The room air temperature stands in for the operative temperature.
if RE2020:
    DH_hour  = adaptive_discomfort(T_room, T_rm_h, occ_h, T_cool_set=26.0, category=1)
    re_ev.add_adaptive_discomfort_hours(T_room, T_rm_h, occ_h, T_cool_set=26.0, category=1, dt=int(DT))
    DH_total = float(DH_hour.sum())

# Comfort classification used by the comfort map
month_day = np.array([day_date(k).month for k in range(n_days)])
summer_h  = np.repeat(_is_summer(month_day), S)
if COMFORT_MAP == "RE2020":
    # RE2020 comfort zone: from the heating setpoint up to the adaptive limit (category 1),
    # evaluated over the occupied hours of the simulated period (like the DH indicator)
    T_low, T_high = np.full(N, HEATING_SETPOINT), np.maximum(26.0, 0.33 * T_rm_h + 18.8 + 2.0)
    ref_h         = occ_h
    COMFORT_LABEL = "RE2020 comfort (occupied hours)"
else:
    # EN 15251 category II band (±3 K around 0.33·T_rm + 18.8), summer months only
    T_low, T_high = 0.33 * T_rm_h + 18.8 - 3.0, 0.33 * T_rm_h + 18.8 + 3.0
    ref_h         = summer_h
    COMFORT_LABEL = "EN 15251 cat. II comfort (summer months)"
in_zone   = ref_h & (T_room >= T_low) & (T_room <= T_high)
too_hot   = ref_h & (T_room >  T_high)
too_cold  = ref_h & (T_room <  T_low)
n_ref     = int(ref_h.sum())
pct_ok    = 100 * in_zone.sum()  / n_ref if n_ref else 0.0
pct_hot   = 100 * too_hot.sum()  / n_ref if n_ref else 0.0
pct_cold  = 100 * too_cold.sum() / n_ref if n_ref else 0.0

# Hygric indicators: indoor humidity and moisture exchange with the walls
RH_amp      = float(np.mean(daily(RH_room, np.max) - daily(RH_room, np.min)))   # mean daily amplitude [% RH]
pct_RH_hi   = float(100 * np.mean(RH_room > 70.0))
pct_RH_lo   = float(100 * np.mean(RH_room < 30.0))
Q_lat       = np.array(sim.StockQ_lat_total)[:N]        # [W] latent flux wall → room (+ = walls release moisture)
LAT_REL_KWH = float(Q_lat[Q_lat > 0].sum() * DT / 3.6e6)
LAT_ABS_KWH = float(-Q_lat[Q_lat < 0].sum() * DT / 3.6e6)

print(f"\n  Hottest day : {fmt_date(HOTTEST)}  T_out = {T_out_max[HOTTEST]:.1f} °C")
print(f"  Coldest day : {fmt_date(COLDEST)}  T_out = {T_out_min[COLDEST]:.1f} °C")
print(f"  T max out = {T_out.max():.1f} °C    T max room = {T_room.max():.1f} °C")
print(f"  T min out = {T_out.min():.1f} °C    T min room = {T_room.min():.1f} °C")
print(f"  Indoor RH : mean {RH_room.mean():.0f} %  (min {RH_room.min():.0f}, max {RH_room.max():.0f})   "
      f"mean daily amplitude {RH_amp:.1f} %   >70 %: {pct_RH_hi:.1f} % of hours   <30 %: {pct_RH_lo:.1f} %")
print(f"  Moisture exchange with the walls (latent): released {LAT_REL_KWH:.0f} kWh, absorbed {LAT_ABS_KWH:.0f} kWh")
if RE2020:
    print(f"  DH RE2020 = {DH_total:.0f} °C·h  ({DH_total/1250*100:.0f}% of the 1 250 °C·h threshold)"
          f"   [simplified DH, fixed 28 °C: {DH_28:.0f} °C·h]")
else:
    print(f"  DH simplified (fixed 28 °C) = {DH_28:.0f} °C·h  ({DH_28/1250*100:.0f}% of 1 250 °C·h)")
if HEATING:
    # hourly heating power [W] (sim.StockQ_HVAC: initial 0 + one value per step)
    Q_heat         = np.maximum(np.array(sim.StockQ_HVAC)[:N], 0.0)
    HEAT_NEED_KWH  = float(Q_heat.sum() * DT / 3.6e6)          # need at the emitter
    HEAT_FINAL_KWH = float(sim.E_heat_kWh)                     # final energy = need / efficiency
    HEAT_DAYS      = int(sum(heating_ctrl.authorized_days(n_days)))
    print(f"  Heating need = {HEAT_NEED_KWH:.0f} kWh  ({HEAT_NEED_KWH/FLOOR_AREA:.1f} kWh/m²)   "
          f"peak {Q_heat.max():.0f} W   season open {HEAT_DAYS} of {n_days} days")
else:
    print(f"  DJ winter = {DJ_total:.0f} °C·d (base 18 °C)")


# ══════════════════════════════════════════════════════════════════════════════
# 8. GRAPHICS  (all text inside the figures is in French)
# ══════════════════════════════════════════════════════════════════════════════

days = np.arange(n_days)

def date_ticks():
    pos, lab = [], []
    if n_days >= 75:
        for k in range(n_days):
            f = day_date(k)
            if f.day == 1:
                pos.append(k); lab.append(f.strftime("%b" if n_days > 200 else "%B"))
    else:
        for k in range(0, n_days, max(1, n_days // 8)):
            pos.append(k); lab.append(day_date(k).strftime("%d/%m"))
    return pos, lab

TICK_POS, TICK_LAB = date_ticks()

def set_date_ticks(ax):
    ax.set_xticks(TICK_POS); ax.set_xticklabels(TICK_LAB, fontsize=10)

def save_fig(fig, name, **kw):
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

TITLE = ((f"[{SCENARIO}] " if SCENARIO != "base" else "") +
         f"{DESCRIPTION} — Pièce {LX:g}×{LY:g} m  |  {LABEL}  |  "
         f"{START_DATE} → {END_DATE} ({N_DAYS} jours)")


# ── 8.1 3D model ──────────────────────────────────────────────────────────────
def plot_3d():
    print("\n  Plotting: model_3d ...")
    tS, tN, tE, tW = (THICKNESS[n] for n in ("South", "North", "East", "West"))

    def faces(x0, x1, y0, y1, z0, z1):
        return [
            [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],
            [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],
            [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],
            [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],
            [[x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]],
            [[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[x1,y0,z1]],
        ]

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_axes([0.25, 0.0, 0.75, 0.93], projection="3d")   # keep the left column free for the legend
    ax.set_facecolor("#F8F8F8")

    # South/North span the full width including the corners; East/West fill between them
    boxes = [
        ("South", -tW, LX+tE, -tS, 0,     "#D4956A", (LX/2, -tS/2,     H*0.55)),
        ("North", -tW, LX+tE,  LY, LY+tN, "#7BA7C4", (LX/2, LY+tN/2,   H*0.55)),
        ("West",  -tW, 0,      0,  LY,    "#A8C888", (-tW/2, LY/2,     H*0.55)),
        ("East",   LX, LX+tE,  0,  LY,    "#C8A8C8", (LX+tE/2, LY/2,   H*0.55)),
    ]
    for name, x0, x1, y0, y1, col, lpos in boxes:
        ax.add_collection3d(Poly3DCollection(faces(x0, x1, y0, y1, 0, H), alpha=0.78,
                            facecolor=col, edgecolor="#444444", linewidth=0.5))
        ax.text(*lpos, WALL_FR[name], ha="center", va="center", fontsize=11, fontweight="bold",
                color="#222222", zorder=1000,     # high zorder: otherwise hidden behind the walls
                bbox=dict(boxstyle="round,pad=0.15", fc=col, ec="#444444",
                          alpha=0.85, linewidth=0.8))

    ax.add_collection3d(Poly3DCollection([[[0,0,0],[LX,0,0],[LX,LY,0],[0,LY,0]]], alpha=0.30,
                        facecolor="#E8D8B0", edgecolor="#999999", linewidth=0.6))

    # Windows (one per orientation, with the total window area of that orientation)
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
        if   ori == "S": a, b = (c0, -tS + 0.01),     (c0 + ww, -tS + 0.01)
        elif ori == "N": a, b = (c0, LY + tN - 0.01), (c0 + ww, LY + tN - 0.01)
        elif ori == "W": a, b = (-tW + 0.01, c0),     (-tW + 0.01, c0 + ww)
        else:            a, b = (LX + tE - 0.01, c0), (LX + tE - 0.01, c0 + ww)
        rect = [[*a, z0], [*b, z0], [*b, z0 + hw], [*a, z0 + hw]]
        ax.add_collection3d(Poly3DCollection([rect], alpha=0.55, facecolor="#AED6F1",
                            edgecolor="#2980B9", linewidth=1.2))
        ax.text((a[0]+b[0])/2, (a[1]+b[1])/2, z0+hw+0.07, f"Vitrage {ori}\n{A:g} m²",
                ha="center", va="bottom", fontsize=7.5, color="#1A5276", zorder=1000)

    ax.quiver(LX/2, LY+tN+0.25, H+0.05, 0, 0.35, 0, color="#CC3030", lw=2.5, arrow_length_ratio=0.45)
    ax.text(LX/2, LY+tN+0.65, H+0.05, "N", ha="center", va="bottom", fontsize=13,
            fontweight="bold", color="#CC3030", zorder=1000)

    ax.plot([0, LX], [-tS-0.18]*2, [0, 0], color="#555555", lw=1.2)
    ax.text(LX/2, -tS-0.22, 0, f"{LX:g} m", ha="center", va="top", fontsize=9, color="#555555")
    ax.plot([LX+tE+0.12]*2, [0, LY], [0, 0], color="#555555", lw=1.2)
    ax.text(LX+tE+0.16, LY/2, 0, f"{LY:g} m", ha="left", va="center", fontsize=9,
            color="#555555", rotation=90)
    ax.plot([LX+tE+0.12]*2, [LY+tN+0.12]*2, [0, H], color="#555555", lw=1.2)
    ax.text(LX+tE+0.16, LY+tN+0.15, H/2, f"h = {H:g} m", ha="left", va="center",
            fontsize=9, color="#555555")

    lines = ["Parois (ext → int)", "─" * 26]
    for n in ORIENT:
        lines.append(f"{WALL_FR[n]}   U = {wall_u_value(WALL_LAYERS[n]):.3f} W/(m²·K)")
        lines += [f"   {m} {e*100:g} cm" for m, e in WALL_LAYERS[n]]
    fig.text(0.01, 0.95, "\n".join(lines), fontsize=9, va="top", ha="left", family="monospace",
             bbox=dict(boxstyle="round,pad=0.55", fc="#FFFDF0", ec="#C8A86A", alpha=0.95))

    ax.set_xlim(-tW-0.3, LX+tE+0.5); ax.set_ylim(-tS-0.3, LY+tN+0.8); ax.set_zlim(0, H+0.35)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([]); ax.grid(False)
    ax.set_title(f"Pièce {LX:g}×{LY:g} m  h = {H:g} m\n"
                 f"Surface : {FLOOR_AREA:g} m²    Volume : {VOLUME:g} m³",
                 fontsize=12, fontweight="bold", pad=10)
    ax.view_init(elev=24, azim=-48)
    save_fig(fig, "model_3d", bbox_inches="tight")


# ── 8.2 Temperature and RH series ─────────────────────────────────────────────
def plot_series():
    print("  Plotting: series_T_RH ...")
    fig, (axT, axH) = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(TITLE + f"\nSurventilation nocturne {ACH_OVERVENT:g} vol/h  |  "
                 f"Hygroscopique : {'actif' if HYGRO_EFFECT else 'désactivé'}",
                 fontsize=11, fontweight="bold")

    axT.fill_between(days, daily(T_out, np.min), daily(T_out, np.max), color="#DDDDDD",
                     alpha=0.55, label="Plage extérieure (min–max)")
    axT.fill_between(days, daily(T_room, np.min), daily(T_room, np.max), color="#E07020",
                     alpha=0.22, label="Plage pièce (min–max)")
    axT.plot(days, daily(T_out, np.mean),  color="#888888", lw=1.5, ls="--", label="T moy. extérieure")
    axT.plot(days, daily(T_room, np.mean), color="#CC5500", lw=2.2, label="T moy. pièce")
    axT.axhline(28, color="#CC3030", lw=1.0, ls=":", alpha=0.75, label="Seuil RE2020 = 28 °C")
    axT.axhline(20, color="#1F5F99", lw=1.0, ls=":", alpha=0.75, label="Seuil chauffage = 20 °C")
    axT.set_ylabel("Température [°C]", fontsize=11)
    axT.set_xlim(0, max(n_days - 1, 1)); set_date_ticks(axT)
    axT.legend(fontsize=10, loc="upper left", framealpha=0.92); axT.grid(True, alpha=0.18)
    axT.text(0.99, 0.96,
             f"T max pièce : {T_room.max():.1f} °C   T min pièce : {T_room.min():.1f} °C\n"
             f"T max ext : {T_out.max():.1f} °C   DH = {DH_total:.0f} °C·h",
             transform=axT.transAxes, fontsize=9.5, ha="right", va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#BBBBBB", alpha=0.92))

    axH.plot(days, daily(RH_out, np.mean),  color="#AAAAAA", lw=1.5, ls="--", label="HR moy. extérieure")
    axH.plot(days, daily(RH_room, np.mean), color="#334499", lw=2.2, label="HR moy. pièce")
    axH.axhline(30, color="#E07020", lw=0.8, ls=":", alpha=0.6, label="Limite basse confort 30 %")
    axH.axhline(70, color="#CC3030", lw=0.8, ls=":", alpha=0.6, label="Limite haute confort 70 %")
    axH.set_ylim(0, 105); axH.set_ylabel("Humidité relative [%]", fontsize=11)
    axH.set_xlabel("Période simulée", fontsize=11)
    axH.set_xlim(0, max(n_days - 1, 1)); set_date_ticks(axH)
    axH.legend(fontsize=10, loc="upper left", framealpha=0.92); axH.grid(True, alpha=0.18)
    save_fig(fig, "series_T_RH", bbox_inches="tight")


# ── 8.3 In-wall profiles (hottest and coldest day) ────────────────────────────
def plot_profiles():
    print("  Plotting: wall_profiles ...")
    hot_colors  = ["#1F5F99", "#E07020", "#CC3030", "#2E8B57", "#8E44AD"]
    cold_colors = ["#6699CC", "#FF9944", "#EE6655", "#66BB88", "#BB88DD"]
    names = list(ORIENT)
    fig, axes = plt.subplots(2, len(PROFILE_WALLS), figsize=(7.5 * len(PROFILE_WALLS), 10),
                             constrained_layout=True, squeeze=False)
    fig.suptitle(f"Profils de température et d'HR dans les parois\n"
                 f"Jour le + chaud : {fmt_date(HOTTEST)}  |  Jour le + froid : {fmt_date(COLDEST)}  |  "
                 f"{DESCRIPTION}", fontsize=11, fontweight="bold")

    for col, name in enumerate(PROFILE_WALLS):
        wall = sim.walls[names.index(name)]
        x_cm = wall.layer.x_pos * 100
        thick_cm = THICKNESS[name] * 100
        axT, axH = axes[0, col], axes[1, col]
        for h, c_hot, c_cold in zip(PROFILE_HOURS, hot_colors, cold_colors):
            i_hot, i_cold = HOTTEST * S + h, COLDEST * S + h
            axT.plot(x_cm, wall.StockT[i_hot].flatten() - 273.15,  color=c_hot,  lw=2.5)
            axT.plot(x_cm, wall.StockT[i_cold].flatten() - 273.15, color=c_cold, lw=2.0, ls="--")
            axH.plot(x_cm, wall.StockRH[i_hot].flatten() * 100,    color=c_hot,  lw=2.5)
            axH.plot(x_cm, wall.StockRH[i_cold].flatten() * 100,   color=c_cold, lw=2.0, ls="--")
        for ax in (axT, axH):
            for b in wall.layer.layer_bounds * 100:
                ax.axvline(b, color="#CCCCCC", lw=1.2, ls="--")
            for i, m in enumerate(wall.layer.mat):
                cx = (wall.layer.layer_bounds[i] + wall.layer.layer_bounds[i+1]) / 2 * 100
                ax.text(cx, 0.02, m, transform=ax.get_xaxis_transform(), ha="center",
                        va="bottom", fontsize=8, color="#777777")
            ax.set_xlim(0, thick_cm); ax.grid(True, alpha=0.18)
        legend = ([Line2D([0], [0], color=c, lw=2.5, label=f"{h:02d}h  {fmt_date(HOTTEST)}")
                   for h, c in zip(PROFILE_HOURS, hot_colors)] +
                  [Line2D([0], [0], color=c, lw=2.0, ls="--", label=f"{h:02d}h  {fmt_date(COLDEST)}")
                   for h, c in zip(PROFILE_HOURS, cold_colors)])
        axT.legend(handles=legend, fontsize=8.5, loc="best", framealpha=0.9, ncol=2)
        axT.set_title(f"Paroi {WALL_FR[name]} — Température [°C]", fontsize=11, fontweight="bold")
        axT.set_ylabel("T [°C]", fontsize=11); axT.tick_params(labelbottom=False)
        axH.set_title(f"Paroi {WALL_FR[name]} — Humidité relative [%]", fontsize=11, fontweight="bold")
        axH.set_ylabel("HR [%]", fontsize=11); axH.set_ylim(0, 105)
        axH.set_xlabel("Position dans la paroi [cm]   (ext → int)", fontsize=10)
        y_top = axT.get_ylim()[1]
        axT.text(0.8, y_top, "EXT", fontsize=9, color="#888888", va="top")
        axT.text(thick_cm - 0.8, y_top, "INT", fontsize=9, color="#888888", va="top", ha="right")
    save_fig(fig, "wall_profiles")


# ── 8.4 Weather data ──────────────────────────────────────────────────────────
def plot_climate():
    print("  Plotting: climate ...")
    fig, (axT, axH) = plt.subplots(2, 1, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(f"Données climatiques — {LABEL}  |  {START_DATE} → {END_DATE} ({N_DAYS} jours)",
                 fontsize=12, fontweight="bold")
    T_mn, T_mx = daily(T_out, np.min), daily(T_out, np.max)
    axT.fill_between(days, T_mn, T_mx, color="#E07020", alpha=0.20,
                     label=f"Plage journalière  amplitude moy. {(T_mx - T_mn).mean():.1f} °C")
    axT.plot(days, daily(T_out, np.mean), color="#CC5500", lw=2.0, label="Moyenne journalière")
    axT.axhline(28, color="#CC3030", lw=1.0, ls=":", alpha=0.7, label="Seuil DH = 28 °C")
    axT.set_ylabel("Température extérieure [°C]", fontsize=11)
    axT.set_xlim(0, max(n_days - 1, 1)); set_date_ticks(axT)
    axT.legend(fontsize=10, loc="upper right", framealpha=0.92); axT.grid(True, alpha=0.18)

    axH.fill_between(days, daily(RH_out, np.min), daily(RH_out, np.max), color="#5577CC",
                     alpha=0.20, label="Plage journalière")
    axH.plot(days, daily(RH_out, np.mean), color="#334499", lw=2.0, label="Moyenne journalière")
    axH.set_ylim(0, 105); axH.set_ylabel("Humidité relative extérieure [%]", fontsize=11)
    axH.set_xlabel("Période simulée", fontsize=11)
    axH.set_xlim(0, max(n_days - 1, 1)); set_date_ticks(axH)
    axH.legend(fontsize=10, loc="upper right", framealpha=0.92); axH.grid(True, alpha=0.18)
    save_fig(fig, "climate")


# ── 8.5 Adaptive comfort (EN 15251) ───────────────────────────────────────────
def plot_comfort_en15251():
    print("  Plotting: comfort_map (EN 15251) ...")
    cold_off_season = ~summer_h & (T_room < 20.0)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    # the adaptive model is only defined for 10 °C <= T_rm <= 30 °C
    x = np.linspace(max(10.0, T_rm_h.min() - 1), min(30.0, T_rm_h.max() + 1), 200)
    ax.fill_between(x, 0.33*x + 18.8 - 3, 0.33*x + 18.8 + 3, color="#B8E0B0",
                    alpha=0.55, label="Zone de confort EN 15251 Cat. II (±3 °C), valable 10 ≤ T_rm ≤ 30 °C")
    ax.plot(x, 0.33*x + 18.8, color="#2E8B57", lw=1.5, ls="--", label="T confort optimale")
    ax.scatter(T_rm_h[in_zone],  T_room[in_zone],  c="#2E8B57", s=5, alpha=0.35,
               label=f"Été — en confort  {pct_ok:.0f} %")
    ax.scatter(T_rm_h[too_hot],  T_room[too_hot],  c="#CC3030", s=5, alpha=0.45,
               label=f"Été — trop chaud  {pct_hot:.0f} %")
    ax.scatter(T_rm_h[too_cold], T_room[too_cold], c="#1F5F99", s=5, alpha=0.35,
               label=f"Été — trop froid  {pct_cold:.0f} %")
    if cold_off_season.any():
        ax.scatter(T_rm_h[cold_off_season], T_room[cold_off_season], c="#AACCEE", s=3, alpha=0.20,
                   label="Hors été — sous 20 °C")
    ax.set_xlabel("T extérieure de référence T_rm [°C]  (α = 0.8, EN 15251)", fontsize=11)
    ax.set_ylabel("Température de la pièce [°C]", fontsize=11)
    ax.set_title(f"Confort adaptatif EN 15251 Cat. II — {DESCRIPTION}\n"
                 f"{LABEL}  |  {START_DATE} → {END_DATE}  |  Pièce {LX:g}×{LY:g} m",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.92, markerscale=3)
    ax.grid(True, alpha=0.20)
    ax.text(0.98, 0.03,
            f"Résultats été (mois {SUMMER_MONTHS[0]}–{SUMMER_MONTHS[1]}) :\n"
            f"  En confort : {pct_ok:.0f} %\n  Trop chaud : {pct_hot:.0f} %\n"
            f"  Trop froid : {pct_cold:.0f} %\n\n"
            f"DH RE2020 = {DH_total:.0f} °C·h" + ("" if HEATING else f"\nDJ hiver  = {DJ_total:.0f} °C·j"),
            transform=ax.transAxes, fontsize=9.5, va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#BBBBBB", alpha=0.92))
    save_fig(fig, "comfort_map")


def plot_comfort_re2020():
    """Comfort map with the RE2020 definition (same limits as the DH indicator)."""
    print("  Plotting: comfort_map (RE2020) ...")
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    x = np.linspace(T_rm_h.min() - 1, T_rm_h.max() + 1, 200)
    upper = np.maximum(26.0, 0.33 * x + 18.8 + 2.0)
    ax.fill_between(x, HEATING_SETPOINT, upper, color="#B8E0B0", alpha=0.55,
                    label="Zone de confort RE2020")
    ax.plot(x, upper, color="#CC3030", lw=1.8,
            label="Limite haute (cat. 1) : max(26 ; 0,33·T_rm + 20,8)")
    ax.axhline(HEATING_SETPOINT, color="#1F5F99", lw=1.6,
               label=f"Limite basse = consigne de chauffage {HEATING_SETPOINT:g} °C")
    ax.scatter(T_rm_h[~occ_h], T_room[~occ_h], c="#999999", s=3, alpha=0.10,
               label="Heures inoccupées (non comptées)")
    ax.scatter(T_rm_h[in_zone],  T_room[in_zone],  c="#2E8B57", s=5, alpha=0.35,
               label=f"En confort  {pct_ok:.0f} %")
    ax.scatter(T_rm_h[too_hot],  T_room[too_hot],  c="#CC3030", s=5, alpha=0.45,
               label=f"Trop chaud  {pct_hot:.0f} %")
    ax.scatter(T_rm_h[too_cold], T_room[too_cold], c="#1F5F99", s=5, alpha=0.35,
               label=f"Trop froid  {pct_cold:.0f} %")
    ax.set_xlabel("T extérieure de référence T_rm [°C]  (moyenne glissante, α = 0.8)", fontsize=11)
    ax.set_ylabel("Température de la pièce [°C]", fontsize=11)
    ax.set_title(f"Carte de confort RE2020 — confort adaptatif, catégorie 1 — {DESCRIPTION}\n"
                 f"{LABEL}  |  {START_DATE} → {END_DATE}  |  Pièce {LX:g}×{LY:g} m",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left", framealpha=0.92, markerscale=3)
    ax.grid(True, alpha=0.20)
    ax.text(0.98, 0.03,
            f"Heures occupées : {n_ref} h\n"
            f"  En confort : {pct_ok:.0f} %\n  Trop chaud : {pct_hot:.0f} %\n"
            f"  Trop froid : {pct_cold:.0f} %\n\n"
            f"DH RE2020 = {DH_total:.0f} °C·h ({DH_total/1250*100:.0f} % du seuil)",
            transform=ax.transAxes, fontsize=9.5, va="bottom", ha="right", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#BBBBBB", alpha=0.92))
    save_fig(fig, "comfort_map")


def plot_comfort():
    (plot_comfort_re2020 if COMFORT_MAP == "RE2020" else plot_comfort_en15251)()


# ── 8.5b Fields inside the four walls (hottest and coldest day) ───────────────
def wall_fields(wall, i):
    """Vapour pressure [Pa], capillary pressure [Pa] and heat flux [W/m²] at every node, step i.
    Heat flux = conduction + latent heat carried by the vapour, positive from exterior to interior."""
    lay = wall.layer
    T, RH = wall.StockT[i], wall.StockRH[i]
    x  = lay.x_pos
    Pv = lib.Pv(T, RH).flatten()
    Pc = wall.StockPc[i].flatten()
    k  = lay.k(T, RH).flatten()
    dp = lay.delta_p(T, RH).flatten()
    q  = -k * np.gradient(T.flatten(), x) - lib.Lv * dp * np.gradient(Pv, x)
    return Pv, Pc, q


def plot_wall_fields(day, kind, file_name):
    print(f"  Plotting: {file_name} ...")
    colors = ["#1F5F99", "#E07020", "#CC3030", "#2E8B57", "#8E44AD", "#7F7F7F"]
    names  = list(ORIENT)
    fig, axes = plt.subplots(3, 4, figsize=(20, 11.5), constrained_layout=True, sharex="col")
    fig.suptitle(f"Champs dans les 4 parois — {kind} : {fmt_date(day)}  "
                 f"(T ext max {T_out_max[day]:.1f} °C, min {T_out_min[day]:.1f} °C)  |  {DESCRIPTION}",
                 fontsize=12, fontweight="bold")
    rows = [("Pression de vapeur  Pv [Pa]", 1.0), ("Pression capillaire  Pc [MPa]", 1e6),
            ("Flux de chaleur  q [W/m²]  (+ = ext → int)", None)]
    for col, name in enumerate(names):
        wall = sim.walls[col]
        x_cm = wall.layer.x_pos * 100
        for h, c in zip(PROFILE_HOURS, colors):
            Pv, Pc, q = wall_fields(wall, day * S + h)
            axes[0, col].plot(x_cm, Pv,       color=c, lw=2.2)
            axes[1, col].plot(x_cm, Pc / 1e6, color=c, lw=2.2)
            axes[2, col].plot(x_cm, q,        color=c, lw=2.2)
        for r, (ylabel, _) in enumerate(rows):
            ax = axes[r, col]
            for b in wall.layer.layer_bounds * 100:
                ax.axvline(b, color="#CCCCCC", lw=1.1, ls="--")
            ax.set_xlim(0, THICKNESS[name] * 100); ax.grid(True, alpha=0.18)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10)
        axes[2, col].axhline(0, color="#888888", lw=0.8)
        for i, m in enumerate(wall.layer.mat):
            cx = (wall.layer.layer_bounds[i] + wall.layer.layer_bounds[i+1]) / 2 * 100
            axes[2, col].text(cx, 0.02, m, transform=axes[2, col].get_xaxis_transform(),
                              ha="center", va="bottom", fontsize=8, color="#777777")
        axes[0, col].set_title(f"Paroi {WALL_FR[name]}   ({describe_layers(WALL_LAYERS[name])})",
                               fontsize=10, fontweight="bold")
        axes[2, col].set_xlabel("Position dans la paroi [cm]  (ext → int)", fontsize=10)
    axes[0, 0].legend(handles=[Line2D([0], [0], color=c, lw=2.2, label=f"{h:02d}h")
                               for h, c in zip(PROFILE_HOURS, colors)],
                      title="Heure", fontsize=9, title_fontsize=9, loc="best", framealpha=0.9)
    save_fig(fig, file_name)


# ── 8.6 Degree-hours (and heating degree-days for long periods) ───────────────
def plot_degree_hours():
    print("  Plotting: degree_hours ...")
    DH_day = daily(DH_hour, np.sum)
    DJ_day = daily(T_room, lambda x: float(np.maximum(18.0 - x, 0.0).mean()))
    DH_cum = np.cumsum(DH_day)
    with_heating = N_DAYS >= 150 and not HEATING     # degree-days panel: meaningless when the room is heated
    rows = 3 if with_heating else 2
    fig, axes = plt.subplots(rows, 1, figsize=(15, 11 if with_heating else 8), constrained_layout=True)
    fig.suptitle(f"Indicateurs thermiques — {DESCRIPTION}  |  {LABEL}\n"
                 f"DH = {DH_total:.0f} °C·h ({DH_total/1250*100:.0f}% du seuil 1 250 °C·h)"
                 + (f"  |  DJ hiver = {DJ_total:.0f} °C·j" if with_heating else ""),
                 fontsize=12, fontweight="bold")
    axes[0].bar(days, DH_day, color=["#CC3030" if v > 0 else "#AED6F1" for v in DH_day],
                alpha=0.85, width=0.9)
    axes[0].set_ylabel("DH journalier [°C·h]", fontsize=11)
    axes[0].set_title("Degrés-heures d'inconfort RE2020 (confort adaptatif cat. 1, heures occupées)  (rouge = inconfort)"
                      if RE2020 else "Degrés-heures d'inconfort > 28 °C  (rouge = inconfort)", fontsize=10)
    axes[0].grid(True, alpha=0.18, axis="y")

    axes[1].plot(days, DH_cum, color="#CC3030", lw=2.2, label="DH cumulés")
    axes[1].axhline(1250, color="#333333", lw=1.5, ls="--", label="Seuil RE2020 = 1 250 °C·h/an")
    axes[1].fill_between(days, DH_cum, 1250, where=(DH_cum >= 1250), color="#CC3030",
                         alpha=0.15, label="Dépassement du seuil")
    axes[1].set_ylabel("DH cumulés [°C·h]", fontsize=11)
    axes[1].legend(fontsize=10, framealpha=0.92); axes[1].grid(True, alpha=0.18)

    if with_heating:
        axes[2].fill_between(days, DJ_day, 0, color="#1F5F99", alpha=0.55,
                             label="Besoin chauffage journalier [°C·j]")
        axes[2].set_ylabel("Besoin chauffage [°C·j]", fontsize=11)
        axes[2].set_title(f"Besoins de chauffage journaliers (base 18 °C) — total {DJ_total:.0f} °C·j",
                          fontsize=10)
        axes[2].legend(fontsize=10, framealpha=0.92); axes[2].grid(True, alpha=0.18)
    for ax in axes:
        ax.set_xlim(-0.5, n_days - 0.5); set_date_ticks(ax)
    axes[-1].set_xlabel("Période simulée", fontsize=11)
    save_fig(fig, "degree_hours")


# ── 8.7 Heating (only when HEATING = True) ────────────────────────────────────
def plot_heating():
    print("  Plotting: heating ...")
    need_day = daily(Q_heat, np.sum) * DT / 3.6e6 / FLOOR_AREA          # [kWh/m²] per day
    season   = np.array(heating_ctrl.authorized_days(n_days))
    fig, (axT, axQ) = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(TITLE + f"\nChauffage RE2020 : consigne {HEATING_SETPOINT:g} °C "
                 f"(réduit {HEATING_SETBACK:g} °C)  |  saison ouverte {HEAT_DAYS} jours sur {n_days}",
                 fontsize=11, fontweight="bold")

    def shade(ax):
        k = 0
        while k < n_days:
            if season[k]:
                j = k
                while j + 1 < n_days and season[j + 1]:
                    j += 1
                ax.axvspan(k - 0.5, j + 0.5, color="#FFE9B0", alpha=0.45, lw=0)
                k = j + 1
            else:
                k += 1

    shade(axT); shade(axQ)
    axT.fill_between(days, daily(T_room, np.min), daily(T_room, np.max), color="#E07020",
                     alpha=0.22, label="Plage pièce (min–max)")
    axT.plot(days, daily(T_room, np.mean), color="#CC5500", lw=2.2, label="T moy. pièce")
    axT.plot(days, daily(T_out, np.mean),  color="#888888", lw=1.4, ls="--", label="T moy. extérieure")
    axT.axhline(HEATING_SETPOINT, color="#CC3030", lw=1.0, ls=":", label=f"Consigne {HEATING_SETPOINT:g} °C")
    axT.axhline(HEATING_SETBACK,  color="#1F5F99", lw=1.0, ls=":", label=f"Réduit {HEATING_SETBACK:g} °C")
    axT.set_ylabel("Température [°C]", fontsize=11)
    axT.set_xlim(-0.5, n_days - 0.5); set_date_ticks(axT)
    axT.legend(fontsize=10, loc="upper right", framealpha=0.92); axT.grid(True, alpha=0.18)
    axT.text(0.01, 0.04, "Zone jaune = saison de chauffe ouverte", transform=axT.transAxes,
             fontsize=9, color="#8A6D00", va="bottom")

    axQ.bar(days, need_day, color="#1F5F99", alpha=0.85, width=0.9, label="Besoin journalier [kWh/m²]")
    axQ.set_ylabel("Besoin de chauffage [kWh/m²·jour]", fontsize=11)
    axQ.set_xlim(-0.5, n_days - 0.5); set_date_ticks(axQ)
    axQ.set_xlabel("Période simulée", fontsize=11); axQ.grid(True, alpha=0.18, axis="y")
    axQc = axQ.twinx()
    axQc.plot(days, np.cumsum(need_day), color="#CC3030", lw=2.2, label="Cumul [kWh/m²]")
    axQc.set_ylabel("Besoin cumulé [kWh/m²]", fontsize=11)
    lines = axQ.get_legend_handles_labels()[0] + axQc.get_legend_handles_labels()[0]
    axQ.legend(handles=lines, fontsize=10, loc="upper right", framealpha=0.92)
    axQ.text(0.01, 0.96,
             f"Besoin total : {HEAT_NEED_KWH:.0f} kWh  =  {HEAT_NEED_KWH/FLOOR_AREA:.1f} kWh/m²\n"
             f"Puissance de pointe : {Q_heat.max():.0f} W\n"
             f"Énergie finale (rendement {HEATING_EFFICIENCY:g}) : {HEAT_FINAL_KWH:.0f} kWh",
             transform=axQ.transAxes, fontsize=9.5, va="top", family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#BBBBBB", alpha=0.92))
    save_fig(fig, "heating")


plot_3d()
plot_series()
plot_profiles()
plot_climate()
plot_comfort()
plot_wall_fields(HOTTEST, "jour le + chaud", "wall_fields_hottest")
plot_wall_fields(COLDEST, "jour le + froid", "wall_fields_coldest")
plot_degree_hours()
if HEATING:
    plot_heating()


# ══════════════════════════════════════════════════════════════════════════════
# 9. SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 66)
print("  RESULTS")
print("-" * 66)
print(f"  Weather   : {LABEL}")
print(f"  Period    : {START_DATE} → {END_DATE}  ({N_DAYS} days)")
print(f"  Room      : {LX:g}×{LY:g} m  h={H:g} m  A={FLOOR_AREA:g} m²  V={VOLUME:g} m³")
for n in ORIENT:
    print(f"  Wall {n:5s}: {describe_layers(WALL_LAYERS[n])}   U = {wall_u_value(WALL_LAYERS[n]):.3f} W/(m²·K)")
print(f"  Ventilation: {ACH_BASE:g} 1/h base + {ACH_OVERVENT:g} 1/h at night "
      f"(months {SUMMER_MONTHS[0]}–{SUMMER_MONTHS[1]})")
print(f"  Hygroscopic: {'ON' if HYGRO_EFFECT else 'OFF'}")
print(f"  Spin-up   : {SPINUP_DAYS} days" if SPINUP_DAYS else "  Spin-up   : none (walls start at the outdoor state)")
print(f"  T max out / room : {T_out.max():.1f} / {T_room.max():.1f} °C")
print(f"  T min out / room : {T_out.min():.1f} / {T_room.min():.1f} °C")
if RE2020:
    print(f"  DH RE2020  : {DH_total:.0f} °C·h  ({DH_total/1250*100:.0f}% of the 1 250 °C·h threshold)")
else:
    print(f"  DH simplified (fixed 28 °C) : {DH_28:.0f} °C·h  ({DH_28/1250*100:.0f}% of 1 250 °C·h)")
if HEATING:
    print(f"  Heating (RE2020): need {HEAT_NEED_KWH:.0f} kWh = {HEAT_NEED_KWH/FLOOR_AREA:.1f} kWh/m²,  "
          f"final energy {HEAT_FINAL_KWH:.0f} kWh,  peak {Q_heat.max():.0f} W,  "
          f"season open {HEAT_DAYS}/{n_days} days")
else:
    print(f"  DJ winter  : {DJ_total:.0f} °C·d")
print(f"  {COMFORT_LABEL}: {pct_ok:.0f}% OK  |  {pct_hot:.0f}% too hot  |  {pct_cold:.0f}% too cold")

if RE2020:
    rep = re_ev.report()
    dh_note = "" if (MODE == "Th-D" and ZONE != "MACON") else "   (regulatory in Th-D mode only)"
    re_lines = [
        f"RE2020 indicators - zone {RE2020_ZONE}, mode {'CSV' if ZONE == 'MACON' else MODE}, "
        f"period {START_DATE} -> {END_DATE}, reference area {FLOOR_AREA:g} m2",
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
    re_lines.append("Partial: no cooling, lighting, hot-water or auxiliaries needs; room air temperature "
                    "used as operative temperature.")
    print("-" * 66)
    for line in re_lines:
        print("  " + line)
    with open(os.path.join(OUT_DIR, "re2020_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(re_lines) + "\n")

# Every parameter used and the key indicators, saved next to the PDFs
with open(os.path.join(OUT_DIR, "run_config.txt"), "w", encoding="utf-8") as f:
    f.write(f"# simulation.py  run {RUN_ID}  scenario = {SCENARIO}"
            + (f"  ({SCENARIO_NOTE})" if SCENARIO_NOTE else "") + "\n")
    for k in _CONFIG_NAMES:
        f.write(f"{k} = {globals()[k]!r}\n")

summary = {
    "scenario": SCENARIO, "note": SCENARIO_NOTE, "run_id": RUN_ID,
    "folder": os.path.basename(OUT_DIR),
    "zone": ZONE, "mode": MODE, "start": START_DATE, "end": END_DATE, "days": N_DAYS,
    "walls": {n: describe_layers(WALL_LAYERS[n]) for n in ORIENT},
    "ach_base": ACH_BASE, "ach_overvent": ACH_OVERVENT, "heating": HEATING, "re2020": RE2020,
    "T_max_out": float(T_out.max()), "T_max_room": float(T_room.max()),
    "T_min_room": float(T_room.min()),
    "DH_RE2020": float(DH_total), "DH_28C": float(DH_28),
    "comfort_model": COMFORT_MAP, "comfort_ok_pct": float(pct_ok),
    "comfort_hot_pct": float(pct_hot), "comfort_cold_pct": float(pct_cold),
    "heating_need_kWh_m2": float(HEAT_NEED_KWH / FLOOR_AREA) if HEATING else None,
    "RH_mean": float(RH_room.mean()), "RH_min": float(RH_room.min()), "RH_max": float(RH_room.max()),
    "RH_daily_amplitude": RH_amp, "pct_RH_above_70": pct_RH_hi, "pct_RH_below_30": pct_RH_lo,
    "latent_release_kWh": LAT_REL_KWH, "latent_absorb_kWh": LAT_ABS_KWH,
    "interior_Sd": dict(WALL_INTERIOR_SD), "hygro_effect": HYGRO_EFFECT,
    "latent_to_air": LATENT_TO_AIR, "spinup_days": SPINUP_DAYS,
}
# hourly series (for comparing runs or plotting elsewhere)
np.savetxt(os.path.join(OUT_DIR, "series.csv"),
           np.column_stack([np.arange(N) // S, np.arange(N) % S, T_out, RH_out, T_room, RH_room,
                            Q_heat if HEATING else np.zeros(N), Q_lat, occ_h.astype(int)]),
           delimiter=",", fmt="%.4f", comments="",
           header="day,hour,T_out,RH_out,T_room,RH_room,Q_heat_W,Q_lat_W,occupied")
with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("-" * 66)
print(f"  Folder: {os.path.relpath(OUT_DIR, DIR)}")
print("=" * 66)
