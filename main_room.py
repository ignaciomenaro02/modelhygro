# -*- coding: utf-8 -*-
"""
main_room.py
============
User configuration entry point for the multi-wall room simulation.

HOW TO USE
----------
1. Edit Section 1  → building geometry (walls, windows, thermal bridges).
2. Edit Section 2  → occupants, equipment, ventilation, HVAC.
3. Edit Section 3  → simulation settings (climate file, time step, duration).
4. Edit Section 4  → RE2020 parameters (floor area, climate zone).
5. Run the file.   → Sections 5–7 run automatically (simulation + outputs).

ADDING A NEW WALL  (easy way — make_wall)
-----------------------------------------
    make_wall(
        "East wall", area=15.0, orientation='E',
        layers=[                       # exterior -> interior
            ("Rock_Wool",      0.12),
            ("Concrete",       0.20),
            ("Gypsum_Plaster", 0.01),
        ],
        Mesh_Opt=0,   # 0=uniform, 1=refined   (optional)
        liq=0,        # 0=vapour only, 1=+liquid (optional)
    )

AVAILABLE MATERIALS (wall_layer._MAT_REGISTRY)
----------------------------------------------
    Rammed_Earth    Hempcrete       Rock_Wool     Wood_Fiber
    Concrete        Wood            Vapor_Barrier Earth_Plaster
    Gypsum_Plaster  Lime_Plaster    Fermacell     BA13

ORIENTATION CODES
-----------------
    'N' 'S' 'E' 'W'  'NE' 'NW' 'SE' 'SW'  'roof'  'floor'

IMPORTING FROM SKETCHUP / PLEIADES
-----------------------------------
    Use import_geometry() below to load walls/windows from a JSON file
    exported from SketchUp or Pleiades.  See the function docstring.
"""

import os
import sys
import numpy as np
import pandas as pd

data = os.path.dirname(os.path.abspath(__file__))
os.chdir(data)
if data not in sys.path:
    sys.path.insert(0, data)

# Print UTF-8 safely on any console (Windows cp1252 would otherwise crash on the
# "°", "─", "✅" characters used in the summaries).
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from wall_config      import (WallConfig, WindowConfig, ThermalBridge,
                              OpeningConfig, make_wall)
from sources          import OccupantConfig, EquipmentConfig, LightingConfig, \
                             VentilationConfig, HVACConfig, build_daily_schedule
from solar            import SolarCalculator
from re2020           import RE2020Evaluator
from room_simulation  import RoomSimulation
from outputs          import (plot_room_climate, plot_hvac_energy,
                               plot_wall_profiles, plot_re2020_comparison,
                               plot_comfort_map, table_energy_balance,
                               compare_scenarios, export_csv, plot_walls_3d)
from weather          import load_weather_csv


###############################################################################
# ─────────────────────────  0. Climate data  ─────────────────────────────────
###############################################################################

# ── Simulation time step ─────────────────────────────────────────────────────
DT = 600          # [s]  → 600 s = 10 min.  The room state is updated every DT.

# ── Outdoor climate: REAL annual weather file ────────────────────────────────
# The room is driven by a real hourly weather file for Mâcon (prospective
# +2 °C climate, full year = 8760 h). The simulation interpolates between
# hourly points down to the DT (10-min) step automatically.
WEATHER_FILE = "donnees-climatiques-prospectives-france-2c_macon.csv"

weather  = load_weather_csv(WEATHER_FILE)
time_bc  = weather.time_s              # [s]
Text_bc  = weather.T_ext               # [°C]
RHext_bc = weather.RH_ext              # [-]
t_tot    = weather.t_tot               # [s]  (one full year)

print(f"Weather: {weather.station}  {weather.n_hours} h  "
      f"(T {Text_bc.min():.1f}…{Text_bc.max():.1f} °C)")

# ── Quick-test option ────────────────────────────────────────────────────────
# Full year by default. For a fast check, simulate only the first N days by
# setting the SIM_DAYS environment variable, e.g. (Git Bash):  SIM_DAYS=7 python main_room.py
SIM_DAYS = int(os.environ.get("SIM_DAYS", "0")) or None
if SIM_DAYS is not None:
    keep     = SIM_DAYS * 24 + 1
    time_bc  = time_bc[:keep]
    Text_bc  = Text_bc[:keep]
    RHext_bc = RHext_bc[:keep]
    t_tot    = int(time_bc[-1])
    print(f"   (quick test: first {SIM_DAYS} days only)")

# ── OLD outdoor source: laboratory wall-test data "B30" (kept for reuse) ─────
# Before, the "outdoor" boundary was the *measured chamber conditions* of the
# B30 wall experiment (column B30_Ta_l_Avg = the left climatic chamber).
# That is meant for validating ONE wall against a lab test — not for a real
# room. To go back to it, comment the weather block above and uncomment this:
#
# SHEET    = "B30"
# clim_df  = pd.read_excel("boundary_data.xlsx", sheet_name=SHEET)
# clim     = clim_df.values
# time_bc  = clim[:, 0]                # [s]
# Text_bc  = clim[:, 1]                # [°C]  (B30_Ta_l_Avg — left chamber)
# RHext_bc = clim[:, 3] / 100.0        # [-]   (B30_RHa_l)
# t_tot    = int(time_bc.max())


###############################################################################
# ─────────────────────────  1. Building geometry  ────────────────────────────
###############################################################################
# Each WallConfig = one opaque wall.
# Layers are ordered exterior → interior.
# Units: thicknesses in [m], area in [m²].

# Each wall is built with make_wall(): give it a name, area, orientation and a
# list of (material, thickness_in_metres) layers ordered exterior -> interior.
walls = [

    make_wall("South wall — Hempcrete + Rammed Earth", area=18.0, orientation='S',
              layers=[("Hempcrete",     0.20),    # exterior
                      ("Rammed_Earth",  0.30)],   # interior
              mesh_size=0.01),

    make_wall("North wall — Concrete + Rock Wool + BA13", area=18.0, orientation='N',
              layers=[("Concrete",   0.20),
                      ("Rock_Wool",  0.12),
                      ("BA13",       0.013)]),

    make_wall("East wall — Wood + Wood Fiber + Fermacell", area=12.0, orientation='E',
              layers=[("Wood",        0.04),
                      ("Wood_Fiber",  0.12),
                      ("Fermacell",   0.015)]),

    make_wall("West wall — Rammed Earth + Earth Plaster", area=12.0, orientation='W',
              layers=[("Rammed_Earth",  0.30),
                      ("Earth_Plaster", 0.02)]),

]

# ── Windows & glazing ──────────────────────────────────────────────────────

windows = [

    WindowConfig(
        name        = "South glazing",
        area        = 8.0,
        orientation = 'S',
        U_value     = 1.1,      # [W/(m²K)] — double glazing
        g_value     = 0.60,     # SHGC [-]
        shading     = 0.20,     # 20% shading (slight overhang)
    ),

    WindowConfig(
        name        = "North glazing",
        area        = 2.0,
        orientation = 'N',
        U_value     = 1.1,
        g_value     = 0.60,
        shading     = 0.0,
    ),

]

# ── Thermal bridges ────────────────────────────────────────────────────────
# ψ × length = extra heat loss per °C [W/K]

bridges = [

    ThermalBridge(
        name   = "Floor/wall junction",
        psi    = 0.15,      # [W/(m·K)] — typical insulated junction
        length = 40.0,      # [m] — perimeter of the floor
    ),

    ThermalBridge(
        name   = "Window frames",
        psi    = 0.04,      # [W/(m·K)] — good installation
        length = 20.0,      # [m] — total window perimeter
    ),

]

# ── Uncontrolled openings (infiltration) ──────────────────────────────────

openings = [
    OpeningConfig(name="Infiltration", ach_contribution=0.1),   # [1/h]
]


###############################################################################
# ─────────────────────────  2. Internal sources  ─────────────────────────────
###############################################################################

n_steps = int(t_tot / DT)     # number of DT-long steps over the whole year

# Occupancy: present 8h–22h
# (schedules are built at the DT step so they line up with the simulation;
#  build_daily_schedule maps each step back to its hour-of-day internally.)
occ_schedule = build_daily_schedule(
    hours_on=list(range(8, 22)), n_steps=n_steps, dt=DT)

occupants = OccupantConfig(
    n_occupants         = 2.0,
    sensible_per_person = 80.0,   # [W/person]
    latent_per_person   = 60.0,   # [W/person]
    schedule            = occ_schedule,
)

equipment = EquipmentConfig(
    power    = 300.0,             # [W] — fridge, TV, misc
    schedule = occ_schedule,
)

lighting = LightingConfig(
    power_density = 5.0,          # [W/m²]
    floor_area    = 80.0,         # [m²]
    schedule      = build_daily_schedule(hours_on=list(range(18, 23)),
                                         n_steps=n_steps, dt=DT),
)

ventilation = VentilationConfig(
    n_ach            = 0.5,       # [1/h] — RE2020 minimum
    hrv_efficiency   = 0.75,      # VMC double flux efficiency
    moisture_recovery= 0.0,       # sensible only
    air_velocity     = 0.1,       # [m/s] indoor air speed
)

hvac = HVACConfig(
    T_heat_set      = 20.0,       # [°C]
    T_cool_set      = 26.0,       # [°C]
    efficiency_heat = 3.0,        # COP heat pump
    efficiency_cool = 3.5,        # COP
    max_power_heat  = 5000.0,     # [W]
    max_power_cool  = 5000.0,     # [W]
    energy_carrier  = 'electricity',
    f_ep            = 2.3,        # RE2020 primary energy factor
)


###############################################################################
# ─────────────────────────  3. Simulation settings  ──────────────────────────
###############################################################################

# DT is defined at the top (Section 0) = 600 s (10 min).
ROOM_VOLUME     = 200.0         # [m³] — total air volume of the zone
T_ROOM_INIT     = 20.0          # [°C]   initial room temperature
RH_ROOM_INIT    = 0.50          # [-]    initial room relative humidity

# Effective internal thermal mass (furniture, floor slab, partitions) [J/K].
# Without it the room has only the (tiny) heat capacity of the air and its
# temperature oscillates unrealistically. ISO 13790 'medium-weight' building
# ≈ 110 kJ/(m²·K) of floor area. Set to 0.0 to recover the air-only behaviour.
INTERNAL_MASS   = 110e3 * 80.0  # [J/K]  (110 kJ/m²K × ~80 m² floor)

# Solar calculator — location taken straight from the weather file (Mâcon).
solar = SolarCalculator(
    latitude       = weather.lat,   # [°N]  from the weather file
    longitude      = weather.lon,   # [°E]  from the weather file
    start_doy      = 1,          # day-of-year of simulation start (file starts 1 Jan)
    cloud_factor   = 0.35,       # 0 = clear sky, 0.5 = moderately cloudy
    ground_albedo  = 0.20,
    timezone_offset= 1.0,        # CET
)


###############################################################################
# ─────────────────────────  4. RE2020 parameters  ────────────────────────────
###############################################################################

FLOOR_AREA   = 80.0             # [m²] habitable floor area SHON_RT
CLIMATE_ZONE = 'H1b'            # H1a/H1b/H1c/H2a/H2b/H2c/H2d/H3

re2020 = RE2020Evaluator(
    floor_area   = FLOOR_AREA,
    climate_zone = CLIMATE_ZONE,
)
re2020.set_carrier(heating='electricity', cooling='electricity')


###############################################################################
# ─────────────────────────  5. Run simulation  ───────────────────────────────
###############################################################################

sim = RoomSimulation(
    wall_configs   = walls,
    window_configs = windows,
    bridge_configs = bridges,
    opening_configs= openings,
    occupants      = occupants,
    equipment      = equipment,
    lighting       = lighting,
    ventilation    = ventilation,
    hvac           = hvac,
    solar_calc     = solar,
    volume         = ROOM_VOLUME,
    internal_mass  = INTERNAL_MASS,
    T_room_init    = T_ROOM_INIT,
    RH_room_init   = RH_ROOM_INIT,
    re2020         = re2020,
)

# Print wall summary before running
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


###############################################################################
# ─────────────────────────  6. RE2020 report  ────────────────────────────────
###############################################################################

# Add lighting energy to the RE2020 evaluator.
# Energy [kWh] = power [W] × hours_on / 1000.  The number of lighting-on
# hours is obtained from the schedule: (sum of on-factors) × DT / 3600.
hours_light = sum(lighting.schedule) * DT / 3600.0          # [h] over the year
E_light_kWh = lighting.power_density * FLOOR_AREA * hours_light / 1000.0
re2020.add_lighting_kWh(E_light_kWh)

report = re2020.report()
re2020.print_report(report)


###############################################################################
# ─────────────────────────  7. Outputs  ──────────────────────────────────────
###############################################################################

# Time axis in days, using the real time step DT (not assuming hours).
n_out  = len(T_room_arr)
t_days = np.arange(n_out) * DT / 86400.0

# Outdoor temperature resampled onto the simulation steps (the weather file is
# hourly, the simulation is 10-min) so the comfort map pairs matching instants.
t_steps  = np.arange(n_out) * DT
Text_out = np.interp(t_steps, time_bc, Text_bc)

# 3D schematic of the walls that were built
plot_walls_3d(sim, save='walls_3d.pdf')

# Room temperature and humidity
plot_room_climate(sim, t_days=t_days, save='room_climate.pdf')

# HVAC power and cumulative energy
plot_hvac_energy(sim, t_days=t_days, save='room_energy.pdf')

# Wall profiles at day 50
plot_wall_profiles(sim, day=50, save='wall_profiles.pdf')

# RE2020 bar chart  (single scenario here — add more to compare)
plot_re2020_comparison({'Hempcrete + RE': report}, save='re2020.pdf')

# Adaptive comfort map
plot_comfort_map(sim, Text_out, save='comfort_map.pdf')

# Energy balance table
table_energy_balance(sim, floor_area=FLOOR_AREA)

# CSV export
export_csv(sim, filepath='results_room.csv', dt=DT)

print("\nAll figures and results saved.")


###############################################################################
# ─────────────────────────  8. Multi-scenario comparison (optional)  ─────────
###############################################################################
# To compare two configurations, run a second simulation and call:
#
#   compare_scenarios(
#       {'Hempcrete walls': sim, 'Concrete walls': sim2},
#       save='comparison.pdf'
#   )
#
#   plot_re2020_comparison(
#       {'Hempcrete': report1, 'Concrete': report2},
#       save='re2020_comparison.pdf'
#   )


###############################################################################
# ─────────────────────────  9. SketchUp / Pleiades import  ───────────────────
###############################################################################

def import_geometry(json_path: str):
    """
    Import building geometry from a JSON file.

    Expected JSON format (exported from SketchUp plugin or Pleiades)
    -----------------------------------------------------------------
    {
      "walls": [
        { "name": "...", "mat": [...], "emat": [...],
          "area": 20.0, "orientation": "S" }
      ],
      "windows": [
        { "name": "...", "area": 8.0, "orientation": "S",
          "U_value": 1.1, "g_value": 0.6 }
      ]
    }

    Returns
    -------
    walls   : list of WallConfig
    windows : list of WindowConfig
    """
    import json
    with open(json_path, 'r', encoding='utf-8') as f:
        data_json = json.load(f)

    walls_out = [
        WallConfig(
            name        = w['name'],
            mat         = w['mat'],
            emat        = w['emat'],
            area        = w['area'],
            orientation = w.get('orientation', 'S'),
            Mesh_Opt    = w.get('Mesh_Opt', 0),
            liq         = w.get('liq', 0),
        )
        for w in data_json.get('walls', [])
    ]

    windows_out = [
        WindowConfig(
            name        = ww['name'],
            area        = ww['area'],
            orientation = ww.get('orientation', 'S'),
            U_value     = ww.get('U_value', 1.1),
            g_value     = ww.get('g_value', 0.60),
            shading     = ww.get('shading', 0.0),
        )
        for ww in data_json.get('windows', [])
    ]

    return walls_out, windows_out

# Usage:
#   walls, windows = import_geometry('my_building.json')
