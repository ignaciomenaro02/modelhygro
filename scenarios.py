# -*- coding: utf-8 -*-
"""
scenarios.py
============
Named variants of the simulation. A scenario lists ONLY what differs from section 1 of
simulation.py; everything else keeps its value there.

    "name": {CONFIG_VARIABLE: new value, ..., "note": "what this scenario is"}

Keys are the upper-case variables of simulation.py section 1 (ZONE, MODE, WALL_STACK,
MATERIAL_PARAMS, ACH_BASE, HEATING, ...); a misspelled key stops the run. Two shortcuts for walls:

    "WALL_STACK"  : one list of layers for the four walls
    "WALL_LAYERS" : {"South": [...], ...}, only the walls listed change

Scenarios combine with "+", applied from left to right (the last one wins on a shared variable):

    python simulation.py --scenario poster_hemp+hemp_sourced+dry

Command-line flags (--zone, --mode, --start, ...) override everything.
"""

_FILM = 100.0     # polyethylene film: equivalent air-layer thickness Sd [m]

SCENARIOS = {

    "base": {"note": "section 1 of simulation.py, unchanged"},

    # ── Walls ─────────────────────────────────────────────────────────────────
    "hemp_sourced":  {"WALL_STACK": [("Hempcrete", 0.30)],
                      "note": "hemp concrete 30 cm, sourced properties (Moujalled 2018), lambda(w)"},
    "earth_sourced": {"WALL_STACK": [("Rammed_Earth", 0.30)],
                      "note": "rammed earth 30 cm, sourced properties (Chabriac/Soudani 2016, Bui 2025), lambda(w)"},
    "hemp_earth_plaster": {"WALL_STACK": [("Hempcrete", 0.30), ("Earth_Plaster", 0.02)],
                           "note": "hemp concrete + 2 cm earth plaster inside"},
    "earth_insulated": {"WALL_STACK": [("Wood_Fiber", 0.10), ("Rammed_Earth", 0.30)],
                        "note": "rammed earth with 10 cm wood-fibre insulation outside"},
    "south_insulated": {"WALL_LAYERS": {"South": [("Wood_Fiber", 0.10), ("Hempcrete", 0.30)]},
                        "note": "wood-fibre insulation on the south wall only"},

    # ── Material settings (materials_library.py PARAMS) ───────────────────────
    "iso_mean": {"MATERIAL_PARAMS": {"Hempcrete": {"iso_branch": "mean"}},
                 "note": "hemp concrete isotherm: mean of the adsorption and desorption curves"},

    # ── Physics switches ──────────────────────────────────────────────────────
    "dry":         {"HYGRO_EFFECT": False,
                    "note": "dry walls: heat conduction only, dry heat capacity (as in dynamic thermal simulation)"},
    "comfort_air": {"COMFORT_TEMP": "air", "note": "comfort indicators with the air temperature"},

    # ── Moisture exchange of 30 cm hemp concrete with the room (hygro study, Sept. 2026) ─
    "hemp_open":   {"WALL_STACK": [("Hempcrete", 0.30)], "WALL_INTERIOR_SD": {}, "HYGRO_EFFECT": True,
                    "note": "bare interior surface: the walls exchange moisture with the room"},
    "hemp_film":   {"WALL_STACK": [("Hempcrete", 0.30)], "HYGRO_EFFECT": True,
                    "WALL_INTERIOR_SD": {"South": _FILM, "North": _FILM, "East": _FILM, "West": _FILM,
                                         "Roof": _FILM},
                    "note": "PE film (Sd = 100 m) on the interior surface: no moisture exchange with the room"},
    "hemp_closed": {"WALL_STACK": [("Hempcrete", 0.30)], "WALL_INTERIOR_SD": {}, "HYGRO_EFFECT": False,
                    "note": "classic thermal model: no moisture in the walls"},

    # ── Poster GdR MBGS, Pornichet, October 2026 ──────────────────────────────
    # Hemp concrete walls (old material values), vapour-tight roof, no windows, RE2020 Th-D H1c
    # over one year, RE2020 heating and internal gains, hygiene ventilation 35 m³/h (minimum
    # exhaust flow of the arrêté of 24 March 1982 for a one-room dwelling) = 0.56 1/h.
    "poster_hemp": {
        "ZONE": "H1c", "MODE": "Th-D", "START_DATE": "01/01", "END_DATE": "01/01",
        "SPINUP_DAYS": 730, "HEATING": True, "HEATING_SEASON": "RE2020",
        "WALL_STACK": [("Hempcrete_legacy", 0.30)], "WALL_INTERIOR_SD": {}, "HYGRO_EFFECT": True,
        "ROOF": True, "ROOF_HYGRO": False, "WINDOWS": [], "THERMAL_BRIDGES": [],
        "ACH_BASE": 35.0 / 62.5, "ACH_OVERVENT": 0.0, "INTERNAL_GAINS": "RE2020",
        "note": "poster: hemp concrete 30 cm, RE2020 Th-D H1c, RE2020 gains, 35 m3/h ventilation",
    },

    # ── Use cases ─────────────────────────────────────────────────────────────
    "typical_heated": {"MODE": "Th-BC", "START_DATE": "01/01", "END_DATE": "01/01", "SPINUP_DAYS": 730,
                       "HEATING": True, "HEATING_SEASON": (274, 120),
                       "note": "typical year, heated 1 Oct - 30 Apr (same calendar for every wall)"},
    "heatwave_free":  {"MODE": "Th-D", "START_DATE": "01/01", "END_DATE": "01/01", "SPINUP_DAYS": 730,
                       "HEATING": False,
                       "note": "year with a heat wave, free-floating room"},

    # ── Ventilation ───────────────────────────────────────────────────────────
    "ach_low":         {"ACH_BASE": 0.3, "note": "hygiene ventilation 0.3 1/h"},
    "ach_high":        {"ACH_BASE": 1.0, "note": "hygiene ventilation 1.0 1/h"},
    "overvent_none":   {"ACH_OVERVENT": 0.0, "note": "no night overventilation"},
    "overvent_base":   {"ACH_OVERVENT": 4.0, "note": "4 1/h added on summer nights"},
    "overvent_strong": {"ACH_OVERVENT": 8.0, "note": "8 1/h added on summer nights"},

    # ── Spin-up and climate ───────────────────────────────────────────────────
    "spinup_365": {"SPINUP_DAYS": 365, "note": "one year of spin-up"},
    "spinup_0":   {"SPINUP_DAYS": 0,   "note": "no spin-up: walls start at the outdoor state"},
    "zone_h1a":   {"ZONE": "H1a", "note": "climate zone H1a"},
    "zone_h3":    {"ZONE": "H3",  "note": "climate zone H3"},
}

# Poster with night window opening by the occupants (22h-8h, May-September), controlled as in the
# RE2020 manual window management: open at 23 °C, closed below 22 °C, only when the outdoor air
# is cooler than the room, scaled 0 -> 1 between 10 and 18 °C outdoors.
for _name, _ach in (("poster_hemp_vent2", 2.0), ("poster_hemp_vent", 4.0)):
    SCENARIOS[_name] = {**SCENARIOS["poster_hemp"], "ACH_OVERVENT": _ach, "OVERVENT_CONTROL": True,
                        "note": f"poster + controlled night window opening, up to {_ach:g} 1/h"}

# Old names kept so that earlier studies still run
SCENARIOS["hemp"]         = {**SCENARIOS["hemp_sourced"], "note": "same as hemp_sourced"}
SCENARIOS["rammed_earth"] = {**SCENARIOS["earth_sourced"], "note": "same as earth_sourced"}
