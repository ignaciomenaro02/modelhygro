# -*- coding: utf-8 -*-
"""
scenarios.py
============
Named variants of the simulation. A scenario lists ONLY what differs from the
configuration in section 1 of simulation.py; everything not listed keeps its value there.

    "name": {CONFIG_VARIABLE: new value, ..., "note": "free text"}

Keys are the upper-case variables of simulation.py section 1 (ZONE, MODE, START_DATE,
END_DATE, WALL_STACK, WALL_LAYERS, ACH_BASE, ACH_OVERVENT, HEATING, ...). A misspelled key stops
the run with an error message. Two shortcuts for the walls:

    "WALL_STACK"  : one list of layers applied to ALL four walls
    "WALL_LAYERS" : {"South": [...], ...}  only the walls you list change

Lower-case keys ("note") are documentation and are ignored.

Scenarios can be COMBINED with "+": they are applied from left to right and the last one wins
when two of them set the same variable. Each scenario below changes one thing, so a study is a
combination of a wall, a use case and, if needed, a ventilation:

    hemp_open+typical_heated+ach_high

Run one       :  python simulation.py --scenario hemp_open+typical_heated
Run several   :  python run_scenarios.py hemp_open hemp_film        (or --all)
Command-line flags (--zone, --start, ...) still override the scenarios.
"""

_FILM = 100.0     # PE vapour film: equivalent air-layer thickness Sd [m]

SCENARIOS = {

    "base": {
        "note": "Configuration of simulation.py section 1, unchanged",
    },

    # ── Materials ─────────────────────────────────────────────────────────────
    "hemp": {
        "WALL_STACK": [("Hempcrete", 0.30)],
        "note": "30 cm hempcrete on all four walls",
    },
    "rammed_earth": {
        "WALL_STACK": [("Rammed_Earth", 0.30)],
        "note": "30 cm rammed earth on all four walls",
    },
    "hemp_earth_plaster": {
        "WALL_STACK": [("Hempcrete", 0.30), ("Earth_Plaster", 0.02)],
        "note": "hempcrete + 2 cm earth plaster on the inside",
    },
    "earth_insulated": {
        "WALL_STACK": [("Wood_Fiber", 0.10), ("Rammed_Earth", 0.30)],
        "note": "10 cm wood-fibre exterior insulation on rammed earth",
    },
    "south_insulated": {
        "WALL_LAYERS": {"South": [("Wood_Fiber", 0.10), ("Hempcrete", 0.30)]},
        "note": "only the south wall gets exterior wood-fibre insulation",
    },

    # ── Hygroscopic behaviour of 30 cm hempcrete (the walls of the hygro study) ─────────
    "hemp_open": {
        "WALL_STACK": [("Hempcrete", 0.30)], "WALL_INTERIOR_SD": {}, "HYGRO_EFFECT": True,
        "note": "hempcrete, bare interior surface: the walls exchange moisture with the room",
    },
    "hemp_film": {
        "WALL_STACK": [("Hempcrete", 0.30)], "HYGRO_EFFECT": True,
        "WALL_INTERIOR_SD": {"South": _FILM, "North": _FILM, "East": _FILM, "West": _FILM,
                             "Roof": _FILM},     # the roof too, since ROOF = True adds one
        "note": "same walls with a PE film (Sd = 100 m) inside: no moisture exchange with the room "
                "(the vapour barrier is a surface resistance, not a layer)",
    },
    "hemp_closed": {
        "WALL_STACK": [("Hempcrete", 0.30)], "WALL_INTERIOR_SD": {}, "HYGRO_EFFECT": False,
        "note": "classic thermal model: walls closed to vapour on both faces (no moisture at all)",
    },

    # ── Poster (GdR MBGS, Pornichet 2026): one regulatory case, no variants ──────────
    # Hempcrete walls (an earth version will follow), roof thermal only (vapour-tight),
    # no windows, RE2020 Th-D climate H1c over a full year with the RE2020 heating scenario and
    # the RE2020 internal gains (1.28 equivalent adults for 25 m², see heating.py),
    # hygiene ventilation = minimum total exhaust flow of the arrêté du 24 mars 1982 for a
    # one-main-room dwelling, 35 m³/h, i.e. 35 / 62.5 m³ = 0.56 1/h, constant, no night overventilation.
    "poster_hemp": {
        "ZONE": "H1c", "MODE": "Th-D", "START_DATE": "01/01", "END_DATE": "01/01",
        "SPINUP_DAYS": 730, "HEATING": True, "HEATING_SEASON": "RE2020",
        "WALL_STACK": [("Hempcrete", 0.30)], "WALL_INTERIOR_SD": {}, "HYGRO_EFFECT": True,
        "ROOF": True, "ROOF_HYGRO": False, "WINDOWS": [], "THERMAL_BRIDGES": [],
        "ACH_BASE": 35.0 / 62.5, "ACH_OVERVENT": 0.0, "INTERNAL_GAINS": "RE2020",
        "note": "poster: hempcrete 30 cm, vapour-tight roof, RE2020 Th-D H1c, RE2020 internal gains, "
                "35 m3/h hygiene ventilation",
    },

    # ── Use cases (climate, period, heating, spin-up) ─────────────────────────
    "typical_heated": {
        "MODE": "Th-BC", "START_DATE": "01/01", "END_DATE": "01/01", "SPINUP_DAYS": 730,
        "HEATING": True, "HEATING_SEASON": (274, 120),
        "note": "typical year, heated 1 Oct-30 Apr with the RE2020 setpoints (same calendar for every "
                "wall, so energy is compared at equal service), 730 days of spin-up",
    },
    "heatwave_free": {
        "MODE": "Th-D", "START_DATE": "01/01", "END_DATE": "01/01", "SPINUP_DAYS": 730,
        "HEATING": False,
        "note": "year with a heat wave (Th-D), free-floating room, 730 days of spin-up",
    },

    # ── Ventilation (only these variables change; combine with any wall / use case) ───
    # simulation.py section 1: ACH_BASE (all day), ACH_OVERVENT (added on summer nights),
    # NIGHT_HOURS, SUMMER_MONTHS, and INFILTRATION (fixed 0.1 1/h, not changed here).
    "ach_low":   {"ACH_BASE": 0.3, "note": "hygiene ventilation 0.3 1/h (base 0.5)"},
    "ach_high":  {"ACH_BASE": 1.0, "note": "hygiene ventilation 1.0 1/h (base 0.5)"},
    "overvent_none":   {"ACH_OVERVENT": 0.0, "note": "no night overventilation: constant ACH_BASE"},
    "overvent_base":   {"ACH_OVERVENT": 4.0, "note": "4 1/h added on summer nights (value of section 1)"},
    "overvent_strong": {"ACH_OVERVENT": 8.0, "note": "8 1/h added on summer nights (base 4)"},

    # ── Spin-up length ────────────────────────────────────────────────────────
    "spinup_365": {"SPINUP_DAYS": 365, "note": "one year of spin-up"},
    "spinup_0":   {"SPINUP_DAYS": 0,   "note": "no spin-up: walls start at the outdoor state"},

    # ── Climate ───────────────────────────────────────────────────────────────
    "zone_h1a": {"ZONE": "H1a", "note": "climate zone H1a"},
    "zone_h3":  {"ZONE": "H3",  "note": "climate zone H3"},
}

# Poster, ventilated cases: the same room with night window opening by the occupants (22h-8h,
# May-September), controlled as in the RE2020 manual window management: open when the room
# reaches 23 °C, closed below 22 °C, only when the outdoor air is cooler than the room, scaled
# 0 → 1 between 10 and 18 °C outdoor. Two opening rates: up to 2 and up to 4 extra air changes.
for _name, _ach in (("poster_hemp_vent2", 2.0), ("poster_hemp_vent", 4.0)):
    SCENARIOS[_name] = {
        **SCENARIOS["poster_hemp"], "ACH_OVERVENT": _ach, "OVERVENT_CONTROL": True,
        "note": f"poster: as poster_hemp + controlled night window opening, up to {_ach:g} 1/h "
                "(22h-8h, May-September)",
    }
