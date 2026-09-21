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
        "WALL_INTERIOR_SD": {"South": _FILM, "North": _FILM, "East": _FILM, "West": _FILM},
        "note": "same walls with a PE film (Sd = 100 m) inside: no moisture exchange with the room "
                "(the vapour barrier is a surface resistance, not a layer)",
    },
    "hemp_closed": {
        "WALL_STACK": [("Hempcrete", 0.30)], "WALL_INTERIOR_SD": {}, "HYGRO_EFFECT": False,
        "note": "classic thermal model: walls closed to vapour on both faces (no moisture at all)",
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
    "overvent_strong": {"ACH_OVERVENT": 8.0, "note": "8 1/h added on summer nights (base 4)"},

    # ── Spin-up length ────────────────────────────────────────────────────────
    "spinup_365": {"SPINUP_DAYS": 365, "note": "one year of spin-up"},
    "spinup_0":   {"SPINUP_DAYS": 0,   "note": "no spin-up: walls start at the outdoor state"},

    # ── Climate ───────────────────────────────────────────────────────────────
    "zone_h1a": {"ZONE": "H1a", "note": "climate zone H1a"},
    "zone_h3":  {"ZONE": "H3",  "note": "climate zone H3"},
}
