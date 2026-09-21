# -*- coding: utf-8 -*-
"""
scenarios.py
============
Named variants of the simulation. A scenario lists ONLY what differs from the
configuration in section 1 of simulation.py; everything not listed keeps its value there.

    "name": {CONFIG_VARIABLE: new value, ..., "note": "free text"}

Keys are the upper-case variables of simulation.py section 1 (ZONE, MODE, START_DATE,
END_DATE, WALL_STACK, WALL_LAYERS, ACH_OVERVENT, HEATING, ...). A misspelled key stops
the run with an error message. Two shortcuts for the walls:

    "WALL_STACK"  : one list of layers applied to ALL four walls
    "WALL_LAYERS" : {"South": [...], ...}  only the walls you list change

Lower-case keys ("note") are documentation and are ignored.

Run one scenario :  python simulation.py --scenario rammed_earth
Run several      :  python run_scenarios.py rammed_earth h3        (or --all)
Command-line flags (--zone, --start, ...) still override the scenario.
"""

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

    # ── Ventilation ───────────────────────────────────────────────────────────
    "constant_ventilation": {
        "ACH_OVERVENT": 0.0,
        "note": "no night overventilation: constant 0.5 1/h",
    },
    "strong_overventilation": {
        "ACH_OVERVENT": 8.0,
        "note": "8 1/h added on summer nights",
    },

    # ── Climate ───────────────────────────────────────────────────────────────
    "zone_h1a": {"ZONE": "H1a", "note": "climate zone H1a"},
    "zone_h3":  {"ZONE": "H3",  "note": "climate zone H3"},

    # ── Vapour-barrier / hygroscopicity study (30 cm hempcrete on all four walls) ─────
    # Each scenario sets WALL_STACK, WALL_INTERIOR_SD and HYGRO_EFFECT itself, so the result
    # does not depend on the walls written in simulation.py section 1.
    # Reading the comparison:
    #   vb_bare  -> vb_film_sd100 : effect of the room-side moisture buffering (the barrier removes it)
    #   vb_layer_2cm vs vb_film   : thermal artefact of modelling the barrier as a 2 cm layer
    #   vb_no_hygro               : heat-only control (both faces closed to vapour)
    "vb_bare": {
        "WALL_STACK": [("Hempcrete", 0.30)],
        "WALL_INTERIOR_SD": {}, "HYGRO_EFFECT": True,
        "note": "hempcrete, bare interior surface: full moisture buffering (no barrier)",
    },
    "vb_film_sd100": {
        "WALL_STACK": [("Hempcrete", 0.30)],
        "WALL_INTERIOR_SD": {"South": 100.0, "North": 100.0, "East": 100.0, "West": 100.0},
        "HYGRO_EFFECT": True,
        "note": "hempcrete + PE film (Sd = 100 m) on the room side of all walls, modelled as a "
                "surface resistance: barrier without thermal artefact",
    },
    "vb_layer_2cm": {
        "WALL_STACK": [("Hempcrete", 0.30), ("Vapor_Barrier", 0.02)],
        "WALL_INTERIOR_SD": {}, "HYGRO_EFFECT": True,
        "note": "hempcrete + 2 cm Vapor_Barrier LAYER on the room side (Sd = 1000 m, adds thermal "
                "mass): the way the barrier was written originally",
    },
    "vb_no_hygro": {
        "WALL_STACK": [("Hempcrete", 0.30)],
        "WALL_INTERIOR_SD": {}, "HYGRO_EFFECT": False,
        "note": "control: both faces closed to vapour (heat-only behaviour)",
    },

    # ── Full year, with and without RE2020 heating ────────────────────────────
    "year_free_floating": {
        "MODE": "Th-BC", "START_DATE": "01/01", "END_DATE": "01/01", "HEATING": False,
        "note": "typical year, no heating",
    },
    "year_heated": {
        "MODE": "Th-BC", "START_DATE": "01/01", "END_DATE": "01/01", "HEATING": True,
        "note": "typical year with the RE2020 heating scenario",
    },
}
