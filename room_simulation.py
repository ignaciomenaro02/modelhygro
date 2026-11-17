# -*- coding: utf-8 -*-
"""
room_simulation.py
==================
One room: a well-mixed air node surrounded by walls, each wall solved in 1D with heat and
moisture transfer (wall_class.Wall).

Every time step does, in this order:

1. Room air temperature, implicit in time:
       C·dT/dt = Σ_walls h·A·(T_surface − T) + (Σ U·A_windows + Σ ψ·L + ventilation)·(T_ext − T)
                 + solar gains through windows + internal gains + heating
   C is the heat capacity of the air plus an internal mass (furniture, floor, partitions).
   Heating is an ideal thermostat: exactly the power needed to reach the setpoint, capped.
2. Room air vapour pressure, implicit in time:
       (V / (Rv·T))·dPv/dt = Σ_walls hm·A·(Pv_surface − Pv) + ventilation·(Pv_ext − Pv) + sources
3. Walls, with the new room state as their inner boundary condition and the sol-air
   temperature T_ext + α·I/h_ext on their outer face.

The air and the walls are thus coupled sequentially: the air sees the wall surfaces of the
previous step, the walls see the air of the current step.
"""

import sys
import numpy as np

import library as lib
from wall_layer import WallLayer
from wall_class import Wall

RHO_AIR = 1.2       # air density [kg/m³]


class RoomSimulation:
    """
    wall_configs    : list of WallConfig
    window_configs  : list of WindowConfig
    bridge_configs  : list of ThermalBridge
    opening_configs : list of OpeningConfig (air leaks)
    occupants, equipment, lighting, ventilation, hvac : configurations from sources.py
    solar_calc      : SolarCalculator
    volume          : room air volume [m³]
    internal_mass   : heat capacity added to the air [J/K] (ISO 13790 medium: 110 kJ/(m²·K) × floor)
    T_room_init     : initial room and wall temperature [°C]
    RH_room_init    : initial room and wall relative humidity [-]
    re2020          : RE2020Evaluator collecting the energy and comfort indicators (or None)
    wall_substeps   : wall steps per room step. The wall coefficients are taken at the start of
                      each wall step; with a 1 h step a wet outer surface in the sun oscillates by
                      several K. 6 sub-steps (600 s) agree with 60 within 0.1 K.
    """

    def __init__(self, wall_configs, window_configs=None, bridge_configs=None, opening_configs=None,
                 occupants=None, equipment=None, lighting=None, ventilation=None, hvac=None,
                 solar_calc=None, volume=None, internal_mass=0.0,
                 T_room_init=20.0, RH_room_init=0.5, re2020=None, wall_substeps=1):
        self.wall_configs    = wall_configs
        self.window_configs  = window_configs or []
        self.bridge_configs  = bridge_configs or []
        self.opening_configs = opening_configs or []
        self.occupants       = occupants
        self.equipment       = equipment
        self.lighting        = lighting
        self.ventilation     = ventilation
        self.hvac            = hvac
        self.solar_calc      = solar_calc
        self.volume          = volume
        self.internal_mass   = float(internal_mass)
        self.re2020_ev       = re2020
        self.wall_substeps   = max(1, int(wall_substeps))

        self.T_room     = float(T_room_init)     # [°C]
        self.RH_room    = float(RH_room_init)    # [-]
        self.t_sim      = 0.0                    # [s]
        self.step_count = 0

        # The walls start in equilibrium with the room air
        self.walls = []
        for cfg in wall_configs:
            layer = WallLayer(cfg.mat, cfg.emat, Mesh_Opt=cfg.Mesh_Opt, liq=cfg.liq,
                              mesh_size=cfg.mesh_size, label=cfg.name)
            self.walls.append(Wall(layer, cfg.h_ext, cfg.hm_ext, cfg.h_int, cfg.hm_int,
                                   T_init=T_room_init + 273.15, RH_init=RH_room_init, hygro=cfg.hygro))

        # Histories: the initial value, then one value per step
        self.StockT_room      = [self.T_room]
        self.StockRH_room     = [self.RH_room]
        self.StockQ_HVAC      = [0.0]           # heating (+) or cooling (−) power [W]
        self.StockQ_lat_total = [0.0]           # latent heat of the wall moisture exchange [W]
        self.StockACH         = []              # air change rate of each step [1/h]
        self.StockT_walls     = {cfg.name: [float(w.T[-1, 0]) - 273.15]   # inner surface [°C]
                                 for cfg, w in zip(wall_configs, self.walls)}

        self.E_heat_kWh = 0.0                   # final heating energy
        self.E_cool_kWh = 0.0                   # final cooling energy

    def load_state(self, other):
        """
        Start from the final state of another simulation with the same walls (the spin-up run).
        Room air and wall fields are copied node by node, and every history restarts from there,
        so the spin-up leaves no trace in the results.
        """
        self.T_room, self.RH_room = other.T_room, other.RH_room
        for wall, ref in zip(self.walls, other.walls):
            wall.T, wall.RH, wall.Pc, wall.U = (a.copy() for a in (ref.T, ref.RH, ref.Pc, ref.U))
            wall.StockT  = [wall.T.copy()]
            wall.StockRH = [wall.RH.copy()]
            wall.StockPc = [wall.Pc.copy()]
        self.StockT_room  = [self.T_room]
        self.StockRH_room = [self.RH_room]
        for name, wall in zip(list(self.StockT_walls), self.walls):
            self.StockT_walls[name] = [float(wall.T[-1, 0]) - 273.15]

    def step(self, T_ext_C, RH_ext, dt, Q_extra_W=0.0):
        """
        Advance the room and its walls by dt [s].

        T_ext_C, RH_ext : outdoor air temperature [°C] and relative humidity [-]
        Q_extra_W       : extra heat injected in the room air [W], e.g. the exchange with a
                          neighbouring room when several rooms are coupled
        """
        n        = self.step_count
        T_room_K = self.T_room + 273.15
        T_ext_K  = T_ext_C + 273.15
        Pv_room  = lib.Pv(T_room_K, self.RH_room)
        Pv_ext   = lib.Pv(T_ext_K, RH_ext)

        # ── Exchanges with the walls (surface state of the previous step) ──────
        # Written as conductance × (driving value − room value) so the room can be solved implicitly
        G_walls, Q_walls = 0.0, 0.0       # Σ h·A [W/K] and Σ h·A·T_surface [W]
        Gm_walls, M_walls = 0.0, 0.0      # Σ hm·A [kg/(s·Pa)] and Σ hm·A·Pv_surface [kg/s]
        for wall, cfg in zip(self.walls, self.wall_configs):
            T_s  = float(wall.T[-1, 0])
            Pv_s = lib.Pv(T_s, float(wall.RH[-1, 0]))
            G_walls  += cfg.h_int * cfg.area
            Q_walls  += cfg.h_int * cfg.area * T_s
            Gm_walls += cfg.hm_int * cfg.area
            M_walls  += cfg.hm_int * cfg.area * Pv_s

        # Latent heat of the moisture the walls release (+) or take up (−) [W]. Reported only:
        # the evaporation heat is already taken from the wall surface in its own boundary
        # condition, so adding it to the air would count it twice.
        Q_lat_walls = lib.Lv * (M_walls - Gm_walls * Pv_room)

        # ── Gains that do not depend on the room temperature ───────────────────
        Q_solar = sum(self.solar_calc.solar_gains_window(self.t_sim, w) for w in self.window_configs) \
            if self.solar_calc is not None else 0.0
        Q_int, M_int = 0.0, 0.0           # internal heat [W] and vapour [kg/s]
        if self.occupants is not None:
            Q_int += self.occupants.sensible_at(n)
            M_int += self.occupants.moisture_at(n)
        if self.equipment is not None:
            Q_int += self.equipment.heat_at(n)
        if self.lighting is not None:
            Q_int += self.lighting.heat_at(n)

        # ── Exchanges with the outdoor air ─────────────────────────────────────
        ach = self.ventilation.flow_at(n, self.T_room, T_ext_C) if self.ventilation is not None else 0.0
        ach += sum(op.ach_contribution for op in self.opening_configs)
        q_air    = ach * self.volume / 3600.0                                   # [m³/s]
        hrv      = getattr(self.ventilation, "hrv_efficiency", 0.0)    if self.ventilation else 0.0
        hrv_mois = getattr(self.ventilation, "moisture_recovery", 0.0) if self.ventilation else 0.0

        G_ext = (sum(w.U_value * w.area for w in self.window_configs)          # windows
                 + sum(b.psi * b.length for b in self.bridge_configs)          # thermal bridges
                 + q_air * RHO_AIR * lib.CpA * (1.0 - hrv))                    # air renewal [W/K]
        # vapour carried by the air: flow × Pv/(Rv·T)
        Gm_vent = q_air / (lib.Rv * T_room_K) * (1.0 - hrv_mois)               # [kg/(s·Pa)]

        # ── Room temperature (implicit Euler) ──────────────────────────────────
        #   T_new = (a·T_old + Q_drive + Q_HVAC) / D,   a = C/dt,   D = a + Σ conductances
        a       = (RHO_AIR * self.volume * lib.CpA + self.internal_mass) / dt
        D       = a + G_walls + G_ext
        Q_drive = Q_walls + G_ext * T_ext_K + Q_solar + Q_int + Q_extra_W
        T_free  = (a * T_room_K + Q_drive) / D           # temperature without heating [K]

        Q_HVAC = self._thermostat(T_free, a * T_room_K + Q_drive, D, dt)

        self.T_room = float(np.clip((a * T_room_K + Q_drive + Q_HVAC) / D - 273.15, -10.0, 50.0))

        # ── Room humidity (implicit Euler) ─────────────────────────────────────
        #   Pv_new = (b·Pv_old + sources) / (b + Σ conductances),   b = V/(Rv·T·dt)
        b      = self.volume / (lib.Rv * T_room_K) / dt
        Pv_new = (b * Pv_room + M_walls + Gm_vent * Pv_ext + M_int) / (b + Gm_walls + Gm_vent)
        Psat   = lib.Psat(self.T_room + 273.15, self.RH_room)
        self.RH_room = float(np.clip(max(Pv_new, 0.0) / Psat, 0.05, 1.0))

        # ── Walls ──────────────────────────────────────────────────────────────
        # Outer face: sol-air temperature for the heat, but the vapour pressure of the outdoor
        # air (the sun heats the surface, it does not add vapour).
        T_room_new_K = self.T_room + 273.15
        for wall, cfg in zip(self.walls, self.wall_configs):
            T_sol_air = T_ext_K
            if self.solar_calc is not None:
                I = self.solar_calc.irradiance(self.t_sim, cfg.orientation, cfg.tilt)
                T_sol_air = T_ext_K + cfg.absorptance * I / cfg.h_ext
            for j in range(self.wall_substeps):
                wall.step(T_sol_air, RH_ext, T_room_new_K, self.RH_room, dt / self.wall_substeps,
                          Pv_ext=Pv_ext, store=(j == self.wall_substeps - 1))

        if self.re2020_ev is not None:
            self.re2020_ev.add_discomfort_hours(np.array([self.T_room]), dt)

        self.StockT_room.append(self.T_room)
        self.StockRH_room.append(self.RH_room)
        self.StockQ_HVAC.append(Q_HVAC)
        self.StockQ_lat_total.append(Q_lat_walls)
        self.StockACH.append(ach)
        for cfg, wall in zip(self.wall_configs, self.walls):
            self.StockT_walls[cfg.name].append(float(wall.T[-1, 0]) - 273.15)

        self.t_sim      += dt
        self.step_count += 1

    def _thermostat(self, T_free, num, D, dt):
        """
        Ideal heating and cooling: if the free temperature T_free [K] is outside the setpoints,
        return the power [W] that brings the room exactly onto the setpoint, within the power
        limits (T_new = (num + Q)/D, so Q = T_set·D − num). Also counts the energy.
        """
        if self.hvac is None:
            return 0.0
        # A heating controller (heating.RE2020Heating) can change the setpoint hour by hour,
        # or forbid heating (setpoint None) outside the heating season
        ctrl  = getattr(self.hvac, "heating_control", None)
        T_set = self.hvac.T_heat_set if ctrl is None else ctrl.heating_setpoint(self.step_count)
        T_heat = None if T_set is None else T_set + 273.15
        T_cool = self.hvac.T_cool_set + 273.15

        Q = 0.0
        if T_heat is not None and T_free < T_heat:
            Q = float(np.clip(T_heat * D - num, 0.0, self.hvac.max_power_heat))
            E = Q / self.hvac.efficiency_heat * dt / 3.6e6
            self.E_heat_kWh += E
            if self.re2020_ev is not None:
                self.re2020_ev.add_heating_kWh(E, self.hvac.energy_carrier)       # final energy
                self.re2020_ev.add_heating_need_kWh(Q * dt / 3.6e6)               # need (Bbio)
        elif T_free > T_cool:
            Q = float(np.clip(T_cool * D - num, -self.hvac.max_power_cool, 0.0))
            E = -Q / self.hvac.efficiency_cool * dt / 3.6e6
            self.E_cool_kWh += E
            if self.re2020_ev is not None:
                self.re2020_ev.add_cooling_kWh(E, self.hvac.energy_carrier)
                self.re2020_ev.add_cooling_need_kWh(-Q * dt / 3.6e6)

        if ctrl is not None:
            ctrl.record(self.step_count, T_free - 273.15, max(Q, 0.0), dt)
        return Q

    def run(self, time_bc, Text_bc, RHext_bc, dt=3600.0, verbose=True):
        """
        Run over the outdoor conditions Text_bc [°C] and RHext_bc [-] given at the times time_bc [s].
        Each step uses the outdoor conditions at its end.
        """
        n_steps = int(time_bc.max() / dt)
        for s in range(n_steps):
            t = (s + 1) * dt
            self.step(float(np.interp(t, time_bc, Text_bc)), float(np.interp(t, time_bc, RHext_bc)), dt)

            if verbose:
                done = (s + 1) / n_steps
                bar  = "#" * int(round(20 * done))
                sys.stdout.write(f"\rSimulation: [{bar:<20}] {done * 100:.1f}%" + (" DONE\n" if done >= 1 else ""))
                sys.stdout.flush()
            if np.isnan(self.T_room) or np.isnan(self.RH_room):
                print(f"\n[!] NaN in the room state at step {s + 1}: stopping.")
                break
        return np.array(self.StockT_room), np.array(self.StockRH_room)
