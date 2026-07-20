# -*- coding: utf-8 -*-
"""
room_simulation.py
==================
RoomSimulation — couples multiple 1D hygrothermal wall models with a
lumped room air model.

Physics
-------
Each opaque wall is solved with the full 1D coupled heat-and-moisture
transfer model (WallLayer + Wall).  The room air is modelled as a
well-mixed single zone:

  Thermal balance (explicit coupling):
    m_air × Cp_air × dT_room/dt =
        Σ_walls  [h_int_i × A_i × (T_surf_i − T_room)]   ← wall convection
      + Σ_windows [U_w   × A_w × (T_ext  − T_room)]       ← window conduction
      + Σ_bridges [ψ_i   × L_i × (T_ext  − T_room)]       ← thermal bridges
      + Q_solar                                             ← solar through windows
      + Q_internal                                          ← occupants + equipment
      + Q_ventilation                                       ← supply air
      + Q_HVAC                                              ← heating / cooling

  Moisture balance:
    V × d(Pv_room)/dt =
        (G_walls + G_vent + G_occ) × Rv × T_room           ← [Pa/s]

  where G [kg/s] is vapour mass flux.

Usage
-----
  See main_room.py for a complete example.
"""

import sys
import time
import numpy as np

import library as lib
from wall_layer import WallLayer
from wall_class  import Wall
from solar       import SolarCalculator


class RoomSimulation:
    """
    Multi-wall room hygrothermal simulation.

    Parameters
    ----------
    wall_configs   : list of WallConfig
    window_configs : list of WindowConfig
    bridge_configs : list of ThermalBridge
    opening_configs: list of OpeningConfig
    occupants      : OccupantConfig
    equipment      : EquipmentConfig
    lighting       : LightingConfig
    ventilation    : VentilationConfig
    hvac           : HVACConfig
    solar_calc     : SolarCalculator
    volume         : float   Room air volume [m³].
    internal_mass  : float   Effective internal thermal mass [J/K] (furniture,
                             floor, partitions). 0 = air only (oscillatory).
                             ISO 13790 'medium' ≈ 110 kJ/(m²·K) × floor area.
    T_room_init    : float   Initial room temperature [°C].
    RH_room_init   : float   Initial room relative humidity [-].
    re2020         : RE2020Evaluator  (optional, pass None to skip)
    """

    def __init__(
        self,
        wall_configs,
        window_configs  = None,
        bridge_configs  = None,
        opening_configs = None,
        occupants       = None,
        equipment       = None,
        lighting        = None,
        ventilation     = None,
        hvac            = None,
        solar_calc      = None,
        volume          = 150.0,
        internal_mass   = 0.0,
        T_room_init     = 20.0,
        RH_room_init    = 0.50,
        re2020          = None,
    ):
        self.wall_configs   = wall_configs
        self.window_configs = window_configs  or []
        self.bridge_configs = bridge_configs  or []
        self.opening_configs= opening_configs or []
        self.occupants      = occupants
        self.equipment      = equipment
        self.lighting       = lighting
        self.ventilation    = ventilation
        self.hvac           = hvac
        self.solar_calc     = solar_calc
        self.volume         = volume
        # Effective internal thermal mass (furniture, floor, partitions) [J/K].
        # Added to the air heat capacity; gives the room realistic inertia.
        self.internal_mass  = float(internal_mass)
        self.re2020_ev      = re2020

        # Room state
        self.T_room  = float(T_room_init)    # [°C]
        self.RH_room = float(RH_room_init)   # [-]
        self.t_sim   = 0.0                   # [s]
        self.step_count = 0
        self.dt      = None                  # [s] time step, set when run() is called

        # Build wall objects
        self.layers = []
        self.walls  = []
        for cfg in wall_configs:
            layer = WallLayer(
                mat       = cfg.mat,
                emat      = cfg.emat,
                Mesh_Opt  = cfg.Mesh_Opt,
                liq       = cfg.liq,
                mesh_size = cfg.mesh_size,
                label     = cfg.name,
            )
            wall = Wall(
                layer   = layer,
                h_ext   = cfg.h_ext,
                hm_ext  = cfg.hm_ext,
                h_int   = cfg.h_int,
                hm_int  = cfg.hm_int,
                T_init  = T_room_init + 273.15,
                RH_init = RH_room_init,
            )
            self.layers.append(layer)
            self.walls.append(wall)

        # Result storage
        self.StockT_room  = [self.T_room]
        self.StockRH_room = [self.RH_room]
        self.StockQ_HVAC  = [0.0]
        # Store the initial interior-surface temperature too, so every result
        # series has the same length (initial value + one per step).
        self.StockT_walls = {
            cfg.name: [float(wall.T[-1, 0]) - 273.15]
            for cfg, wall in zip(wall_configs, self.walls)
        }

        # Energy accumulators [kWh]
        self.E_heat_kWh = 0.0
        self.E_cool_kWh = 0.0

    # ── Single time step ───────────────────────────────────────────────────────

    def step(self, T_ext_C: float, RH_ext: float, dt: float):
        """
        Advance the room and all walls by one time step dt [s].

        Parameters
        ----------
        T_ext_C : float   Exterior temperature [°C].
        RH_ext  : float   Exterior relative humidity [-].
        dt      : float   Time step [s].
        """
        T_room_K = self.T_room + 273.15
        T_ext_K  = T_ext_C    + 273.15
        Pv_room  = lib.Pv(T_room_K, self.RH_room)
        Pv_ext   = lib.Pv(T_ext_K,  RH_ext)

        # ── 1. Wall ↔ room exchange (conductance form) ─────────────────────────
        # Thermal flux is  Σ h_int·A·(T_surf − T_room).  We split it into a
        # conductance G_walls_th [W/K] and a "drive" Σ h_int·A·T_surf so the air
        # temperature can be solved IMPLICITLY below (stable at any time step).
        G_walls_th   = 0.0   # [W/K]  air↔walls thermal conductance
        Qdrive_walls = 0.0   # [W]    Σ h_int·A·T_surf
        G_walls      = 0.0   # [kg/s] vapour from walls into the room (moisture)

        for wall_obj, cfg in zip(self.walls, self.wall_configs):
            T_surf  = float(wall_obj.T[-1, 0])    # interior surface [K]
            RH_surf = float(wall_obj.RH[-1, 0])
            Pv_surf = lib.Pv(T_surf, RH_surf)

            G_walls_th   += cfg.h_int  * cfg.area
            Qdrive_walls += cfg.h_int  * cfg.area * T_surf
            G_walls      += cfg.hm_int * cfg.area * (Pv_surf - Pv_room)

        # ── 2-3. Windows + thermal bridges (exchange with OUTDOOR air) ─────────
        G_windows = sum(w.U_value * w.area for w in self.window_configs)   # [W/K]
        G_bridges = sum(b.psi * b.length   for b in self.bridge_configs)   # [W/K]

        # ── 4. Solar gains (independent of T_room) ─────────────────────────────
        Q_solar = 0.0
        if self.solar_calc is not None:
            for w in self.window_configs:
                Q_solar += self.solar_calc.solar_gains_window(self.t_sim, w)

        # ── 5. Internal gains (independent of T_room) ──────────────────────────
        Q_internal = 0.0
        G_internal = 0.0   # [kg/s]

        if self.occupants is not None:
            Q_internal += self.occupants.sensible_at(self.step_count)
            G_internal += self.occupants.moisture_at(self.step_count)
        if self.equipment is not None:
            Q_internal += self.equipment.heat_at(self.step_count)
        if self.lighting is not None:
            Q_internal += self.lighting.heat_at(self.step_count)

        # ── 6. Ventilation + infiltration (exchange with OUTDOOR air) ──────────
        n_ach = 0.0
        if self.ventilation is not None:
            n_ach = self.ventilation.flow_at(self.step_count)
        for op in self.opening_configs:
            n_ach += op.ach_contribution

        n_vol_s  = n_ach * self.volume / 3600.0     # [m³/s]
        rho_air  = 1.2                               # [kg/m³]
        hrv      = getattr(self.ventilation, 'hrv_efficiency',    0.0) if self.ventilation else 0.0
        hrv_mois = getattr(self.ventilation, 'moisture_recovery', 0.0) if self.ventilation else 0.0

        G_vent_th = n_vol_s * rho_air * lib.CpA * (1.0 - hrv)             # [W/K]
        G_vent    = n_vol_s * rho_air * (Pv_ext - Pv_room) / (lib.Rv * T_room_K) * (1.0 - hrv_mois)

        # ── 7. Implicit room-air energy balance ────────────────────────────────
        # Backward-Euler step of   C_eff·dT/dt = Σ G_i·(T_i − T_room) + Q_indep:
        #
        #     T_new = (a·T_old + Q_drive + Q_HVAC) / (a + G_tot),   a = C_eff/dt
        #
        # This is STABLE for any dt — it removes the explicit-scheme oscillation
        # that appeared when the air time-constant (~7 min) was below the step.
        # C_eff = air heat capacity + internal thermal mass (furniture/floor).
        m_air   = rho_air * self.volume                    # [kg]
        C_eff   = m_air * lib.CpA + self.internal_mass     # [J/K]
        a       = C_eff / dt                               # [W/K]

        G_env   = G_windows + G_bridges + G_vent_th        # conductance to outdoor air
        G_tot   = G_walls_th + G_env                       # [W/K] total air conductance
        Q_indep = Q_solar + Q_internal                     # [W] T-independent gains
        Q_drive = Qdrive_walls + G_env * T_ext_K + Q_indep # [W]  (temperatures in K)
        D       = a + G_tot

        T_old_K  = T_room_K
        T_free_K = (a * T_old_K + Q_drive) / D             # room temperature WITHOUT HVAC

        # ── 8. Ideal thermostat — solved implicitly (no overshoot, no cycling) ─
        # If the free-floating temperature would leave the comfort band, inject
        # exactly the power needed to land ON the setpoint (capped at max power).
        Q_HVAC = 0.0
        if self.hvac is not None:
            T_heat_K = self.hvac.T_heat_set + 273.15
            T_cool_K = self.hvac.T_cool_set + 273.15

            if T_free_K < T_heat_K:
                Q_HVAC = float(np.clip(T_heat_K * D - (a * T_old_K + Q_drive),
                                       0.0, self.hvac.max_power_heat))
                dE_heat = Q_HVAC / self.hvac.efficiency_heat * dt / 3.6e6
                self.E_heat_kWh += dE_heat
                if self.re2020_ev is not None:
                    self.re2020_ev.add_heating_kWh(dE_heat, self.hvac.energy_carrier)  # final
                    self.re2020_ev.add_heating_need_kWh(Q_HVAC * dt / 3.6e6)           # need (BBio)

            elif T_free_K > T_cool_K:
                Q_HVAC = float(np.clip(T_cool_K * D - (a * T_old_K + Q_drive),
                                       -self.hvac.max_power_cool, 0.0))
                dE_cool = abs(Q_HVAC) / self.hvac.efficiency_cool * dt / 3.6e6
                self.E_cool_kWh += dE_cool
                if self.re2020_ev is not None:
                    self.re2020_ev.add_cooling_kWh(dE_cool, self.hvac.energy_carrier)  # final
                    self.re2020_ev.add_cooling_need_kWh(abs(Q_HVAC) * dt / 3.6e6)      # need (BBio)

        # ── 9. New room temperature (implicit solution) ────────────────────────
        T_new_K     = (a * T_old_K + Q_drive + Q_HVAC) / D
        self.T_room = float(np.clip(T_new_K - 273.15, -10.0, 50.0))

        # ── 10. Update room humidity ───────────────────────────────────────────
        G_total = G_walls + G_vent + G_internal
        dPv     = G_total * lib.Rv * T_room_K * dt / self.volume
        Pv_new  = max(Pv_room + dPv, 0.0)
        Psat_new= lib.Psat(self.T_room + 273.15, self.RH_room)
        self.RH_room = float(np.clip(Pv_new / Psat_new, 0.05, 1.0))

        # ── 11. Step walls with updated room state as interior BC ──────────────
        T_room_new_K = self.T_room + 273.15
        for wall_obj, cfg in zip(self.walls, self.wall_configs):
            wall_obj.step(T_ext_K, RH_ext, T_room_new_K, self.RH_room, dt)

        # ── 12. RE2020 discomfort ──────────────────────────────────────────────
        if self.re2020_ev is not None:
            self.re2020_ev.add_discomfort_hours(
                np.array([self.T_room]), dt)

        # ── Store results ──────────────────────────────────────────────────────
        self.StockT_room.append(self.T_room)
        self.StockRH_room.append(self.RH_room)
        self.StockQ_HVAC.append(Q_HVAC)
        for cfg, wall_obj in zip(self.wall_configs, self.walls):
            self.StockT_walls[cfg.name].append(float(wall_obj.T[-1, 0]) - 273.15)

        self.t_sim += dt
        self.step_count += 1

    # ── Full simulation loop ───────────────────────────────────────────────────

    def run(
        self,
        time_bc  : np.ndarray,
        Text_bc  : np.ndarray,
        RHext_bc : np.ndarray,
        dt       : float = 3600.0,
        verbose  : bool  = True,
    ):
        """
        Run the full simulation over the climate boundary conditions.

        Parameters
        ----------
        time_bc  : array   Time vector [s].
        Text_bc  : array   Exterior temperature [°C].
        RHext_bc : array   Exterior relative humidity [-].
        dt       : float   Time step [s].  Default 3600 s (1 h).
        verbose  : bool    Print progress bar.
        """
        self.dt = dt                 # remember the step so outputs can scale correctly
        t_tot   = int(time_bc.max())
        n_steps = int(t_tot / dt)

        def _interp(t):
            return (
                float(np.interp(t, time_bc, Text_bc)),
                float(np.interp(t, time_bc, RHext_bc)),
            )

        def _update_progress(title, progress):
            length = 20
            block  = int(round(length * progress))
            msg    = "\r{0}: [{1}] {2}%".format(
                        title,
                        "#" * block + "-" * (length - block),
                        round(progress * 100, 1))
            if progress >= 1.0:
                msg += " DONE\r\n"
            sys.stdout.write(msg)
            sys.stdout.flush()

        if verbose:
            print(f"\nRoom simulation: {n_steps} steps "
                  f"({t_tot / 86400:.0f} days), dt={dt:.0f} s")
        t_start = time.process_time()

        for step in range(n_steps):
            t_sim     = step * dt
            T_e, RH_e = _interp(t_sim + dt)
            self.step(T_e, RH_e, dt)

            if verbose:
                _update_progress("Simulation", (step + 1) / n_steps)

            # NaN guard
            if np.isnan(self.T_room) or np.isnan(self.RH_room):
                print(f"\n[!] NaN in room state at step {step+1} — stopping.")
                break

        elapsed = time.process_time() - t_start
        if verbose:
            print(f"Completed in {elapsed:.1f} s")

        return np.array(self.StockT_room), np.array(self.StockRH_room)

    # ── Summary ───────────────────────────────────────────────────────────────

    def print_summary(self):
        """Print a quick energy and comfort summary."""
        T_arr  = np.array(self.StockT_room)
        RH_arr = np.array(self.StockRH_room)
        Q_arr  = np.array(self.StockQ_HVAC)

        print("\n" + "─"*50)
        print("  ROOM SIMULATION SUMMARY")
        print("─"*50)
        print(f"  Walls     : {len(self.wall_configs)}")
        for cfg, layer in zip(self.wall_configs, self.layers):
            print(f"    · {cfg.name:<20}  U = {layer.U_value():.2f} W/(m²K)"
                  f"  A = {cfg.area:.1f} m²  [{cfg.orientation}]")
        print(f"  Windows   : {len(self.window_configs)}")
        for w in self.window_configs:
            print(f"    · {w.name:<20}  U = {w.U_value:.2f}  g = {w.g_value}  [{w.orientation}]")
        print(f"  Bridges   : {len(self.bridge_configs)}")
        print("─"*50)
        print(f"  Room temperature  :  mean {T_arr.mean():.1f}°C  "
              f"min {T_arr.min():.1f}°C  max {T_arr.max():.1f}°C")
        print(f"  Room RH           :  mean {RH_arr.mean()*100:.1f}%  "
              f"min {RH_arr.min()*100:.1f}%  max {RH_arr.max()*100:.1f}%")
        print(f"  Heating energy    :  {self.E_heat_kWh:.1f} kWh")
        print(f"  Cooling energy    :  {self.E_cool_kWh:.1f} kWh")
        dt_h = (self.dt or 3600) / 3600.0       # hours per step
        DH = float(np.maximum(T_arr - 28.0, 0.0).sum()) * dt_h   # [°C·h]
        print(f"  DH discomfort     :  {DH:.0f} °C·h  (RE2020 limit 1250)")
        print("─"*50 + "\n")
