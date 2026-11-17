# -*- coding: utf-8 -*-
"""
sources.py
==========
What happens inside the room besides the walls: occupants, appliances, lighting, ventilation
and the heating/cooling system.

Schedules are lists with one factor (0-1) per simulation step; None means "always on".
"""

from dataclasses import dataclass
from typing import List, Optional

LV = 2.5e6   # latent heat of evaporation [J/kg]


def _factor(schedule, step):
    return schedule[step] if schedule is not None else 1.0


@dataclass
class OccupantConfig:
    """
    Occupants: sensible heat, and water vapour given as a latent power.
    The vapour flow is latent_per_person / Lv (60 W ≈ 86 g/h per person).
    """
    n_occupants         : float = 2.0
    sensible_per_person : float = 80.0    # [W]
    latent_per_person   : float = 60.0    # [W]
    schedule            : Optional[List[float]] = None

    def sensible_at(self, step: int) -> float:
        """Sensible heat [W]."""
        return self.n_occupants * self.sensible_per_person * _factor(self.schedule, step)

    def moisture_at(self, step: int) -> float:
        """Vapour production [kg/s]."""
        return self.n_occupants * self.latent_per_person * _factor(self.schedule, step) / LV


@dataclass
class EquipmentConfig:
    """Appliances: all their power ends up as heat in the room."""
    power    : float = 200.0              # [W]
    schedule : Optional[List[float]] = None

    def heat_at(self, step: int) -> float:
        return self.power * _factor(self.schedule, step)


@dataclass
class LightingConfig:
    """Lighting, as a power per floor area (LED housing: 3-6 W/m²)."""
    power_density : float = 5.0           # [W/m²]
    floor_area    : float = 50.0          # [m²]
    schedule      : Optional[List[float]] = None

    def heat_at(self, step: int) -> float:
        return self.power_density * self.floor_area * _factor(self.schedule, step)


@dataclass
class VentilationConfig:
    """
    Ventilation, as an air change rate n_ach [1/h] multiplied by the schedule.

    hrv_efficiency, moisture_recovery : heat and moisture recovery of a balanced system
                                        (0 for single-flow exhaust ventilation)

    Optional window opening by the occupants (overvent_ach > 0), added to n_ach, following the
    RE2020 manual window management (method Th-BCE, Annex III §5.13): only at the steps where
    overvent_window is True; the window opens when the room reaches T_open and closes below
    T_close; it stays shut when the outdoor air is not cooler than the room; the opening grows
    from 0 to 1 between T_ext_min and T_ext_full outdoors.
    """
    n_ach             : float = 0.5
    hrv_efficiency    : float = 0.0
    moisture_recovery : float = 0.0
    schedule          : Optional[List[float]] = None

    overvent_ach      : float = 0.0
    overvent_window   : Optional[List[bool]] = None
    T_open            : float = 23.0
    T_close           : float = 22.0
    T_ext_min         : float = 10.0
    T_ext_full        : float = 18.0
    _is_open          : bool  = False

    def flow_at(self, step: int, T_room: float = None, T_ext: float = None) -> float:
        """Air change rate [1/h]; T_room and T_ext [°C] are only needed for the window opening."""
        n = self.n_ach * _factor(self.schedule, step)
        if self.overvent_ach > 0 and self.overvent_window is not None and T_room is not None:
            if T_room >= self.T_open:
                self._is_open = True
            elif T_room < self.T_close:
                self._is_open = False
            if self.overvent_window[step] and self._is_open and T_ext < T_room:
                opening = min(1.0, max(0.0, (T_ext - self.T_ext_min) / (self.T_ext_full - self.T_ext_min)))
                n += self.overvent_ach * opening
        return n


@dataclass
class HVACConfig:
    """
    Ideal heating and cooling (see RoomSimulation._thermostat).

    efficiency_heat : 1 for a direct electric heater, the COP for a heat pump
    heating_control : optional controller (heating.RE2020Heating) that gives the heating setpoint
                      hour by hour, or None outside the heating season; it replaces T_heat_set
    """
    T_heat_set      : float = 20.0        # [°C]
    T_cool_set      : float = 26.0        # [°C]
    efficiency_heat : float = 1.0
    efficiency_cool : float = 3.0
    max_power_heat  : float = 5000.0      # [W]
    max_power_cool  : float = 5000.0      # [W]
    energy_carrier  : str   = "electricity"
    heating_control : Optional[object] = None


def build_daily_schedule(hours_on, n_steps, dt=3600, value_on=1.0, value_off=0.0):
    """Same schedule every day: value_on at the hours listed in hours_on (0-23), value_off otherwise."""
    return [value_on if int((s * dt / 3600) % 24) in hours_on else value_off for s in range(n_steps)]
