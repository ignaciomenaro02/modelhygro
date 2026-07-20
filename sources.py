# -*- coding: utf-8 -*-
"""
sources.py
==========
Internal heat/moisture sources and HVAC system configuration.

Classes
-------
OccupantConfig      Sensible + latent gains from occupants.
EquipmentConfig     Plug loads and appliances.
LightingConfig      Artificial lighting gains.
VentilationConfig   Mechanical ventilation (with optional heat recovery).
HVACConfig          Heating/cooling setpoints and system parameters.

All schedule arrays must have exactly n_steps values (one per time step),
or be None (constant operation assumed).
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Occupants
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class OccupantConfig:
    """
    Internal gains from building occupants.

    Parameters
    ----------
    n_occupants        : float   Average number of occupants [-].
    sensible_per_person: float   Sensible heat gain [W/person].
                                 Typical: 70 W (sedentary), 120 W (light activity).
    latent_per_person  : float   Latent (moisture) heat gain [W/person].
                                 Typical: 60 W → ~0.04 kg/h water vapour per person.
    schedule           : list    Hourly occupation factor [0–1] over the full
                                 simulation (length = n_steps).  None = always full.

    Moisture production
    -------------------
    Water vapour mass flow from occupants:
        G_occ [kg/s] = n_occ × latent_per_person / Lv
    where Lv = 2.5e6 J/kg (latent heat of evaporation).
    """
    n_occupants         : float         = 2.0
    sensible_per_person : float         = 80.0    # [W/person]
    latent_per_person   : float         = 60.0    # [W/person]
    schedule            : Optional[List[float]] = None   # [0–1] per step

    def sensible_at(self, step: int) -> float:
        """Sensible gain [W] at simulation step `step`."""
        factor = self.schedule[step] if self.schedule is not None else 1.0
        return self.n_occupants * self.sensible_per_person * factor

    def moisture_at(self, step: int) -> float:
        """Moisture vapour production [kg/s] at step `step`."""
        Lv = 2.5e6  # J/kg
        factor = self.schedule[step] if self.schedule is not None else 1.0
        return self.n_occupants * self.latent_per_person * factor / Lv


# ══════════════════════════════════════════════════════════════════════════════
# Equipment (plug loads)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EquipmentConfig:
    """
    Heat gains from electrical appliances (computers, TV, cooking, etc.).

    Parameters
    ----------
    power       : float   Total installed power [W].
    usage_factor: float   Fraction that becomes heat gain [-].  Default 1.0.
    schedule    : list    Hourly usage factor [0–1].  None = constant.

    Typical values
    --------------
    Residential kitchen + appliances : 200–400 W
    Office workstation               : 100–200 W
    """
    power        : float                    = 200.0   # [W]
    usage_factor : float                    = 1.0
    schedule     : Optional[List[float]]    = None

    def heat_at(self, step: int) -> float:
        """Sensible heat gain [W] at step `step`."""
        factor = self.schedule[step] if self.schedule is not None else 1.0
        return self.power * self.usage_factor * factor


# ══════════════════════════════════════════════════════════════════════════════
# Lighting
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LightingConfig:
    """
    Artificial lighting heat gains.

    Parameters
    ----------
    power_density : float   Installed power density [W/m²].
    floor_area    : float   Zone floor area [m²].
    schedule      : list    Hourly usage factor [0–1].

    Typical values (EN 15193)
    --------------------------
    LED residential : 3–6 W/m²
    LED office      : 6–12 W/m²
    """
    power_density : float                   = 5.0    # [W/m²]
    floor_area    : float                   = 50.0   # [m²]
    schedule      : Optional[List[float]]   = None

    def heat_at(self, step: int) -> float:
        factor = self.schedule[step] if self.schedule is not None else 1.0
        return self.power_density * self.floor_area * factor


# ══════════════════════════════════════════════════════════════════════════════
# Ventilation
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class VentilationConfig:
    """
    Mechanical ventilation system.

    Parameters
    ----------
    n_ach           : float   Air-change rate [1/h].
                              Minimum hygiene: 0.5 for residential (RT2012/RE2020).
    hrv_efficiency  : float   Heat-recovery ventilator (VMC double flux) efficiency [-].
                              0.0 = simple exhaust (VMC simple flux)
                              0.85 = high-efficiency HRV
    moisture_recovery: float  Moisture recovery efficiency of the HRV [-].
                              0.0 for simple flux, 0.5–0.7 for enthalpy exchangers.
    air_velocity    : float   Mean indoor air velocity [m/s] — affects interior
                              convective coefficients.  Used for comfort assessment.
    schedule        : list    Hourly flow-rate factor [0–1].

    Note: total infiltration = n_ach + Σ opening.ach_contribution
    """
    n_ach             : float                   = 0.5
    hrv_efficiency    : float                   = 0.0
    moisture_recovery : float                   = 0.0
    air_velocity      : float                   = 0.1   # [m/s] — indoor air speed
    schedule          : Optional[List[float]]   = None

    def flow_at(self, step: int) -> float:
        """Effective air-change rate [1/h] at step `step`."""
        factor = self.schedule[step] if self.schedule is not None else 1.0
        return self.n_ach * factor


# ══════════════════════════════════════════════════════════════════════════════
# HVAC system
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class HVACConfig:
    """
    Heating and cooling system.

    Parameters
    ----------
    T_heat_set     : float   Heating setpoint [°C].  Below this → heating on.
    T_cool_set     : float   Cooling setpoint [°C].  Above this → cooling on.
    efficiency_heat: float   Heating system efficiency (COP for heat pump,
                             η for boiler).  Use 1.0 for ideal electric heater.
    efficiency_cool: float   Cooling COP.  Typical: 2.5–4.5.
    max_power_heat : float   Maximum heating power [W].
    max_power_cool : float   Maximum cooling power [W].
    energy_carrier : str     'electricity' or 'gas' — used for primary energy (RE2020).
    f_ep           : float   Primary energy conversion factor.
                             RE2020: electricity = 2.3, gas = 1.0.

    Schedules
    ---------
    heating_schedule / cooling_schedule: list of 0/1 per step.
        Allows setback temperatures (e.g. 16°C at night) by multiplying
        the setpoint. None = system always available.
    """
    T_heat_set      : float  = 20.0
    T_cool_set      : float  = 26.0
    efficiency_heat : float  = 1.0     # COP or η
    efficiency_cool : float  = 3.0     # COP
    max_power_heat  : float  = 5000.0  # [W]
    max_power_cool  : float  = 5000.0  # [W]
    energy_carrier  : str    = 'electricity'
    f_ep            : float  = 2.3     # RE2020 primary energy factor


def build_daily_schedule(
    hours_on: list,
    n_steps: int,
    dt: int = 3600,
    value_on: float = 1.0,
    value_off: float = 0.0,
) -> List[float]:
    """
    Build an hourly schedule repeated daily over the full simulation.

    Parameters
    ----------
    hours_on  : list of int   Hours (0–23) when the schedule is active.
    n_steps   : int           Total number of simulation steps.
    dt        : int           Time step [s].  Default 3600 (1 hour).
    value_on  : float         Value during active hours.
    value_off : float         Value during inactive hours.

    Example
    -------
    # Occupants present 8h–22h:
    sched = build_daily_schedule(hours_on=list(range(8, 22)), n_steps=8760)
    """
    hours_per_step = dt / 3600
    schedule = []
    for step in range(n_steps):
        hour = int((step * hours_per_step) % 24)
        schedule.append(value_on if hour in hours_on else value_off)
    return schedule
