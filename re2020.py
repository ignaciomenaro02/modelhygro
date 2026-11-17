# -*- coding: utf-8 -*-
"""
re2020.py
=========
The RE2020 indicators the room model can compute, following the Th-BCE 2020 method
(arrêté of 4 August 2021, Annex III).

DH (summer comfort) [°C·h]
    Sum, over the occupied hours of the adaptive-comfort period, of the excess of the operative
    temperature over the hot-discomfort limit (§13.5.3, C_BAT_Confort_d'été):
        T_max = min( max(26, 0.33·T_rm + 20.8), 28 )   from 6 h to 22 h
        T_max = 26                                      during the night (dwellings)
    with T_rm the running mean of the outdoor temperature. The adaptive-comfort period runs from
    the first to the last day of the year with T_rm > 16 °C. Threshold: 1250 °C·h.
    It is regulatory in Th-D mode only (weather with a heat wave).
    A simpler version, the hours above a fixed 28 °C, is also kept.

Bbio, heating part [points]
    2 × B_ch, with B_ch the heating need per m² (§13.1.3: Bbio = 2·B_ch + 2·B_fr + 5·B_ecl).
    Cooling and lighting needs are not modelled here, so the Bbio is partial.

Primary energy of the heating [kWh_ep/m²]
    Final energy × 2.3 for electricity.
"""

import numpy as np

DH_MAX       = 1250.0   # [°C·h]
ADAPTIVE_TRM = 16.0     # [°C] T_rm bounding the adaptive-comfort period

_F_EP = {"electricity": 2.3, "gas": 1.0, "wood": 0.6, "district": 0.6}   # primary energy factors
_ALPHA_HEAT, _ALPHA_COOL = 2.0, 2.0                                      # Bbio weights

# Hot-discomfort limit: T_inc = max(T_cool_set, 0.33·T_rm + 18.8 + delta), delta = 2/3/4 K for
# category 1/2/3, never above T_cool_set + 2 K
_DELTA_CATEGORY   = {1: 2.0, 2: 3.0, 3: 4.0}
_DELTA_OP_MIN_MAX = 2.0


def hot_discomfort_limit(T_rm, hour, T_cool_set=26.0, category=1, dwelling=True):
    """
    Hourly hot-discomfort limit [°C]. `hour` is 0..23, the step covering [hour, hour + 1);
    its legal hour in the RE2020 tables is hour + 1, so the day period 6 < h_legal ≤ 22 is
    hour = 6..21.
    """
    T_rm  = np.asarray(T_rm, dtype=float)
    T_day = np.minimum(np.maximum(T_cool_set, 0.33 * T_rm + 18.8 + _DELTA_CATEGORY[category]),
                       T_cool_set + _DELTA_OP_MIN_MAX)
    if not dwelling:
        return T_day
    hour = np.asarray(hour) % 24
    return np.where((hour >= 6) & (hour <= 21), T_day, T_cool_set)


def adaptive_period(T_rm_day):
    """True for every day from the first to the last day of the year with T_rm > 16 °C."""
    T_rm_day = np.asarray(T_rm_day, dtype=float)
    out  = np.zeros(len(T_rm_day), dtype=bool)
    warm = np.where(T_rm_day > ADAPTIVE_TRM)[0]
    if len(warm):
        out[warm[0]:warm[-1] + 1] = True
    return out


def adaptive_discomfort(T_op, T_rm, occupied, hour, in_period, T_cool_set=26.0, category=1):
    """Hourly excess [K] of T_op over the limit, counted only when occupied and in the period."""
    keep  = np.asarray(occupied, dtype=bool) & np.asarray(in_period, dtype=bool)
    T_max = hot_discomfort_limit(T_rm, hour, T_cool_set, category)
    return np.maximum(np.asarray(T_op, dtype=float) - T_max, 0.0) * keep


class RE2020Evaluator:
    """
    Collects, during the run, the heating and cooling energy (RoomSimulation) and, at the end,
    the summer discomfort (simulation.py), then gives the indicators with report().

    Two energies are kept apart: the NEED (thermal energy given to the room, used for Bbio)
    and the FINAL energy (need / efficiency, converted to primary energy).
    """

    def __init__(self, floor_area, climate_zone="H1c"):
        self.A    = float(floor_area)
        self.zone = climate_zone
        self._carrier_heat = self._carrier_cool = "electricity"
        self._E_heat = self._E_cool = 0.0        # final energy [kWh]
        self._B_heat = self._B_cool = 0.0        # need [kWh]
        self._DH_28    = 0.0                     # hours above 28 °C [°C·h]
        self._DH_adapt = None                    # official DH [°C·h], None until computed

    def set_carrier(self, heating="electricity", cooling="electricity"):
        self._carrier_heat, self._carrier_cool = heating, cooling

    def add_heating_kWh(self, kWh_final, carrier=None):
        self._E_heat += kWh_final
        self._carrier_heat = carrier or self._carrier_heat

    def add_cooling_kWh(self, kWh_final, carrier=None):
        self._E_cool += kWh_final
        self._carrier_cool = carrier or self._carrier_cool

    def add_heating_need_kWh(self, kWh):
        self._B_heat += kWh

    def add_cooling_need_kWh(self, kWh):
        self._B_cool += kWh

    def add_discomfort_hours(self, T_room, dt=3600):
        """Simplified DH: degree-hours above a fixed 28 °C, all hours."""
        self._DH_28 += float(np.maximum(np.asarray(T_room, dtype=float) - 28.0, 0.0).sum()) * dt / 3600.0

    def add_adaptive_discomfort_hours(self, T_op, T_rm, occupied, hour, in_period,
                                      T_cool_set=26.0, category=1, dt=3600):
        """Official DH (see adaptive_discomfort), one value per step in every array."""
        excess = adaptive_discomfort(T_op, T_rm, occupied, hour, in_period, T_cool_set, category)
        self._DH_adapt = (self._DH_adapt or 0.0) + float(excess.sum()) * dt / 3600.0

    def report(self):
        """Indicators per m² of floor area."""
        B_heat = self._B_heat / self.A
        B_cool = self._B_cool / self.A
        DH     = self._DH_28 if self._DH_adapt is None else self._DH_adapt
        return {
            "floor_area_m2": self.A,
            "climate_zone" : self.zone,
            "B_heat"       : round(B_heat, 1),                                   # [kWh/m²]
            "B_cool"       : round(B_cool, 1),
            "BBio"         : round(_ALPHA_HEAT * B_heat + _ALPHA_COOL * B_cool, 1),
            "ep_heat"      : round(self._E_heat * _F_EP[self._carrier_heat] / self.A, 1),   # [kWh_ep/m²]
            "ep_cool"      : round(self._E_cool * _F_EP[self._carrier_cool] / self.A, 1),
            "DH_adaptive"  : None if self._DH_adapt is None else round(self._DH_adapt, 0),
            "DH_28C"       : round(self._DH_28, 0),
            "DH_max"       : DH_MAX,
            "compliant_DH" : DH <= DH_MAX,
        }
