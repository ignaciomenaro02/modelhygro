# -*- coding: utf-8 -*-
"""
heating.py
==========
Heating control following the RE2020 conventions (Th-BCE 2020 method) for dwellings.

Source
------
Arrêté of 4 August 2021, Annex III "Méthode de calcul détaillée Th-BCE 2020"
(heating/cooling seasons, pages 82-88) and the official conventional scenarios
workbook (sheets MI = single-family house and LC = collective dwelling, which are
identical for heating).

What is implemented
-------------------
1. Conventional scenario
   - setpoint 19 °C while the dwelling is occupied, 16 °C in reduced mode
   - Monday/Tuesday/Thursday/Friday: reduced 09h-17h; Wednesday: reduced 09h-13h;
     Saturday/Sunday: normal all day
   - one holiday week (4th week of December = days 358-364), reduced all week
   - the year starts on a Monday; hours are legal time
2. Heating season (the system may only run inside the season)
   - days 1-56    : allowed
   - days 57-182  : stops when the moving mean (28 days) of the heating need falls to
                    2 Wh/m² per hour or less, provided heating has been on for at least
                    7 days. It can restart ONCE if the cold-discomfort degree-hours of the
                    last 28 days reach 40 °C·h (scaled by the occupation of the window)
   - days 183-252 : forbidden
   - day 253 - end: starts when the cold-discomfort degree-hours reach the threshold,
                    then stays on until the end of the year
   The tests run every day at 09:00 on the previous 24 hours.
   In Th-D mode heating is never allowed during the adaptive-comfort period.

Simplifications
---------------
- Air temperature is used instead of the operative temperature, and the "zero system
  power" temperature is the one-step free-floating temperature of the room model.
- Legal time is taken as the simulation clock (no daylight-saving shift); the run is
  assumed to start at 00:00 with an hourly time step.
- Without spin-up, a run that starts after 1 January has moving windows that only contain the
  simulated days; with spin-up (simulation.py) the season state is inherited from the warm-up.
- The heating need is measured at an ideal emitter (no distribution losses).
"""

from collections import deque

import numpy as np

# Occupied (1 = normal setpoint) / reduced (0) pattern per hour of the day (00h..23h).
# A cell k of the official table covers the legal hour [k-1, k).
_MON_TUE_THU_FRI = "111111111" + "00000000" + "1111111"    # reduced 09h-17h
_WEDNESDAY       = "111111111" + "0000"     + "11111111111"  # reduced 09h-13h
_WEEKEND         = "1" * 24
PATTERN = [_MON_TUE_THU_FRI, _MON_TUE_THU_FRI, _WEDNESDAY, _MON_TUE_THU_FRI,
           _MON_TUE_THU_FRI, _WEEKEND, _WEEKEND]            # Monday .. Sunday

NBH_OCC_REF = sum(p.count("1") for p in PATTERN)            # occupied hours in a reference week (132)
HOLIDAY_DAYS = range(358, 365)                              # 4th week of December

# Season boundaries (day of the year, 1 = 1 January) and thresholds
LAST_FREE_DAY   = 56
LAST_STOP_DAY   = 182
FIRST_START_DAY = 253
THRESHOLD_START = 40.0     # [°C·h]   cold-discomfort degree-hours
THRESHOLD_STOP  = 2.0      # [Wh/m²]  mean hourly heating need per m²
WINDOW_DAYS     = 28
MAX_RESTARTS    = 1
ADAPTIVE_TRM    = 16.0     # [°C]     running-mean outdoor temperature bounding the adaptive-comfort period


def occupancy_series(start_doy, n_steps):
    """
    Boolean array [n_steps]: dwelling occupied (RE2020 scenario) at each hourly step.
    Same calendar conventions as RE2020Heating (year starts on a Monday, run starts at 00:00).
    Used for the occupied-hours-only summer-comfort indicator (DH).
    """
    occ = np.zeros(n_steps, dtype=bool)
    for s in range(n_steps):
        doy = (int(start_doy) - 1 + s // 24) % 365 + 1
        occ[s] = RE2020Heating.occupied(doy, (doy - 1) % 7, s % 24)
    return occ


class RE2020Heating:
    """
    Heating setpoint and season for an hourly simulation (see module docstring).

    Parameters
    ----------
    start_doy        : int    Day of the year of the first simulated day (1 = 1 January).
    floor_area       : float  Useful floor area [m²].
    setpoint         : float  Occupied setpoint [°C].
    setback          : float  Reduced setpoint [°C].
    adaptive_comfort : array of bool, optional
                       Indexed by day of the year - 1; True inside the adaptive-comfort
                       period. Pass it only in Th-D mode (heating is then never allowed
                       during that period).
    """

    def __init__(self, start_doy, floor_area, setpoint=19.0, setback=16.0, adaptive_comfort=None):
        self.start_doy = int(start_doy)
        self.area      = float(floor_area)
        self.setpoint  = float(setpoint)
        self.setback   = float(setback)
        self.adaptive  = adaptive_comfort

        self.on        = self.start_doy <= LAST_FREE_DAY     # initial authorization
        self.restarts  = 0
        self.days_on   = 0
        self.days_off  = 0

        # daily accumulators (window 09:00 → 09:00)
        self._cold_dh = 0.0
        self._occ_h   = 0
        self._need_wh = 0.0
        self._ready   = False
        self._win_disc = deque(maxlen=WINDOW_DAYS)    # (cold degree-hours, occupied hours)
        self._win_need = deque(maxlen=WINDOW_DAYS)    # heating need [Wh]

        self.day_authorized = {}                      # simulated day → season authorization

    def continue_from(self, previous):
        """
        Take over the season state of a controller that ended at midnight of the day before
        this run starts (spin-up run): authorization, restarts, day counters and the moving
        windows. The per-day authorization record stays empty.
        """
        self.on, self.restarts = previous.on, previous.restarts
        self.days_on, self.days_off = previous.days_on, previous.days_off
        self._cold_dh, self._occ_h, self._need_wh = previous._cold_dh, previous._occ_h, previous._need_wh
        self._ready = previous._ready
        self._win_disc = deque(previous._win_disc, maxlen=WINDOW_DAYS)
        self._win_need = deque(previous._win_need, maxlen=WINDOW_DAYS)

    # ── calendar ──────────────────────────────────────────────────────────────

    def _clock(self, step):
        day  = step // 24
        doy  = (self.start_doy - 1 + day) % 365 + 1
        return doy, step % 24, (doy - 1) % 7, day

    @staticmethod
    def occupied(doy, weekday, hour):
        if doy in HOLIDAY_DAYS:
            return False
        return PATTERN[weekday][hour] == "1"

    # ── interface used by RoomSimulation ──────────────────────────────────────

    def heating_setpoint(self, step):
        """Setpoint [°C] for this step, or None when heating is not authorized."""
        doy, hour, weekday, day = self._clock(step)
        if hour == 9:
            self._daily_update(doy)
        blocked = self.adaptive is not None and bool(self.adaptive[doy - 1])
        authorized = self.on and not blocked
        self.day_authorized[day] = authorized
        if not authorized:
            return None
        return self.setpoint if self.occupied(doy, weekday, hour) else self.setback

    def record(self, step, T_free_C, Q_heat_W, dt=3600.0):
        """Feed the accumulators with the room state of this step."""
        doy, hour, weekday, _ = self._clock(step)
        if self.occupied(doy, weekday, hour):
            self._occ_h   += 1
            self._cold_dh += max(0.0, self.setpoint - T_free_C) * dt / 3600.0
        self._need_wh += Q_heat_W * dt / 3600.0

    # ── season logic ──────────────────────────────────────────────────────────

    def _daily_update(self, doy):
        if self._ready:
            if doy == FIRST_START_DAY and not self.on:
                self._win_disc.clear()
            self._win_disc.append((self._cold_dh, self._occ_h))
            self._win_need.append(self._need_wh)
            cold_dh   = sum(x[0] for x in self._win_disc)
            occ_hours = sum(x[1] for x in self._win_disc)
            factor    = max(0.5, occ_hours / (4.0 * NBH_OCC_REF))
            need_mean = sum(self._win_need) / (24.0 * len(self._win_need)) / self.area
            self._decide(doy, cold_dh, THRESHOLD_START * factor, need_mean)
        self._cold_dh, self._occ_h, self._need_wh = 0.0, 0, 0.0
        self._ready = True

    def _decide(self, doy, cold_dh, threshold, need_mean):
        was_on = self.on
        if doy <= LAST_FREE_DAY:
            self.on = True
        elif doy <= LAST_STOP_DAY:
            if self.on:
                if need_mean <= THRESHOLD_STOP and self.days_on >= 7:
                    self.on = False                       # stop: on for at least a week, low need
            elif (self.days_off >= 2 and cold_dh >= threshold
                    and self.restarts < MAX_RESTARTS):
                self.on = True                            # the single allowed restart
                self.restarts += 1
        elif doy < FIRST_START_DAY:
            self.on = False
        elif not self.on and cold_dh >= threshold:
            self.on = True                                # autumn start, stays on until year end

        if self.on and not was_on:                        # (re)started: need window starts again
            self.days_on = 0
            self._win_need.clear()
        elif was_on and not self.on:                      # stopped: discomfort window starts again
            self.days_off = 0
            self._win_disc.clear()
        if self.on:
            self.days_on += 1
        else:
            self.days_off += 1

    # ── results ───────────────────────────────────────────────────────────────

    def authorized_days(self, n_days):
        """Boolean array [n_days]: heating authorized (season open) on each simulated day."""
        return [bool(self.day_authorized.get(d, False)) for d in range(n_days)]
