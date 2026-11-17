# -*- coding: utf-8 -*-
"""
solar.py
========
Solar irradiance on the walls, the roof and the windows, from the RE2020 weather file.

The weather file already gives, hour by hour, the sun position and the measured direct normal
and diffuse horizontal irradiances, so nothing is modelled here except the projection on a
surface of given orientation and tilt:

    I = I_beam + I_diffuse + I_ground
      = DNI·max(cos θ, 0) + I_dif·(1 + cos β)/2 + GHI·ρ_ground·(1 − cos β)/2

with θ the angle between the sun and the surface normal, β the tilt (90° = wall, 0° = roof)
and ρ_ground the ground albedo. The diffuse sky is isotropic (Liu & Jordan 1960).
Each hourly value is held constant over its hour.
"""

import numpy as np


# Surface orientation -> azimuth of its normal, from South, positive to the West [°]
_AZIMUTH_DEG = {
    "S":    0, "SW":  45, "W":   90, "NW":  135,
    "N":  180, "NE": -135, "E": -90, "SE": -45,
    "roof": 0, "floor": 0,
}


class SolarCalculator:
    """
    Irradiance on tilted surfaces from a Weather object (weather.load_re2020_weather).

    weather       : Weather   hourly sun position, dirN and diff, index 0 = 1 January 00:00
    start_doy     : int       day of year of the simulation start (1 = 1 January)
    ground_albedo : float     ground reflectance (0.2 = grass)
    """

    def __init__(self, weather, start_doy: int = 1, ground_albedo: float = 0.2):
        self._wx       = weather
        self.start_doy = start_doy
        self.rho_g     = ground_albedo

    def _hour_index(self, t_sim: float) -> int:
        """Row of the weather file for the simulation time t_sim [s]."""
        t_abs = (self.start_doy - 1) * 86400.0 + t_sim
        return int(t_abs // 3600.0) % len(self._wx.dirN)

    def irradiance(self, t_sim: float, orientation: str = "S", tilt: float = 90.0) -> float:
        """Total irradiance on a surface [W/m²] (orientation 'N', 'S', ..., tilt in degrees)."""
        if orientation == "floor":
            return 0.0

        i    = self._hour_index(t_sim)
        alt  = np.radians(self._wx.sun_alt_deg[i])
        az   = np.radians(self._wx.sun_az_deg[i])
        DNI  = self._wx.dirN[i]
        Id_h = self._wx.diff[i]

        # Night. The file can keep a little diffuse light at dawn and dusk with the sun at 0°,
        # so the diffuse part is still counted then.
        if alt <= 0 and Id_h <= 0.0:
            return 0.0

        Ib_h = DNI * np.sin(alt) if alt > 0 else 0.0           # direct on the horizontal
        beta = np.radians(tilt)
        az_s = np.radians(_AZIMUTH_DEG.get(orientation, 0))
        cos_theta = np.sin(alt) * np.cos(beta) + np.cos(alt) * np.sin(beta) * np.cos(az - az_s)

        I_beam   = DNI * max(float(cos_theta), 0.0) if alt > 0 else 0.0
        I_dif    = Id_h * (1.0 + np.cos(beta)) / 2.0
        I_ground = (Ib_h + Id_h) * self.rho_g * (1.0 - np.cos(beta)) / 2.0
        return float(I_beam + I_dif + I_ground)

    def solar_gains_window(self, t_sim: float, window) -> float:
        """Solar heat entering through a window [W] (WindowConfig: area, orientation, g, shading, frame)."""
        I = self.irradiance(t_sim, window.orientation, getattr(window, "tilt", 90.0))
        return float(window.g_value * (1.0 - window.shading) * window.frame_factor * window.area * I)
