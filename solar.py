# -*- coding: utf-8 -*-
"""
solar.py
========
Solar irradiance calculator for building simulation.

Computes hourly solar gains on arbitrarily oriented surfaces
from first principles (no external weather file needed for solar).

Method
------
1. Solar position: declination, hour angle → altitude & azimuth.
2. Clear-sky direct normal irradiance (DNI) — simplified Iqbal model C.
3. Isotropic diffuse model for tilted surfaces (Liu & Jordan).
4. Ground-reflected irradiance.

For each surface orientation the result is:
    I_total [W/m²] = I_beam + I_diffuse + I_ground_reflected

Usage
-----
    sol = SolarCalculator(latitude=46.3, longitude=4.8, start_doy=1)
    I = sol.irradiance(t_sim_s, orientation='S', tilt=90.0)

References
----------
- Iqbal M. (1983) An Introduction to Solar Radiation.
- Liu B.Y.H. & Jordan R.C. (1960) Solar Energy 4(3):1-19.
- ISO 52010-1:2017 External conditions for energy calculation.
"""

import numpy as np


# Orientation → azimuth angle from South, positive West [degrees]
_AZIMUTH_DEG = {
    'S':   0,   'SW':  45,  'W':   90,  'NW':  135,
    'N':  180,  'NE': -135, 'E':  -90,  'SE':  -45,
    'roof': 0,  'floor': 0,
}


class SolarCalculator:
    """
    Hourly solar irradiance on tilted surfaces.

    Parameters
    ----------
    latitude   : float   Site latitude [°N].  Default: Mâcon, France (46.3°N).
    longitude  : float   Site longitude [°E]. Default: Mâcon (4.8°E).
    start_doy  : int     Day-of-year of simulation start (1 = Jan 1).
    ground_albedo : float  Ground reflectivity [-].  Typical 0.2 (grass).
    cloud_factor  : float  Cloud attenuation of DNI [0=clear sky, 1=overcast].
                           0.0 gives a clear-sky upper bound.
                           Use 0.3–0.5 for average European climate.
    timezone_offset: float UTC offset [h].  Default 1 (CET, France).
    """

    def __init__(
        self,
        latitude       : float = 46.3,
        longitude      : float = 4.8,
        start_doy      : int   = 1,
        ground_albedo  : float = 0.2,
        cloud_factor   : float = 0.3,
        timezone_offset: float = 1.0,
    ):
        self.lat_rad        = np.radians(latitude)
        self.lon_deg        = longitude
        self.start_doy      = start_doy
        self.rho_g          = ground_albedo
        self.cloud          = cloud_factor
        self.tz             = timezone_offset

    # ── Solar position ─────────────────────────────────────────────────────────

    def _doy(self, t_sim: float) -> float:
        """Fractional day of year at simulation time t_sim [s]."""
        return self.start_doy + t_sim / 86400.0

    def _declination(self, doy: float) -> float:
        """Solar declination [rad] — Spencer formula."""
        B = 2 * np.pi * (doy - 1) / 365
        return np.radians(
            0.006918 - 0.399912 * np.cos(B) + 0.070257 * np.sin(B)
            - 0.006758 * np.cos(2*B) + 0.000907 * np.sin(2*B)
        )

    def _equation_of_time(self, doy: float) -> float:
        """Equation of time [hours] — Spencer formula."""
        B = 2 * np.pi * (doy - 1) / 365
        return (0.0000075 + 0.001868 * np.cos(B) - 0.032077 * np.sin(B)
                - 0.014615 * np.cos(2*B) - 0.04089 * np.sin(2*B)) * 229.18 / 60.0

    def position(self, t_sim: float):
        """
        Solar altitude and azimuth at time t_sim [s] from simulation start.

        Returns
        -------
        altitude : float   Solar altitude above horizon [rad].  < 0 = night.
        azimuth  : float   Solar azimuth from South, positive West [rad].
        """
        doy    = self._doy(t_sim)
        hour_local = (t_sim % 86400) / 3600.0   # local standard time [h]
        delta  = self._declination(doy)
        ET     = self._equation_of_time(doy)

        # Solar time
        t_solar = hour_local + (self.lon_deg - 15 * self.tz) / 15.0 + ET

        # Hour angle [rad]
        omega = np.radians(15.0 * (t_solar - 12.0))

        # Altitude
        sin_alt = (np.sin(self.lat_rad) * np.sin(delta)
                   + np.cos(self.lat_rad) * np.cos(delta) * np.cos(omega))
        altitude = float(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))

        # Azimuth (from South, + = West)
        cos_az = ((np.sin(altitude) * np.sin(self.lat_rad) - np.sin(delta))
                  / (np.cos(altitude) * np.cos(self.lat_rad) + 1e-12))
        azimuth = float(np.arccos(np.clip(cos_az, -1.0, 1.0)))
        if omega > 0:       # afternoon → west of south
            azimuth = -azimuth

        return altitude, azimuth

    # ── Clear-sky irradiance ───────────────────────────────────────────────────

    def _extra_terrestrial(self, doy: float) -> float:
        """Extraterrestrial irradiance I0 [W/m²]."""
        return 1361.0 * (1 + 0.033 * np.cos(2 * np.pi * doy / 365))

    def _air_mass(self, altitude_rad: float) -> float:
        """Optical air mass (Kasten & Young 1989)."""
        if altitude_rad <= 0:
            return 40.0
        alt_deg = np.degrees(altitude_rad)
        return min(40.0,
                   1.0 / (np.sin(altitude_rad)
                          + 0.50572 * (alt_deg + 6.07995)**(-1.6364)))

    def horizontal_irradiance(self, t_sim: float):
        """
        Clear-sky global horizontal irradiance, attenuated by cloud_factor.

        Returns
        -------
        Ib_h : float   Beam (direct) horizontal irradiance [W/m²].
        Id_h : float   Diffuse horizontal irradiance [W/m²].
        """
        alt, _ = self.position(t_sim)
        if alt <= 0:
            return 0.0, 0.0

        doy = self._doy(t_sim)
        I0  = self._extra_terrestrial(doy)
        am  = self._air_mass(alt)

        tau_b = 0.7   # beam atmospheric transmittance (clear sky)
        tau_d = 0.10  # diffuse fraction (clear sky)

        DNI  = I0 * tau_b**am * (1.0 - self.cloud)
        Ib_h = DNI * np.sin(alt)
        Id_h = I0 * tau_d * np.sin(alt) * (1.0 + self.cloud)

        return float(Ib_h), float(Id_h)

    # ── Irradiance on tilted surface ───────────────────────────────────────────

    def irradiance(
        self,
        t_sim      : float,
        orientation: str   = 'S',
        tilt       : float = 90.0,
    ) -> float:
        """
        Total solar irradiance incident on a tilted surface [W/m²].

        Parameters
        ----------
        t_sim       : float   Simulation time [s] from start.
        orientation : str     Surface orientation code ('N','S','E','W',…,'roof').
        tilt        : float   Surface tilt from horizontal [°].
                              90 = vertical, 0 = horizontal (roof).

        Returns
        -------
        I_total : float   Total irradiance [W/m²].  0 at night.
        """
        if orientation == 'floor':
            return 0.0

        alt, az_sun = self.position(t_sim)
        Ib_h, Id_h  = self.horizontal_irradiance(t_sim)

        if alt <= 0:
            return 0.0

        tilt_rad   = np.radians(tilt)
        az_surf    = np.radians(_AZIMUTH_DEG.get(orientation, 0))

        # Angle of incidence on tilted surface
        cos_theta = (np.sin(alt) * np.cos(tilt_rad)
                     + np.cos(alt) * np.sin(tilt_rad) * np.cos(az_sun - az_surf))

        # DNI from air-mass model
        doy = self._doy(t_sim)
        I0  = self._extra_terrestrial(doy)
        am  = self._air_mass(alt)
        DNI = I0 * 0.7**am * (1.0 - self.cloud)

        # Beam on tilted surface
        Ib_surf = DNI * max(float(cos_theta), 0.0)

        # Diffuse (isotropic Liu-Jordan model)
        Id_surf = Id_h * (1.0 + np.cos(tilt_rad)) / 2.0

        # Ground-reflected
        GHI     = Ib_h + Id_h
        Ig_surf = GHI * self.rho_g * (1.0 - np.cos(tilt_rad)) / 2.0

        return float(Ib_surf + Id_surf + Ig_surf)

    def solar_gains_window(
        self,
        t_sim      : float,
        window,             # WindowConfig
    ) -> float:
        """
        Solar heat gain through a window [W].

        Parameters
        ----------
        window : WindowConfig
            Must have: area, orientation, tilt (default 90), g_value,
            shading, frame_factor.
        """
        tilt = getattr(window, 'tilt', 90.0)
        I    = self.irradiance(t_sim, window.orientation, tilt)
        gain = (window.g_value
                * (1.0 - window.shading)
                * window.frame_factor
                * window.area
                * I)
        return float(gain)

    def solar_gains_opaque(
        self,
        t_sim      : float,
        wall_cfg,           # WallConfig
        absorptance: float = 0.6,
    ) -> float:
        """
        Solar heat gain absorbed by an opaque wall surface [W].
        (Radiative flux enters the wall as an additional exterior BC term.)

        absorptance : float   Solar absorptance of exterior finish.
                              Light colour ~0.3, dark colour ~0.9.
        """
        I    = self.irradiance(t_sim, wall_cfg.orientation, wall_cfg.tilt)
        gain = absorptance * wall_cfg.area * I
        return float(gain)
