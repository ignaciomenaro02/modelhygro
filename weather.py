# -*- coding: utf-8 -*-
"""
weather.py
==========
Loader for a real outdoor climate file (French "données climatiques
prospectives" format) so the room simulation is driven by actual weather
instead of laboratory wall-test data.

THE FILE FORMAT
---------------
The official French prospective-climate CSV is *semicolon-separated* and
*hourly over a full year* (8760 rows + 1 header). Its columns are:

    STATION LATITUDE LONGITUDE ALTITUDE MOIS JOUR HEURE
    T2m  T-1m  Hur  Hus  RR  FF  DD  Tc  Tr  Hs  Os  Os_2
    Rgh  Rdn  Rdi  Ne  Ps

The ones we use here:
    T2m  : outdoor air temperature at 2 m            [°C]
    Hur  : outdoor relative humidity                 [%]
    Rgh  : global horizontal solar irradiance        [W/m²]   (optional)
    MOIS, JOUR, HEURE : month / day / hour            (for reference)

WHAT THIS RETURNS
-----------------
A `Weather` object whose three core arrays plug straight into
RoomSimulation.run():

    w = load_weather_csv("donnees-climatiques-prospectives-france-2c_macon.csv")
    sim.run(time_bc=w.time_s, Text_bc=w.T_ext, RHext_bc=w.RH_ext, dt=600)

Because the file is hourly but the simulation step can be smaller (e.g.
10 min), RoomSimulation linearly interpolates the boundary conditions
between hourly points — no resampling is needed here.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class Weather:
    """Parsed weather data, ready for RoomSimulation."""
    time_s : np.ndarray   # time vector from start of file [s]
    T_ext  : np.ndarray   # outdoor air temperature        [°C]
    RH_ext : np.ndarray   # outdoor relative humidity       [-]   (0–1)
    Rgh    : np.ndarray   # global horizontal irradiance    [W/m²]
    station: str          # station name (e.g. "MACON")
    lat    : float        # latitude  [°N]
    lon    : float        # longitude [°E]
    alt    : float        # altitude  [m]
    df     : pd.DataFrame  # the full raw table, in case other columns are needed

    @property
    def t_tot(self) -> int:
        """Total duration covered by the file [s]."""
        return int(self.time_s[-1])

    @property
    def n_hours(self) -> int:
        return len(self.time_s)


def load_weather_csv(path: str, sep: str = ';') -> Weather:
    """
    Load a French prospective-climate CSV into a `Weather` object.

    Parameters
    ----------
    path : str   Path to the .csv file.
    sep  : str   Column separator (';' for the official format).

    Returns
    -------
    Weather
    """
    df = pd.read_csv(path, sep=sep)

    # The file is hourly and already in chronological order, so the time
    # axis is simply one point per hour. We build it in seconds so it
    # matches the units RoomSimulation expects.
    n        = len(df)
    time_s   = np.arange(n, dtype=float) * 3600.0      # [s]

    T_ext    = df['T2m'].to_numpy(dtype=float)         # [°C]
    RH_ext   = df['Hur'].to_numpy(dtype=float) / 100.0  # [%] → [-]

    # Solar irradiance is optional (used only if you later feed it to the
    # solar model). Missing column → zeros, so nothing breaks.
    Rgh = (df['Rgh'].to_numpy(dtype=float)
           if 'Rgh' in df.columns else np.zeros(n))

    return Weather(
        time_s  = time_s,
        T_ext   = T_ext,
        RH_ext  = RH_ext,
        Rgh     = Rgh,
        station = str(df['STATION'].iloc[0]) if 'STATION'  in df.columns else '?',
        lat     = float(df['LATITUDE'].iloc[0]) if 'LATITUDE'  in df.columns else np.nan,
        lon     = float(df['LONGITUDE'].iloc[0]) if 'LONGITUDE' in df.columns else np.nan,
        alt     = float(df['ALTITUDE'].iloc[0]) if 'ALTITUDE'  in df.columns else np.nan,
        df      = df,
    )


# Quick self-test:  python weather.py  donnees-climatiques-...csv
if __name__ == '__main__':
    import sys
    f = sys.argv[1] if len(sys.argv) > 1 \
        else 'donnees-climatiques-prospectives-france-2c_macon.csv'
    w = load_weather_csv(f)
    print(f"Station {w.station}  ({w.lat:.3f}N, {w.lon:.3f}E, {w.alt:.0f} m)")
    print(f"{w.n_hours} hourly points  ->  {w.t_tot/86400:.0f} days")
    print(f"T_ext : min {w.T_ext.min():.1f}  mean {w.T_ext.mean():.1f}  "
          f"max {w.T_ext.max():.1f} degC")
    print(f"RH_ext: min {w.RH_ext.min()*100:.0f}  mean {w.RH_ext.mean()*100:.0f}  "
          f"max {w.RH_ext.max()*100:.0f} %")
