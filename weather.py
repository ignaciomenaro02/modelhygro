# -*- coding: utf-8 -*-
"""
weather.py
==========
Reads the RE2020 conventional weather (scenarios_meteorologiques_th-bc_th-d_re2020.xlsx).

The workbook has one sheet per climate zone and mode, named '<zone>_<mode>' (for example
'H1c_Th-BC'), each with 8760 hourly rows starting on 1 January. The columns used are:

    te0     outdoor air temperature                    [°C]
    we0     humidity ratio                             [g/kg dry air]
    dirN    direct normal irradiance                   [W/m²]
    diff    diffuse horizontal irradiance              [W/m²]
    Gamma   solar altitude                             [°]
    Psi     solar azimuth, 0 = South, positive to West [°]

Mode Th-BC is a typical year (energy needs); mode Th-D contains a heat wave (summer comfort).
The licence of the workbook allows regulatory use only: it must not be redistributed.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd

import library as lib

RE2020_ZONES = ("H1a", "H1b", "H1c", "H2a", "H2b", "H2c", "H2d", "H3")
RE2020_MODES = ("Th-BC", "Th-D")


@dataclass
class Weather:
    """One year of hourly weather, index 0 = 1 January 00:00."""
    name        : str          # e.g. "RE2020 H1c Th-BC"
    T_ext       : np.ndarray   # outdoor air temperature       [°C]
    RH_ext      : np.ndarray   # outdoor relative humidity     [-]
    dirN        : np.ndarray   # direct normal irradiance      [W/m²]
    diff        : np.ndarray   # diffuse horizontal irradiance [W/m²]
    sun_alt_deg : np.ndarray   # solar altitude                [°]
    sun_az_deg  : np.ndarray   # solar azimuth from South, + West [°]


def load_re2020_weather(path: str, zone: str = "H1c", mode: str = "Th-BC") -> Weather:
    """Load the sheet '<zone>_<mode>' of the RE2020 weather workbook."""
    if zone not in RE2020_ZONES:
        raise ValueError(f"Unknown RE2020 zone '{zone}'. Choose from {RE2020_ZONES}.")
    if mode not in RE2020_MODES:
        raise ValueError(f"Unknown RE2020 mode '{mode}'. Choose from {RE2020_MODES}.")

    df = pd.read_excel(path, sheet_name=f"{zone}_{mode}")
    # Headers carry units and sometimes leading spaces ('dirN (W/m2)', '  we0'): keep the name only
    df.columns = [str(c).split("(")[0].strip() for c in df.columns]

    # The file gives the humidity ratio, not RH: convert through the vapour pressure.
    # A few winter hours come out slightly above 100 %, hence the clip.
    T_ext  = df["te0"].to_numpy(dtype=float)
    x      = df["we0"].to_numpy(dtype=float) / 1000.0            # g/kg -> kg/kg
    Pv     = lib.Patm * x / (0.622 + x)
    RH_ext = np.clip(Pv / lib.Psat(T_ext + 273.15, 0.0), 0.05, 1.0)

    return Weather(
        name        = f"RE2020 {zone} {mode}",
        T_ext       = T_ext,
        RH_ext      = RH_ext,
        dirN        = df["dirN"].to_numpy(dtype=float),
        diff        = df["diff"].to_numpy(dtype=float),
        sun_alt_deg = df["Gamma"].to_numpy(dtype=float),
        sun_az_deg  = df["Psi"].to_numpy(dtype=float),
    )
