# -*- coding: utf-8 -*-
"""
library.py
==========
Physical constants and the moist-air relations used everywhere else.

Temperatures are in kelvin and relative humidity is a fraction (0-1) in all functions.
Psat, dPsat and Pv take RH as an argument even when they do not need it, so that every
property in the code can be called the same way, f(T, RH).
"""

import numpy as np

# General
R    = 8.314       # universal gas constant [J/(mol·K)]
Patm = 101325.0    # atmospheric pressure [Pa]

# Dry air
CpA = 1004.0       # heat capacity [J/(kg·K)]

# Liquid water
rhoL = 1000.0      # density [kg/m³]
CpL  = 4180.0      # heat capacity [J/(kg·K)]
Mw   = 18e-3       # molar mass [kg/mol]

# Water vapour
CpV = 1850.0       # heat capacity [J/(kg·K)]
Lv  = 2.5e6        # latent heat of evaporation at 0 °C [J/kg]
Rv  = 461.5        # gas constant [J/(kg·K)]


def Psat(T, RH):
    """Saturation vapour pressure [Pa] (ISO 7730)."""
    return 1000.0 * np.exp(16.6536 - 4030.183 / (T - 273.15 + 235.0))


def dPsat(T, RH):
    """Derivative of Psat with respect to temperature [Pa/K]."""
    return Psat(T, RH) * 4030.183 / (T - 273.15 + 235.0) ** 2


def Pv(T, RH):
    """Vapour pressure [Pa]."""
    return RH * Psat(T, RH)


def rhoV(T, RH):
    """Vapour density [kg/m³] (ideal gas)."""
    return Pv(T, RH) / (Rv * T)


def Pc(T, RH):
    """Capillary pressure [Pa], counted positive (Kelvin law): Pc = -ln(RH)·rhoL·Rv·T."""
    return -np.log(RH) * rhoL * Rv * T


def Da(T):
    """Vapour diffusion coefficient in still air, as a permeability [kg/(m·s·Pa)] (Künzel 1995)."""
    return 2e-7 * T ** 0.81 / Patm
