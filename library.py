# -*- coding: utf-8 -*-

"""
Constants and functions
"""

import numpy as np


def partial_derivative(f, x, y, var="x", h=1e-5):

    if var == "x":
        return (f(x + h, y) - f(x - h, y)) / (2 * h)
    elif var == "y":
        return (f(x, y + h) - f(x, y - h)) / (2 * h)
    else:
        raise ValueError("var must be 'x' or 'y'")
    

# Constants
T_ref = 273.15      # Reference temperature [K] => 0°C
R = 8.314           # Ideal gas constant [J.kg-1.K-1]
Patm = 101325.      # Atmospheric pressure [Pa]

# Properties of air
Ra = 287.1          # Ideal gas consant of dry air [J.kg-1.K-1]
CpA = 1004.         # Heat capacity of air [J.kg-1.K-1]

# Properties of liquid water
rhoL = 1000.        # Density of liquid water [kg.m-3]
CpL = 4180.         # Heat capacity of liquid water [J.kg-1.K-1]    
Mw = 18e-3          # Molar mass of water [kg.mol-1]
eta = 1e-3          # Dynamic viscosity of water [Pa.s]

# Properties of water vapour
# Vapour pressure is defined as a function of both T and RH at the end of this file
CpV = 1850.         # Heat capacity of water vapour [J.kg-1.K-1] 
Lv = 2.5e6          # Latent heat of evaporation at 0°C [J.kg-1] 
Rv = 461.5          # Ideal gas consant of water vapour [J.kg-1.K-1]

# Constants regarding the radiative heat transfer
epsilon = 0.9       # Emissivity
kappa = 0.7         # Absorptivity
sigma = 5.67e-8     # Stefan-Boltzmann constant [W/(m2.K4)]


def Tsky(T,cf):
    """
    Sky temperature
    From Janssen et al. (2017). Int Journ of Heat and Mass Transfer, vol.50
    """
    return T - (23.8 - 0.2025*(T-273.15))*(1-0.87*cf)


def Pc(T,RH):
    """
    Capillary pressure [Pa] (choosen positive)  
    Calculated with Gibbs' law 
    """
    return -np.log(RH)*rhoL*Rv*T
   

def Psat(T,RH):
    """
    Water vapour saturation pressure  [Pa]
    ISO 7730
    """       
    return 1000*np.exp(16.6536-4030.183/(T-T_ref + 235))
    

def dPsat(T,RH):
    """
    Derivative of the water vapour saturation pressure over the temperature
    """
    return partial_derivative(Psat, T, RH, var="x")
     
   
def Pv(T,RH):
    """
    Vapour pressure [Pa]
    """
    return RH*Psat(T,RH)

    
def rhoV(T,RH):
    """
    Density of vapour pressure [kg.m-3]
    """
    return Pv(T,RH)/(Rv*T)
    

def rhoA(T):
    """
    Density of air [kg.m-3]
    """
    return Patm/(Ra*T)
    
    

def Da(T):
    """
    Vapour diffusion coefficient [m2/s]
    Expression from Kunzel's thesis 1995 
    """
    return 2e-7 * T**0.81 / Patm 



    

