# -*- coding: utf-8 -*-
"""
calcul_wall_2layer.py
=====================
Validation script for ONE two-layer wall (Hempcrete + Rammed Earth) against
the B30 laboratory experiment.

What it does
------------
1. Loads the measured chamber boundary conditions (boundary_data.xlsx, B30):
   air temperature and RH on both sides of the wall test specimen.
2. Loads the measured temperature & RH in the MIDDLE of the wall
   (data_middle.xlsx, B30) — the reference to compare against.
3. Runs the `Wall` model with those boundary conditions.
4. Plots model vs experiment (T, RH, vapour pressure Pv) at the mid-wall point.

NOTE — this is NOT the room simulation.
This script deliberately uses the B30 lab data, because its purpose is to check
that the wall solver reproduces a controlled experiment. The whole-room annual
simulation driven by real weather lives in `main_room.py`.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = os.path.dirname(os.path.abspath(__file__))
os.chdir(data)
if data not in sys.path:
    sys.path.insert(0, data)

from wall_layer import WallLayer
from wall_class import Wall
import library as lib

sns.set_style("whitegrid")


###############################################################################
############################### Load data #####################################
###############################################################################

SHEET = "B30"

clim_df = pd.read_excel("boundary_data.xlsx", sheet_name=SHEET)
exp_df  = pd.read_excel("data_middle.xlsx",   sheet_name=SHEET)

# Boundary conditions
clim     = clim_df.values
time_bc  = clim[:, 0]                  # [s]
Text_bc  = clim[:, 1] + 273.15         # [K]
Tint_bc  = clim[:, 2] + 273.15         # [K]
RHext_bc = clim[:, 3] / 100.0          # [-]
RHint_bc = clim[:, 4] / 100.0          # [-]
t_tot    = int(time_bc.max())

# Experimental mid-wall data
exp     = exp_df.values
day_exp = exp[:, 0] / (24 * 3600)
T_exp   = exp[:, 1]                    # °C
RH_exp  = exp[:, 2]                    # %


def bc_at(t):
    """Linearly interpolated BCs at time t [s]."""
    return (
        float(np.interp(t, time_bc, Text_bc)),
        float(np.interp(t, time_bc, RHext_bc)),
        float(np.interp(t, time_bc, Tint_bc)),
        float(np.interp(t, time_bc, RHint_bc)),
    )


###############################################################################
############################ Wall description##################################
###############################################################################

# thicknesses [m] 
E_HEMPCRETE   = 0.20   # exterior layer
E_RAMMED_EARTH = 0.30  # interior layer (terre pisée)

layer = WallLayer(
    mat      = ["Hempcrete", "Rammed_Earth"],
    emat     = [E_HEMPCRETE, E_RAMMED_EARTH],
    Mesh_Opt = 0,
    liq      = 0,
    label    = "Hempcrete + Rammed_Earth",
)
print(layer)

# Mid-wall monitoring point (for intial test, centre of total thickness, but we can put in the interface)
mid_pos = layer.total_thickness / 2.0
mid_idx = int(np.argmin(np.abs(layer.x_pos - mid_pos)))
print(f"Mid-wall position : {layer.x_pos[mid_idx]*100:.1f} cm  (node {mid_idx})")
print(f"  Layer 1 — Hempcrete    : {E_HEMPCRETE*100:.0f} cm")
print(f"  Layer 2 — Rammed Earth : {E_RAMMED_EARTH*100:.0f} cm")
print(f"  Interface at           : {E_HEMPCRETE*100:.0f} cm")

# Initial conditions from experimental data at t = 0
T0  = T_exp[0]  + 273.15
RH0 = RH_exp[0] / 100.0

wall = Wall(
    layer   = layer,
    h_ext   = 25.0,
    hm_ext  = 25e-9,
    h_int   = 8.0,
    hm_int  = 25e-9,
    T_init  = T0,
    RH_init = RH0,
)


###############################################################################
################################ Time loop ####################################
###############################################################################

dt      = 3600          # [s]
n_steps = int(t_tot / dt)
t_days  = [0.0]

# Creating a progress bar
def update_progress(job_title, progress):
    length = 20
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100, 2))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()

print(f"\nRunning {n_steps} steps ({t_tot/(24*3600):.0f} days) …")
t_start = time.process_time()

for step in range(n_steps):
    t_sim = step * dt
    T_e, RH_e, T_i, RH_i = bc_at(t_sim + dt)
    wall.step(T_e, RH_e, T_i, RH_i, dt)
    t_days.append((t_sim + dt) / (24 * 3600))
    update_progress("Simulation", (step + 1) / n_steps)

    if np.any(np.isnan(wall.T)) or np.any(np.isnan(wall.RH)):
        print(f"\n[!] NaN detected at step {step+1} (day {t_days[-1]:.2f}) — stopping.")
        break

elapsed = time.process_time() - t_start
print(f"Done in {elapsed:.2f} s")


###############################################################################
############################### Post-processing ############################### 
###############################################################################

t_days = np.array(t_days)
n_stored = len(wall.StockT)

StockT_arr  = np.asarray(wall.StockT).reshape(n_stored, -1)
StockRH_arr = np.asarray(wall.StockRH).reshape(n_stored, -1)
StockPsat   = lib.Psat(StockT_arr, StockRH_arr)
StockPv_arr = StockRH_arr * StockPsat

# Mid-wall time series
T_sim  = StockT_arr[:,  mid_idx] - 273.15
RH_sim = StockRH_arr[:, mid_idx] * 100.0
Pv_sim = StockPv_arr[:, mid_idx]

# Experimental uncertainty bands
T_sup  = T_exp  + 0.5;  T_inf  = T_exp  - 0.5
RH_sup = RH_exp + 2.0;  RH_inf = RH_exp - 2.0
Pv_exp     = lib.Pv(T_exp + 273.15, RH_exp / 100)
Pv_exp_sup = lib.Pv(T_sup + 273.15, RH_sup / 100)
Pv_exp_inf = lib.Pv(T_inf + 273.15, RH_inf / 100)

XLIM = (0, 100)


###############################################################################
###################################   Plots   #################################
###############################################################################

# Temperature 
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_days,  T_sim,  lw=0.8, color='#1f77b4', label='Model')
ax.plot(day_exp, T_exp,  lw=0.8, color='#ff7f0e', label='Exp')
ax.set_xlim(*XLIM);  ax.set_ylim(10, 40)
ax.set_xlabel('Day', fontsize=16)
ax.set_ylabel('T [°C]', fontsize=16)
ax.tick_params(labelsize=13)
ax.legend(loc='upper right', fontsize=13)
ax.set_title(f'{layer.label} — T at x = {layer.x_pos[mid_idx]*100:.0f} cm', fontsize=12)
plt.tight_layout(); plt.show(); fig.savefig('T in the middle.pdf')

# Relative humidity 
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_days,  RH_sim,  lw=0.8, color='#1f77b4', label='Model')
ax.fill_between(day_exp, RH_inf, RH_sup, color='#ff7f0e', alpha=0.35, label='Exp ± 2 %')
ax.plot(day_exp, RH_exp,  lw=0.8, color='#ff7f0e', label='Exp')
ax.set_xlim(*XLIM);  ax.set_ylim(40, 70)
ax.set_xlabel('Day', fontsize=16)
ax.set_ylabel('RH [%]', fontsize=16)
ax.tick_params(labelsize=13)
ax.legend(loc='upper right', fontsize=13)
ax.set_title(f'{layer.label} — RH at x = {layer.x_pos[mid_idx]*100:.0f} cm', fontsize=12)
plt.tight_layout(); plt.show(); fig.savefig('RH in the middle.pdf')

# Vapour pressure 
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t_days, Pv_sim, lw=0.8, color='#1f77b4', label='Model')
ax.fill_between(day_exp, Pv_exp_inf, Pv_exp_sup,
                color='#ff7f0e', alpha=0.35, label='Exp ± uncertainty')
ax.set_xlim(*XLIM);  ax.set_ylim(1000, 3500)
ax.set_xlabel('Day', fontsize=16)
ax.set_ylabel('$P_v$ [Pa]', fontsize=16)
ax.tick_params(labelsize=13)
ax.legend(loc='upper left', fontsize=13)
ax.set_title(f'{layer.label} — $P_v$ at x = {layer.x_pos[mid_idx]*100:.0f} cm', fontsize=12)
plt.tight_layout(); plt.show(); fig.savefig('Pv in the middle.pdf')

