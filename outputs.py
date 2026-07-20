# -*- coding: utf-8 -*-
"""
outputs.py
==========
Visualisation and reporting for room hygrothermal simulation results.

Functions
---------
plot_room_climate        T and RH of room air over time.
plot_hvac_energy         Heating/cooling power and cumulative energy.
plot_wall_profiles       Temperature and RH profiles across each wall
                         at a given time instant.
plot_wall_timeseries     T and RH at mid-wall point over time.
plot_re2020_comparison   Bar chart comparing BBio/Cep to RE2020 thresholds.
plot_comfort_map         Adaptive comfort map (Nicol & Humphreys).
plot_walls_3d            3D schematic of the walls, layer by layer.
table_energy_balance     Print tabular energy balance.
compare_scenarios        Compare two or more simulation results side by side.
export_csv               Export all results to a CSV file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_style("whitegrid")
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
          '#8c564b', '#e377c2', '#7f7f7f']


# ── Time-axis helpers ─────────────────────────────────────────────────────────
# The simulation may run at any time step (e.g. 10 min, not 1 h). These helpers
# read the step actually used (sim.dt, set by RoomSimulation.run) so every plot
# uses the correct time axis instead of assuming hourly data.

def _dt_seconds(sim) -> float:
    """Simulation time step [s] (falls back to 3600 s if not set)."""
    return float(getattr(sim, 'dt', None) or 3600)

def _auto_days(sim, n: int) -> np.ndarray:
    """Time axis in days for `n` stored points, given the real step."""
    return np.arange(n) * _dt_seconds(sim) / 86400.0

def _steps_per_day(sim) -> float:
    """Number of stored time steps per day."""
    return 86400.0 / _dt_seconds(sim)

def _safe_savefig(fig, path, **kwargs):
    """
    Save a figure without crashing the whole run if the file can't be written
    (most commonly: the PDF is still open in a viewer, which locks it on
    Windows). In that case we just warn and carry on, so a long simulation
    never loses all its other outputs because of one locked file.
    """
    if not path:
        return
    try:
        fig.savefig(path, **kwargs)
    except PermissionError:
        print(f"[!] Could not save '{path}' (file open in another program?). "
              f"Close it and re-run to refresh this one. Skipping for now.")


# ══════════════════════════════════════════════════════════════════════════════
# Room climate
# ══════════════════════════════════════════════════════════════════════════════

def plot_room_climate(sim, t_days=None, xlim=None, save=None):
    """
    Plot room air temperature and relative humidity over time.

    Parameters
    ----------
    sim    : RoomSimulation   Completed simulation object.
    t_days : array            Time axis [days]. Auto-computed if None.
    xlim   : tuple            (t_start, t_end) in days.
    save   : str              File path to save (e.g. 'room_climate.pdf').
    """
    T_room  = np.array(sim.StockT_room)
    RH_room = np.array(sim.StockRH_room) * 100.0

    if t_days is None:
        t_days = _auto_days(sim, len(T_room))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax1.plot(t_days, T_room, lw=0.8, color=COLORS[0])
    ax1.axhline(20, ls='--', lw=0.7, color='gray', label='Heat setpoint 20°C')
    ax1.axhline(26, ls='--', lw=0.7, color='red',  label='Cool setpoint 26°C')
    ax1.set_ylabel('Room temperature [°C]', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_title(f'Room thermal & hygric climate — {len(sim.wall_configs)} walls', fontsize=13)

    ax2.plot(t_days, RH_room, lw=0.8, color=COLORS[1])
    ax2.axhline(30, ls='--', lw=0.7, color='gray', label='Min comfort 30%')
    ax2.axhline(70, ls='--', lw=0.7, color='red',  label='Max comfort 70%')
    ax2.set_ylabel('Room RH [%]', fontsize=13)
    ax2.set_xlabel('Day', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 100)

    if xlim:
        ax1.set_xlim(*xlim)

    plt.tight_layout()
    if save:
        _safe_savefig(fig, save, dpi=150)
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# HVAC energy
# ══════════════════════════════════════════════════════════════════════════════

def plot_hvac_energy(sim, t_days=None, xlim=None, save=None):
    """
    Plot instantaneous HVAC power and cumulative energy.
    """
    Q_arr = np.array(sim.StockQ_HVAC) / 1000.0   # [kW]
    Q_heat = np.where(Q_arr > 0, Q_arr, 0)
    Q_cool = np.where(Q_arr < 0, -Q_arr, 0)

    dt_h   = _dt_seconds(sim) / 3600.0   # hours per step (e.g. 10 min → 0.1667)
    E_heat = np.cumsum(Q_heat * dt_h)
    E_cool = np.cumsum(Q_cool * dt_h)

    if t_days is None:
        t_days = _auto_days(sim, len(Q_arr))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax1.fill_between(t_days, Q_heat, 0, color=COLORS[3], alpha=0.6, label='Heating [kW]')
    ax1.fill_between(t_days, -Q_cool, 0, color=COLORS[0], alpha=0.6, label='Cooling [kW]')
    ax1.set_ylabel('HVAC power [kW]', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_title('HVAC power and cumulative energy', fontsize=13)

    ax2.plot(t_days, E_heat, color=COLORS[3], lw=1.2, label=f'Heating {sim.E_heat_kWh:.0f} kWh')
    ax2.plot(t_days, E_cool, color=COLORS[0], lw=1.2, label=f'Cooling {sim.E_cool_kWh:.0f} kWh')
    ax2.set_ylabel('Cumulative energy [kWh]', fontsize=13)
    ax2.set_xlabel('Day', fontsize=13)
    ax2.legend(fontsize=10)

    if xlim:
        ax1.set_xlim(*xlim)

    plt.tight_layout()
    if save:
        _safe_savefig(fig, save, dpi=150)
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Wall profiles (snapshot)
# ══════════════════════════════════════════════════════════════════════════════

def plot_wall_profiles(sim, day=50, save=None):
    """
    Plot temperature and RH profiles through each wall at a given day.

    Parameters
    ----------
    day : float   Day of simulation to snapshot.
    """
    step = int(day * _steps_per_day(sim))   # works for any time step

    n_walls = len(sim.wall_configs)
    fig, axes = plt.subplots(n_walls, 2, figsize=(12, 3.5 * n_walls))
    if n_walls == 1:
        axes = [axes]

    for idx, (cfg, layer, wall_obj) in enumerate(
            zip(sim.wall_configs, sim.layers, sim.walls)):

        ax_T, ax_RH = axes[idx]

        # Retrieve stored profile at `step`
        T_prof  = np.array(wall_obj.StockT)[min(step, len(wall_obj.StockT)-1)].flatten() - 273.15
        RH_prof = np.array(wall_obj.StockRH)[min(step, len(wall_obj.StockRH)-1)].flatten() * 100

        x_cm = layer.x_pos * 100  # [cm]

        ax_T.plot(x_cm, T_prof, color=COLORS[0], lw=1.5)
        ax_T.set_xlabel('Position [cm]', fontsize=11)
        ax_T.set_ylabel('T [°C]', fontsize=11)
        ax_T.set_title(f'{cfg.name} — Temperature (day {day})', fontsize=11)

        ax_RH.plot(x_cm, RH_prof, color=COLORS[1], lw=1.5)
        ax_RH.set_xlabel('Position [cm]', fontsize=11)
        ax_RH.set_ylabel('RH [%]', fontsize=11)
        ax_RH.set_title(f'{cfg.name} — Relative humidity (day {day})', fontsize=11)
        ax_RH.set_ylim(0, 100)

        # Draw layer boundaries
        for ax in (ax_T, ax_RH):
            for xb in layer.interface_pos[:-1] * 100:
                ax.axvline(xb, ls='--', lw=0.8, color='gray', alpha=0.7)

    plt.tight_layout()
    if save:
        _safe_savefig(fig, save, dpi=150)
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Wall time series at mid-point
# ══════════════════════════════════════════════════════════════════════════════

def plot_wall_timeseries(sim, wall_index=0, t_days=None, xlim=None, save=None):
    """
    Plot temperature and RH at the interior surface of a wall over time.
    """
    cfg      = sim.wall_configs[wall_index]
    T_surf   = np.array(sim.StockT_walls[cfg.name])
    RH_room  = np.array(sim.StockRH_room) * 100.0

    if t_days is None:
        t_days = _auto_days(sim, len(T_surf))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)

    ax1.plot(t_days, T_surf, lw=0.8, color=COLORS[0],
             label=f'{cfg.name} — interior surface')
    ax1.set_ylabel('Surface temperature [°C]', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_title(f'Wall interior surface — {cfg.name}', fontsize=13)

    ax2.plot(t_days, RH_room, lw=0.8, color=COLORS[1], label='Room RH')
    ax2.set_ylabel('Room RH [%]', fontsize=13)
    ax2.set_xlabel('Day', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 100)

    if xlim:
        ax1.set_xlim(*xlim)

    plt.tight_layout()
    if save:
        _safe_savefig(fig, save, dpi=150)
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# RE2020 comparison chart
# ══════════════════════════════════════════════════════════════════════════════

def plot_re2020_comparison(reports: dict, save=None):
    """
    Bar chart comparing one or several scenarios against RE2020 thresholds.

    Parameters
    ----------
    reports : dict   {scenario_name: re2020_report_dict}
                     Each report is the dict returned by RE2020Evaluator.report().
    save    : str    Save path.

    Example
    -------
    plot_re2020_comparison({
        'Hempcrete wall': report_1,
        'Concrete wall' : report_2,
    })
    """
    indicators = [
        ('BBio',  'BBio_max',  'BBio [points]'),
        ('Cep',   'Cep_max',   'Cep [kWh_ep/(m²·an)]'),
        ('Cepnr', 'Cepnr_max', 'Cep,nr [kWh_ep/(m²·an)]'),
    ]
    n_ind = len(indicators)
    scenarios = list(reports.keys())
    n_sc  = len(scenarios)

    fig, axes = plt.subplots(1, n_ind, figsize=(5 * n_ind, 5))

    for col, (key, limit_key, label) in enumerate(indicators):
        ax = axes[col]
        values = [reports[sc][key]         for sc in scenarios]
        limits = [reports[sc][limit_key]   for sc in scenarios]

        x = np.arange(n_sc)
        bars = ax.bar(x, values, color=[COLORS[i % len(COLORS)] for i in range(n_sc)],
                      alpha=0.8, edgecolor='k', linewidth=0.5)

        # Draw the RE2020 limit as a horizontal line
        limit_val = limits[0]
        ax.axhline(limit_val, color='red', ls='--', lw=1.5,
                   label=f'RE2020 max = {limit_val}')

        # Value labels on bars
        for bar, val in zip(bars, values):
            color = 'green' if val <= limit_val else 'red'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + limit_val * 0.02,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10, color=color,
                    fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(scenarios, rotation=20, ha='right', fontsize=10)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12)
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(max(values), limit_val) * 1.25)

    fig.suptitle('RE2020 Compliance — Scenario Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    if save:
        _safe_savefig(fig, save, dpi=150, bbox_inches='tight')
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Adaptive comfort map
# ══════════════════════════════════════════════════════════════════════════════

def plot_comfort_map(sim, T_ext_series, t_days=None, save=None):
    """
    Adaptive comfort map (EN 15251 / ASHRAE 55).

    Plots operative temperature vs running mean outdoor temperature,
    with the comfort band overlaid.
    """
    T_room = np.array(sim.StockT_room)
    T_ext  = np.asarray(T_ext_series)

    # 7-day exponential running mean of outdoor temp
    alpha  = 0.8
    T_rm   = np.zeros_like(T_ext, dtype=float)
    T_rm[0]= T_ext[0]
    for i in range(1, len(T_ext)):
        T_rm[i] = alpha * T_rm[i-1] + (1 - alpha) * T_ext[i]

    n = min(len(T_room), len(T_rm))
    T_room = T_room[:n]
    T_rm   = T_rm[:n]

    # EN 15251 Cat II comfort band: T_comf = 0.33 T_rm + 18.8 ± 3°C
    T_rm_range = np.linspace(T_rm.min(), T_rm.max(), 100)
    T_comf     = 0.33 * T_rm_range + 18.8
    T_upper    = T_comf + 3.0
    T_lower    = T_comf - 3.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.fill_between(T_rm_range, T_lower, T_upper,
                    color='green', alpha=0.15, label='Comfort zone (EN 15251 Cat II)')
    ax.plot(T_rm_range, T_comf, 'g--', lw=1, alpha=0.7)
    ax.scatter(T_rm, T_room, s=2, alpha=0.4, color=COLORS[0], label='Simulation')
    ax.set_xlabel('Running mean outdoor temperature [°C]', fontsize=12)
    ax.set_ylabel('Operative (room) temperature [°C]', fontsize=12)
    ax.set_title('Adaptive Comfort Map — EN 15251', fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    if save:
        _safe_savefig(fig, save, dpi=150)
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Tabular energy balance
# ══════════════════════════════════════════════════════════════════════════════

def table_energy_balance(sim, floor_area: float = 1.0):
    """
    Print and return a DataFrame with the energy balance.

    Parameters
    ----------
    floor_area : float   Floor area [m²] for normalisation (kWh/m²).
    """
    rows = []
    T_arr = np.array(sim.StockT_room)
    dt_h  = _dt_seconds(sim) / 3600.0
    DH    = float(np.maximum(T_arr - 28.0, 0.0).sum()) * dt_h   # [°C·h]

    rows.append({'Item': 'Heating energy',
                 'kWh': round(sim.E_heat_kWh, 1),
                 'kWh/m²': round(sim.E_heat_kWh / floor_area, 1)})
    rows.append({'Item': 'Cooling energy',
                 'kWh': round(sim.E_cool_kWh, 1),
                 'kWh/m²': round(sim.E_cool_kWh / floor_area, 1)})
    rows.append({'Item': 'DH discomfort [°C·h]',
                 'kWh': round(DH, 0),
                 'kWh/m²': '—'})
    rows.append({'Item': 'Mean room T [°C]',
                 'kWh': round(T_arr.mean(), 1),
                 'kWh/m²': '—'})
    rows.append({'Item': 'Mean room RH [%]',
                 'kWh': round(np.array(sim.StockRH_room).mean() * 100, 1),
                 'kWh/m²': '—'})

    df = pd.DataFrame(rows).set_index('Item')
    print("\n" + df.to_string() + "\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Multi-scenario comparison
# ══════════════════════════════════════════════════════════════════════════════

def compare_scenarios(sims: dict, t_days=None, xlim=None, save=None):
    """
    Overlay room temperature and RH for multiple scenarios.

    Parameters
    ----------
    sims : dict   {label: RoomSimulation}
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    for i, (label, sim) in enumerate(sims.items()):
        T_room  = np.array(sim.StockT_room)
        RH_room = np.array(sim.StockRH_room) * 100.0
        t       = t_days if t_days is not None else _auto_days(sim, len(T_room))
        c       = COLORS[i % len(COLORS)]
        ax1.plot(t, T_room,  lw=0.9, color=c, label=label)
        ax2.plot(t, RH_room, lw=0.9, color=c, label=label)

    ax1.set_ylabel('Room temperature [°C]', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_title('Scenario comparison — room climate', fontsize=13)
    ax2.set_ylabel('Room RH [%]', fontsize=13)
    ax2.set_xlabel('Day', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 100)

    if xlim:
        ax1.set_xlim(*xlim)

    plt.tight_layout()
    if save:
        _safe_savefig(fig, save, dpi=150)
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3D view of the walls that were built
# ══════════════════════════════════════════════════════════════════════════════

def plot_walls_3d(sim, mag: float = 3.0, room=(5.0, 4.0, 2.7), save=None):
    """
    Draw a 3D schematic of the room and every wall, layer by layer.

    Each wall is placed on its side of a schematic room box (by orientation)
    and shown as a stack of coloured slabs — one per material layer, ordered
    exterior → interior, with thickness drawn to scale (× `mag` so thin finish
    layers stay visible).  A legend maps colours to materials and each wall is
    labelled with its U-value.

    Parameters
    ----------
    sim  : RoomSimulation   Built simulation (uses sim.wall_configs + sim.layers).
    mag  : float            Thickness magnification factor for visibility.
    room : (Lx, Ly, Lz)     Schematic room dimensions [m] (visual only).
    save : str              Save path (e.g. 'walls_3d.pdf').
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    Lx, Ly, H = room
    cx, cy    = Lx / 2.0, Ly / 2.0

    # One stable colour per distinct material (in first-seen order).
    # Use the 20-colour tab20 palette so many materials stay distinguishable.
    palette   = list(plt.cm.tab20.colors)
    mat_color = {}
    for layer in sim.layers:
        for m in layer.mat:
            if m not in mat_color:
                mat_color[m] = palette[len(mat_color) % len(palette)]

    def _placement(orientation, width):
        """Return (P, u, v, n): base corner, two in-plane edge vectors, outward normal."""
        o = orientation.lower()
        if o in ('s',):   return (np.array([cx-width/2, 0.0, 0.0]), np.array([width,0,0]), np.array([0,0,H]), np.array([0,-1,0]))
        if o in ('n',):   return (np.array([cx-width/2, Ly,  0.0]), np.array([width,0,0]), np.array([0,0,H]), np.array([0, 1,0]))
        if o in ('w',):   return (np.array([0.0, cy-width/2, 0.0]), np.array([0,width,0]), np.array([0,0,H]), np.array([-1,0,0]))
        if o in ('e',):   return (np.array([Lx,  cy-width/2, 0.0]), np.array([0,width,0]), np.array([0,0,H]), np.array([1,0,0]))
        if o == 'roof':   return (np.array([0.0,0.0,H]), np.array([Lx,0,0]), np.array([0,Ly,0]), np.array([0,0, 1]))
        if o == 'floor':  return (np.array([0.0,0.0,0.0]), np.array([Lx,0,0]), np.array([0,Ly,0]), np.array([0,0,-1]))
        # diagonals / unknown → treat like South
        return (np.array([cx-width/2, 0.0, 0.0]), np.array([width,0,0]), np.array([0,0,H]), np.array([0,-1,0]))

    def _slab_faces(P, u, v, n, d0, d1):
        """6 faces of the cuboid spanned by rectangle (P,u,v) extruded from d0 to d1 along n."""
        corners = []
        for d in (d0, d1):
            for a, b in ((0,0), (1,0), (1,1), (0,1)):
                corners.append(P + a*u + b*v + d*n)
        c = np.array(corners)
        idx = [[0,1,2,3], [4,5,6,7], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]]
        return [c[f] for f in idx]

    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection='3d')

    for cfg, layer in zip(sim.wall_configs, sim.layers):
        is_horizontal = cfg.orientation.lower() in ('roof', 'floor')
        width   = Lx if is_horizontal else max(cfg.area / H, 1.0)   # in-plane width [m]
        P, u, v, n = _placement(cfg.orientation, width)

        thk = np.asarray(layer.emat, dtype=float) * mag      # magnified thicknesses [m]
        # Distance of each layer's inner boundary from the interior face.
        inner = np.zeros(len(thk)); cum = 0.0
        for k in reversed(range(len(thk))):       # interior layer sits against the face
            inner[k] = cum; cum += thk[k]

        for k, m in enumerate(layer.mat):
            faces = _slab_faces(P, u, v, n, inner[k], inner[k] + thk[k])
            poly  = Poly3DCollection(faces, alpha=0.92, facecolor=mat_color[m],
                                     edgecolor='k', linewidths=0.3)
            ax.add_collection3d(poly)

        # Label the wall at the centre of its exterior face.
        centre = P + 0.5*u + 0.5*v + cum*n
        ax.text(*centre, f"{cfg.orientation}\nU={layer.U_value():.2f}",
                fontsize=8, ha='center', va='center', color='k')

    # Legend (one proxy patch per material).
    handles = [mpatches.Patch(color=c, label=m) for m, c in mat_color.items()]
    ax.legend(handles=handles, loc='upper left', fontsize=9, title='Materials')

    ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]'); ax.set_zlabel('z [m]')
    ax.set_title(f'Walls — 3D view ({len(sim.wall_configs)} walls, '
                 f'layer thickness ×{mag:g})', fontsize=13)
    ax.set_box_aspect((Lx, Ly, H))
    ax.view_init(elev=22, azim=-60)

    plt.tight_layout()
    if save:
        _safe_savefig(fig, save, dpi=150)
    plt.show()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CSV export
# ══════════════════════════════════════════════════════════════════════════════

def export_csv(sim, filepath: str = 'results.csv', dt: int = 3600):
    """
    Export all room-level results to a CSV file.

    Columns: time_h, T_room_C, RH_room_pct, Q_HVAC_W, [T_surf_wall1, ...]
    """
    n = len(sim.StockT_room)
    data = {
        'time_h'     : np.arange(n) * dt / 3600,
        'T_room_C'   : np.array(sim.StockT_room),
        'RH_room_pct': np.array(sim.StockRH_room) * 100,
        'Q_HVAC_W'   : np.array(sim.StockQ_HVAC),
    }
    for cfg in sim.wall_configs:
        key = f'T_surf_{cfg.name.replace(" ","_")}_C'
        data[key] = np.array(sim.StockT_walls[cfg.name])

    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False, float_format='%.3f')
    print(f"Results exported to: {filepath}  ({len(df)} rows)")
    return df
