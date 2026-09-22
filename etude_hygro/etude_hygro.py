# -*- coding: utf-8 -*-
"""
etude_hygro.py
==============
Figures and tables that support one statement of the thesis:

    "Hygroscopic materials stabilise the indoor humidity and reduce overheating during a heat
     wave; their effect on the heating energy is small (< 3 %) in this configuration."

What is simulated (scenarios.py, combined with "+")
---------------------------------------------------
Walls: 30 cm hempcrete on the four walls, three hygric treatments
    hemp_open   : bare interior surface, the walls exchange moisture with the room
    hemp_film   : PE film (Sd = 100 m) inside, no exchange with the room. This is the reference
                  for the room-side hygroscopic effect: only the interior surface differs.
    hemp_closed : both faces closed to vapour, the classic thermal model
Use cases:
    typical_heated : typical year (Th-BC), heating 1 Oct-30 Apr, RE2020 setpoints 19/16 °C.
                     The SAME heating calendar for every wall, so the energy is compared at equal service.
    heatwave_free  : year with a heat wave (Th-D), free-floating room
Both use 730 days of spin-up. Extra runs: ventilation variants (open and film only) and a spin-up
of 365 days. The heating need of the open walls is still decreasing after 730 days, so a long
spin-up check (open and film, typical year) is run separately with --spinup-check.

Usage
-----
    python etude_hygro/etude_hygro.py                    # run the 16 simulations, then figures + report
    python etude_hygro/etude_hygro.py --zone H3          # another climate zone
    python etude_hygro/etude_hygro.py --spinup-check 1460   # open + film with a longer spin-up (one run each)
    python etude_hygro/etude_hygro.py --analyse <batch>  # only redo figures and report of a batch

Outputs: etude_hygro/results/batch_<date>/ (simulations), etude_hygro/figures_<zone>/ (PDF and PNG,
text in French) and etude_hygro/RAPPORT_<zone>.md (tables, statement checks, limits).
"""

import os, sys, json, glob, argparse, subprocess
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── the three walls: key, scenario, label, colour (validated categorical slots 1-3), line style ──
WALLS = [("open",   "hemp_open",   "Ouverte (hygroscopique)",  "#2A78D6", "-"),
         ("film",   "hemp_film",   "Film Sd = 100 m",          "#EB6834", "--"),
         ("closed", "hemp_closed", "Fermée (thermique seule)", "#1BAF7A", ":")]
COL  = {k: c for k, _, _, c, _ in WALLS}
LAB  = {k: l for k, _, l, _, _ in WALLS}
LS   = {k: s for k, _, _, _, s in WALLS}
INK, INK2, GRID, OUT = "#0B0B0B", "#52514E", "#D9D8D3", "#8A8A85"

MONTHS = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
MONTH_LONG = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
MONTH_OF_DAY  = np.repeat(np.arange(12), DAYS_IN_MONTH)
DAY_TO_DATE   = [(m, d + 1) for m, n in enumerate(DAYS_IN_MONTH) for d in range(n)]

TYP, HW = "typical_heated", "heatwave_free"


def sc(wall, case, *extra):
    return "+".join([f"hemp_{wall}", case, *extra])


def study_names():
    names = [sc(w, c) for c in (TYP, HW) for w in COL]
    names += [sc(w, TYP, v) for w in ("open", "film") for v in ("ach_low", "ach_high")]
    names += [sc(w, HW, v) for w in ("open", "film") for v in ("overvent_none", "overvent_strong")]
    names += [sc(w, TYP, "spinup_365") for w in ("open", "film")]
    return names


def fr(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


# ══════════════════════════════════════════════════════════════════════════════
# Simulations
# ══════════════════════════════════════════════════════════════════════════════

def run_batch(zone, parallel):
    os.makedirs(RESULTS, exist_ok=True)
    before = set(glob.glob(os.path.join(RESULTS, "batch_*")))
    names = study_names()
    cmd = [sys.executable, os.path.join(ROOT, "run_scenarios.py")] + names \
          + ["--parallel", str(parallel), "--out", RESULTS, "--", "--zone", zone]
    print(f"Running {len(names)} simulations ({parallel} at a time), zone {zone}...", flush=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if subprocess.run(cmd, env=env, cwd=ROOT).returncode != 0:
        raise SystemExit("run_scenarios.py failed: see the .log files in the batch folder")
    new = sorted(set(glob.glob(os.path.join(RESULTS, "batch_*"))) - before)
    if not new:
        raise SystemExit("No batch folder was created")
    return new[-1]


def load(batch):
    """{scenario: (summary, hourly series)} for every simulation of the study found in the batch."""
    wanted, out = set(study_names()), {}
    for path in glob.glob(os.path.join(batch, "*", "summary.json")):
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        if s.get("scenario") in wanted:
            out[s["scenario"]] = (s, np.genfromtxt(os.path.join(os.path.dirname(path), "series.csv"),
                                                   delimiter=",", names=True))
    missing = sorted(wanted - set(out))
    if missing:
        raise SystemExit(f"Simulation(s) missing in {batch}: {missing}")
    for name, (s, _) in out.items():
        if s["days"] != 365:
            raise SystemExit(f"{name}: the study needs a full year")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Derived quantities
# ══════════════════════════════════════════════════════════════════════════════

def running_mean(T_out):
    """T_rm of each day (EN 15251, alpha = 0.8), one cyclic year of spin-up, as in simulation.py."""
    day = T_out[:8760].reshape(365, 24).mean(axis=1)
    seq = np.concatenate([day, day])
    r, rm = seq[0], np.zeros(len(seq))
    for d in range(1, len(seq)):
        r = 0.2 * seq[d - 1] + 0.8 * r
        rm[d] = r
    return rm[365:]


def adaptive_limit(T_out):
    return np.maximum(26.0, 0.33 * np.repeat(running_mean(T_out), 24) + 18.8 + 2.0)


def hourly_dh(ser):
    lim = adaptive_limit(ser["T_out"])
    return ser["occupied"][:8760] * np.maximum(ser["T_room"][:8760] - lim, 0.0), lim


def monthly(x):
    d = x[:8760].reshape(365, 24).sum(axis=1)
    return np.array([d[MONTH_OF_DAY == m].sum() for m in range(12)])


def extreme_week(T_out, hottest):
    day = T_out[:8760].reshape(365, 24).mean(axis=1)
    w7 = np.convolve(day, np.ones(7) / 7, mode="valid")
    return int(np.argmax(w7) if hottest else np.argmin(w7))


def date_label(day):
    m, d = DAY_TO_DATE[day % 365]
    return f"{d} {MONTH_LONG[m]}"


# ══════════════════════════════════════════════════════════════════════════════
# Figure helpers
# ══════════════════════════════════════════════════════════════════════════════

def style():
    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.7, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.facecolor": "white", "legend.frameon": False,
    })


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=11, fontweight="bold", color=INK)


def legend_top(fig, extra=()):
    handles = [Line2D([0], [0], color=COL[k], ls=LS[k], lw=2.4, label=LAB[k]) for k in COL] + list(extra)
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=10, bbox_to_anchor=(0.5, 0.955))


def bars(ax, values, fmt, ylabel=None, ylim=None):
    """One bar per wall, value written above it."""
    keys = list(COL)
    x = np.arange(len(keys))
    ax.bar(x, [values[k] for k in keys], 0.62, color=[COL[k] for k in keys], edgecolor="white", linewidth=1.5)
    top = max(values.values()) if max(values.values()) > 0 else 1.0
    for i, k in enumerate(keys):
        ax.text(i, values[k] + 0.02 * top, fmt(values[k]), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, ["Ouverte", "Film", "Fermée"], fontsize=8.5)
    ax.grid(axis="x", visible=False)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_ylim(*(ylim if ylim else (0, top * 1.18)))


def dots(ax, values, fmt, lim, ylabel=None):
    """One dot per wall (a temperature does not start at zero, so no bars), value above it."""
    keys = list(COL)
    x = np.arange(len(keys))
    ax.scatter(x, [values[k] for k in keys], s=110, c=[COL[k] for k in keys], edgecolor="white",
               linewidth=1.5, zorder=3)
    for i, k in enumerate(keys):
        ax.text(i, values[k] + 0.05 * (lim[1] - lim[0]), fmt(values[k]), ha="center", va="bottom",
                fontsize=9, color=INK)
    ax.set_xticks(x, ["Ouverte", "Film", "Fermée"], fontsize=8.5)
    ax.set_xlim(-0.6, len(keys) - 0.4); ax.set_ylim(*lim); ax.grid(axis="x", visible=False)
    if ylabel:
        ax.set_ylabel(ylabel)


def comma(ax, axis="x", decimals=None):
    fmt = FuncFormatter(lambda v, _: (f"{v:g}" if decimals is None else f"{v:.{decimals}f}").replace(".", ","))
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def hour_axis(ax, day0, n_days=7):
    ax.set_xlim(0, n_days * 24)
    ax.set_xticks(np.arange(0, n_days * 24 + 1, 24)[:-1] + 12,
                  [date_label(day0 + k) for k in range(n_days)], fontsize=8)
    for k in range(1, n_days):
        ax.axvline(k * 24, color=GRID, lw=0.8)
    ax.grid(axis="x", visible=False)


def save(fig, folder, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(folder, f"{name}.{ext}"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    return name


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — indoor humidity (typical year)
# ══════════════════════════════════════════════════════════════════════════════

def fig_humidity(D, zone, folder):
    S = {k: D[sc(k, TYP)][0] for k in COL}
    X = {k: D[sc(k, TYP)][1] for k in COL}
    ref = X["open"]
    fig = plt.figure(figsize=(15, 9.2))
    gs = fig.add_gridspec(2, 12, hspace=0.36, wspace=1.0, top=0.86)

    for c, (label, hottest) in enumerate([("Semaine la plus froide", False), ("Semaine la plus chaude", True)]):
        ax = fig.add_subplot(gs[0, 6 * c:6 * c + 6])
        d0 = extreme_week(ref["T_out"], hottest)
        sl = slice(d0 * 24, (d0 + 7) * 24)
        h = np.arange(7 * 24)
        ax.plot(h, ref["RH_out"][sl], color=OUT, lw=1.3, ls="-.", label="Extérieur")
        for k in COL:
            ax.plot(h, X[k]["RH_room"][sl], color=COL[k], ls=LS[k], lw=2.0)
        ax.set_ylim(15, 100); ax.set_ylabel("HR [%]"); hour_axis(ax, d0)
        title(ax, f"{label} ({date_label(d0)} → {date_label(d0 + 6)})")
        if c == 0:
            ax.plot([], [], color=OUT, lw=1.3, ls="-.", label="Extérieur")

    ax = fig.add_subplot(gs[1, 0:6])
    ax.axvspan(30, 70, color="#EDEDE8", zorder=0)
    ax.text(31, 95, "confort hygrique 30–70 %", ha="left", va="top", fontsize=9, color=INK2)
    p = np.linspace(0, 100, 401)
    ax.plot(np.percentile(ref["RH_out"][:8760], p), p, color=OUT, lw=1.3, ls="-.")
    for k in COL:
        ax.plot(np.percentile(X[k]["RH_room"][:8760], p), p, color=COL[k], ls=LS[k], lw=2.0)
    ax.set_xlabel("Humidité relative intérieure [%]"); ax.set_ylabel("Part des heures de l'année ≤ HR [%]")
    ax.set_xlim(15, 100); ax.set_ylim(0, 100)
    title(ax, "Répartition annuelle de l'humidité (courbe plus raide = plus stable)")

    specs = [("RH_daily_amplitude", "Amplitude journalière\nde HR [points]", lambda v: fr(v, 1)),
             ("pct_RH_above_70", "Heures avec\nHR > 70 % [%]", lambda v: fr(v, 1)),
             ("pct_RH_below_30", "Heures avec\nHR < 30 % [%]", lambda v: fr(v, 1))]
    for c, (key, lab, fmt) in enumerate(specs):
        ax = fig.add_subplot(gs[1, 6 + 2 * c:8 + 2 * c])
        bars(ax, {k: S[k][key] for k in COL}, fmt)
        ax.set_title(lab, fontsize=9.5, color=INK, loc="center")

    legend_top(fig)
    fig.suptitle(f"Humidité intérieure — chanvre 30 cm, {zone}, année type chauffée", fontsize=14,
                 fontweight="bold", y=0.985)
    return save(fig, folder, "1_humidite_interieure")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — overheating in a heat wave
# ══════════════════════════════════════════════════════════════════════════════

def fig_heatwave(D, zone, folder):
    S = {k: D[sc(k, HW)][0] for k in COL}
    X = {k: D[sc(k, HW)][1] for k in COL}
    ref = X["open"]
    DH = {k: hourly_dh(X[k])[0] for k in COL}
    lim = hourly_dh(ref)[1]
    fig = plt.figure(figsize=(15, 9.2))
    gs = fig.add_gridspec(2, 12, hspace=0.36, wspace=1.0, top=0.86)

    ax = fig.add_subplot(gs[0, :])
    d0 = extreme_week(ref["T_out"], True)
    sl = slice(d0 * 24, (d0 + 7) * 24)
    h = np.arange(7 * 24)
    ax.plot(h, ref["T_out"][sl], color=OUT, lw=1.3, ls="-.")
    ax.step(h, lim[sl], where="post", color=INK, lw=1.0, ls=(0, (1, 2)))
    for k in COL:
        ax.plot(h, X[k]["T_room"][sl], color=COL[k], ls=LS[k], lw=2.2)
    ax.set_ylabel("Température [°C]"); hour_axis(ax, d0)
    title(ax, f"Semaine la plus chaude ({date_label(d0)} → {date_label(d0 + 6)}) : température de la pièce")

    ax = fig.add_subplot(gs[1, 0:6])
    x = np.arange(365)
    for k in COL:
        ax.plot(x, np.cumsum(DH[k].reshape(365, 24).sum(axis=1)), color=COL[k], ls=LS[k], lw=2.2)
    starts = np.cumsum([0] + DAYS_IN_MONTH[:-1])
    ax.set_xticks(starts[4:9] + 15, MONTHS[4:9], fontsize=9)
    ax.set_xlim(starts[4], starts[9]); ax.set_ylabel("Degrés-heures d'inconfort cumulés [°C·h]")
    ax.grid(axis="x", visible=False)
    title(ax, "DH RE2020 cumulés (mai → septembre ; aucun DH le reste de l'année)")

    ax = fig.add_subplot(gs[1, 7:9])
    bars(ax, {k: S[k]["DH_RE2020"] for k in COL}, lambda v: fr(v, 0))
    ax.set_title("DH RE2020\n[°C·h]", fontsize=9.5, color=INK, loc="center")
    ax = fig.add_subplot(gs[1, 10:12])
    tm = {k: S[k]["T_max_room"] for k in COL}
    dots(ax, tm, lambda v: fr(v, 1), (min(tm.values()) - 1.5, max(tm.values()) + 1.5))
    ax.set_title("Température max\nde la pièce [°C]", fontsize=9.5, color=INK, loc="center")

    legend_top(fig, extra=[Line2D([0], [0], color=OUT, lw=1.3, ls="-.", label="Extérieur"),
                           Line2D([0], [0], color=INK, lw=1.0, ls=(0, (1, 2)), label="Limite RE2020 (catégorie 1)")])
    fig.suptitle(f"Surchauffe en canicule — chanvre 30 cm, {zone} Th-D, pièce non chauffée",
                 fontsize=14, fontweight="bold", y=0.985)
    return save(fig, folder, "2_surchauffe_canicule")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — heating energy at equal service
# ══════════════════════════════════════════════════════════════════════════════

def rel(a, b):
    return 100.0 * (a / b - 1.0)


def spinup_check():
    """{wall: heating need [kWh/m²]} of the long spin-up runs (results/spinup_check), if they exist."""
    out, days = {}, None
    for path in glob.glob(os.path.join(RESULTS, "spinup_check", "batch_*", "*", "summary.json")):
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
        for w in ("open", "film"):
            if s.get("scenario") == sc(w, TYP) and s.get("heating_need_kWh_m2") is not None:
                out[w], days = s["heating_need_kWh_m2"], s["spinup_days"]
    return (out, days) if len(out) == 2 else ({}, None)


def energy_deviations(D):
    """[(label, deviation in %)] used by figure 3 and by the report."""
    n = lambda w, *e: D[sc(w, TYP, *e)][0]["heating_need_kWh_m2"]
    dev = [("Ouverte / film\n(spin-up 365 j)", rel(n("open", "spinup_365"), n("film", "spinup_365"))),
           ("Ouverte / film\n(spin-up 730 j)", rel(n("open"), n("film")))]
    long_run, days = spinup_check()
    if long_run:
        dev.append((f"Ouverte / film\n(spin-up {days} j)", rel(long_run["open"], long_run["film"])))
    dev += [("Fermée / film\n(spin-up 730 j)", rel(n("closed"), n("film"))),
            ("Ouverte / fermée\n(spin-up 730 j)", rel(n("open"), n("closed")))]
    return dev


def fig_energy(D, zone, folder):
    S = {k: D[sc(k, TYP)][0] for k in COL}
    X = {k: D[sc(k, TYP)][1] for k in COL}
    area = float(X["open"]["Q_heat_W"][:8760].sum()) / 1000.0 / S["open"]["heating_need_kWh_m2"]
    fig = plt.figure(figsize=(15, 9.2))
    gs = fig.add_gridspec(2, 6, hspace=0.42, wspace=1.4, top=0.86)

    ax = fig.add_subplot(gs[0, 0:2])
    need = {k: S[k]["heating_need_kWh_m2"] for k in COL}
    bars(ax, need, lambda v: fr(v, 1), ylabel="Besoin de chauffage annuel [kWh/m²]")
    for i, k in enumerate(COL):
        if k != "film":
            ax.text(i, need[k] * 0.5, f"{rel(need[k], need['film']):+.1f} %".replace(".", ",") + "\nvs film",
                    ha="center", va="center", fontsize=9, color="white", fontweight="bold")
    title(ax, "Besoin annuel")

    ax = fig.add_subplot(gs[0, 2:6])
    mh = {k: monthly(X[k]["Q_heat_W"]) / 1000.0 / area for k in COL}
    xm = np.arange(12)
    for i, k in enumerate(COL):
        ax.bar(xm + (i - 1) * 0.27, mh[k], 0.25, color=COL[k], edgecolor="white", linewidth=1.0)
    ax.set_xticks(xm, MONTHS); ax.set_ylabel("Besoin mensuel [kWh/m²]"); ax.grid(axis="x", visible=False)
    title(ax, "Besoin mensuel (calendrier de chauffe fixe : 1er octobre → 30 avril)")

    ax = fig.add_subplot(gs[1, 0:6])
    dev = energy_deviations(D)
    ax.axvspan(-3, 3, color="#EDEDE8", zorder=0)
    ax.text(0, -0.62, "bande ±3 %", ha="center", va="center", fontsize=9, color=INK2)
    y = np.arange(len(dev))[::-1]
    ax.barh(y, [v for _, v in dev], 0.5, color="#52514E", edgecolor="white")
    for yi, (lab, v) in zip(y, dev):
        ax.text(v + (0.15 if v >= 0 else -0.15), yi, f"{v:+.1f} %".replace(".", ","), va="center",
                ha="left" if v >= 0 else "right", fontsize=10, color=INK)
    ax.set_yticks(y, [lab for lab, _ in dev], fontsize=9.5)
    lim = max(4.0, max(abs(v) for _, v in dev) * 1.4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-0.9, len(dev) - 0.4)
    ax.axvline(0, color=INK, lw=0.9); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Écart relatif du besoin de chauffage annuel [%]")
    title(ax, "Effet des parois hygroscopiques sur le chauffage, à service égal")

    fig.legend(handles=[Line2D([0], [0], color=COL[k], lw=8, label=LAB[k]) for k in COL], loc="upper center",
               ncol=3, fontsize=10, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(f"Énergie de chauffage — chanvre 30 cm, {zone} Th-BC, consignes 19/16 °C", fontsize=14,
                 fontweight="bold", y=0.985)
    return save(fig, folder, "3_energie_chauffage")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — what changes with the ventilation
# ══════════════════════════════════════════════════════════════════════════════

def vent_series(D, case, walls, variants, key):
    """{wall: [value for each ventilation variant]}; variants = [(x, extra scenario or None)]."""
    out = {}
    for w in walls:
        out[w] = [D[sc(w, case, *(() if e is None else (e,)))][0][key] for _, e in variants]
    return out


VENT_HW  = [(0.0, "overvent_none"), (4.0, None), (8.0, "overvent_strong")]
VENT_TYP = [(0.3, "ach_low"), (0.5, None), (1.0, "ach_high")]


def fig_ventilation(D, zone, folder):
    fig, axs = plt.subplots(2, 2, figsize=(14, 9.2))
    fig.subplots_adjust(top=0.85, hspace=0.38, wspace=0.26)
    panels = [(axs[0, 0], HW, VENT_HW, "DH_RE2020", "DH RE2020 [°C·h]", "Surventilation nocturne ajoutée [1/h]",
               "Canicule (Th-D) : inconfort d'été"),
              (axs[0, 1], HW, VENT_HW, "T_max_room", "Température max de la pièce [°C]",
               "Surventilation nocturne ajoutée [1/h]", "Canicule (Th-D) : température maximale"),
              (axs[1, 0], TYP, VENT_TYP, "RH_daily_amplitude", "Amplitude journalière de HR [points]",
               "Ventilation d'hygiène [1/h]", "Année type : stabilité de l'humidité"),
              (axs[1, 1], TYP, VENT_TYP, "heating_need_kWh_m2", "Besoin de chauffage [kWh/m²]",
               "Ventilation d'hygiène [1/h]", "Année type : énergie de chauffage")]
    for ax, case, variants, key, ylab, xlab, ttl in panels:
        xs = [v for v, _ in variants]
        ser = vent_series(D, case, ("open", "film"), variants, key)
        for w in ("open", "film"):
            ax.plot(xs, ser[w], color=COL[w], ls=LS[w], lw=2.2, marker="o", ms=7, mfc=COL[w], mec="white", mew=1.5)
        nd = lambda v: 0 if (key == "DH_RE2020" and v >= 10) else 1
        for xi, a, b in zip(xs, ser["open"], ser["film"]):
            ax.annotate(fr(a, nd(a)), (xi, a), textcoords="offset points",
                        xytext=(0, -15 if a <= b else 9), ha="center", fontsize=8.5, color=INK)
            ax.annotate(fr(b, nd(b)), (xi, b), textcoords="offset points",
                        xytext=(0, 9 if a <= b else -15), ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(xs); ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        comma(ax, "x", 1 if case == TYP else None)
        lo = min(min(ser["open"]), min(ser["film"])); hi = max(max(ser["open"]), max(ser["film"]))
        if key == "DH_RE2020":                                   # spans three orders of magnitude
            ax.set_yscale("log"); ax.set_ylim(0.2, hi * 4)
            ax.set_ylabel(ylab + " (échelle log)")
        else:
            pad = (hi - lo) * 0.25 or 1.0
            ax.set_ylim(lo - pad, hi + pad)
        ax.margins(x=0.08); title(ax, ttl)
    fig.legend(handles=[Line2D([0], [0], color=COL[w], ls=LS[w], lw=2.4, marker="o", label=LAB[w])
                        for w in ("open", "film")], loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 0.94))
    fig.suptitle(f"Sensibilité à la ventilation — chanvre 30 cm, {zone}", fontsize=14, fontweight="bold", y=0.985)
    return save(fig, folder, "4_ventilation")


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def report(D, batch, zone, figures, folder):
    g = lambda w, case, key, *e: D[sc(w, case, *e)][0][key]
    L = [f"# Hygroscopic walls — evidence for the thesis statement ({zone})\n",
         f"Generated {datetime.now():%Y-%m-%d %H:%M} from `{os.path.relpath(batch, ROOT)}`. "
         f"Figures (French): `{os.path.relpath(folder, ROOT)}`.\n",
         "> Hygroscopic materials stabilise the indoor humidity and reduce overheating during a heat wave; "
         "their effect on the heating energy is small (< 3 %) in this configuration.\n",
         "## Set-up\n",
         "- 5 × 5 × 2.5 m room, 30 cm hempcrete on the four walls, windows S 1.5 m² and N 0.5 m², "
         "0.5 1/h hygiene ventilation + 0.1 1/h infiltration, night overventilation 4 1/h from May to September, "
         "1 occupant, 730 days of spin-up.",
         "- `open`: bare interior surface. `film`: PE film (Sd = 100 m) inside; the only difference with `open` "
         "is the moisture exchange with the room. `closed`: both faces closed to vapour (classic thermal model).",
         "- Heat wave: Th-D weather, free-floating room. Typical year: Th-BC weather, ideal heater, RE2020 "
         "setpoints 19/16 °C, the same fixed season (1 October → 30 April) for every wall.\n"]

    # 1. humidity
    L += ["## 1. Indoor humidity (typical year, heated)\n",
          "| Wall | RH daily amplitude [pts] | Hours RH > 70 % [%] | Hours RH < 30 % [%] | RH mean [%] |", "|---|---|---|---|---|"]
    for k in COL:
        L.append(f"| {k} | {g(k, TYP, 'RH_daily_amplitude'):.1f} | {g(k, TYP, 'pct_RH_above_70'):.1f} | "
                 f"{g(k, TYP, 'pct_RH_below_30'):.1f} | {g(k, TYP, 'RH_mean'):.0f} |")
    amp_red = -rel(g("open", TYP, "RH_daily_amplitude"), g("film", TYP, "RH_daily_amplitude"))
    L.append(f"\nThe daily RH amplitude is {amp_red:.0f} % lower with open walls than with the film.\n")

    # 2. heat wave
    L += ["## 2. Heat wave (Th-D, free-floating room)\n",
          "| Wall | DH RE2020 [°C·h] | T max room [°C] | RH daily amplitude [pts] |", "|---|---|---|---|"]
    for k in COL:
        L.append(f"| {k} | {g(k, HW, 'DH_RE2020'):.0f} | {g(k, HW, 'T_max_room'):.1f} | "
                 f"{g(k, HW, 'RH_daily_amplitude'):.1f} |")
    dh_o, dh_f = g("open", HW, "DH_RE2020"), g("film", HW, "DH_RE2020")
    L.append(f"\nOpen walls against the film: DH {rel(dh_o, dh_f):+.0f} %, "
             f"T max {g('open', HW, 'T_max_room') - g('film', HW, 'T_max_room'):+.1f} K.\n")

    # 3. energy
    L += ["## 3. Heating energy at equal service (typical year)\n",
          "| Wall | Heating need [kWh/m²] | Bbio heating part (2·B_ch) | Hours heating | Too cold [%] |", "|---|---|---|---|---|"]
    for k in COL:
        q = D[sc(k, TYP)][1]["Q_heat_W"]
        L.append(f"| {k} | {g(k, TYP, 'heating_need_kWh_m2'):.1f} | {2 * g(k, TYP, 'heating_need_kWh_m2'):.1f} | "
                 f"{int((q > 1.0).sum())} | {g(k, TYP, 'comfort_cold_pct'):.1f} |")
    L += ["", "| Comparison | Deviation of the annual heating need |", "|---|---|"]
    dev = energy_deviations(D)
    for lab, v in dev:
        L.append(f"| {lab.replace(chr(10), ' ')} | {v:+.1f} % |")
    ov = [v for lab, v in dev if lab.startswith("Ouverte / film")]      # open vs film, spin-up increasing
    worst = abs(ov[-1])                                                  # verdict: the longest spin-up
    L.append("")
    L.append("Open against film for a spin-up increasing from 365 days: " + ", ".join(f"{v:+.1f} %" for v in ov)
             + ". The heating need of the open walls is still decreasing with the spin-up, so the effect is not "
             "converged and its sign is not established.\n")

    # 4. ventilation
    L += ["## 4. Ventilation sensitivity (open vs film)\n",
          "| Case | Ventilation | DH open | DH film | T max open | T max film |", "|---|---|---|---|---|---|"]
    for x, e in VENT_HW:
        args = () if e is None else (e,)
        L.append(f"| heat wave | night overventilation +{x:g} 1/h | {g('open', HW, 'DH_RE2020', *args):.0f} | "
                 f"{g('film', HW, 'DH_RE2020', *args):.0f} | {g('open', HW, 'T_max_room', *args):.1f} | "
                 f"{g('film', HW, 'T_max_room', *args):.1f} |")
    L += ["", "| Case | Hygiene ventilation | RH amplitude open | RH amplitude film | Heating open | Heating film | Difference |",
          "|---|---|---|---|---|---|---|"]
    for x, e in VENT_TYP:
        args = () if e is None else (e,)
        o, f = (g("open", TYP, "heating_need_kWh_m2", *args), g("film", TYP, "heating_need_kWh_m2", *args))
        L.append(f"| typical year | {x:g} 1/h | {g('open', TYP, 'RH_daily_amplitude', *args):.1f} | "
                 f"{g('film', TYP, 'RH_daily_amplitude', *args):.1f} | {o:.1f} | {f:.1f} | {rel(o, f):+.1f} % |")
    L.append("")

    # 5. statement check
    L += ["## Check of the statement against these simulations\n",
          "| Part of the statement | Evidence | Consistent |", "|---|---|---|",
          f"| stabilise the indoor humidity | daily RH amplitude {amp_red:.0f} % lower than with the film | "
          f"{'yes' if amp_red > 0 else 'no'} |",
          f"| reduce overheating in a heat wave | DH {rel(dh_o, dh_f):+.0f} % and T max "
          f"{g('open', HW, 'T_max_room') - g('film', HW, 'T_max_room'):+.1f} K against the film | "
          f"{'yes' if dh_o < dh_f else 'no'} |",
          f"| heating energy effect < 3 % | open against film with the longest spin-up: {ov[-1]:+.1f} % "
          f"(range over spin-ups: {min(ov):+.1f} to {max(ov):+.1f} %) | {'yes' if worst < 3.0 else 'no'} |\n",
          "Suggested wording, consistent with the numbers above: hygroscopic walls reduce the daily indoor RH "
          f"amplitude by {amp_red:.0f} %, reduce the summer discomfort of a heat wave with night ventilation "
          f"(DH {rel(dh_o, dh_f):+.0f} %, T max {g('open', HW, 'T_max_room') - g('film', HW, 'T_max_room'):+.1f} K), "
          f"and change the heating need by a few percent only ({min(ov):+.1f} to {max(ov):+.1f} % depending on the "
          "spin-up, not converged).\n",
          "## Limits (what these results do not prove)\n",
          "- Only the sensible heating need at an ideal emitter is counted. Ventilation is fixed: humidity-controlled "
          "ventilation, the main route by which hygroscopic materials save energy in the literature, is not modelled.",
          "- Conductivity and vapour permeability do not depend on the moisture content, there is no hysteresis and no "
          "liquid transport; floor and ceiling are internal mass only.",
          "- The latent exchange between walls and room air is not added to the room-air heat balance (it would count "
          "the same energy twice). A closed-box energy-conservation test of the coupling is still pending.",
          "- `closed` also differs from `film` on the exterior face, so `film` is the clean reference for the room-side effect.",
          "- One climate zone, one wall build-up, one occupancy scenario. The comfort indicators of the heat wave depend "
          "on the night overventilation (section 4); the RE2020 DH is only regulatory in Th-D mode.",
          "- Residual dependence on the initial state: compare the two spin-up lengths in section 3."]
    path = os.path.join(HERE, f"RAPPORT_{zone}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return path, dict(amp_red=amp_red, dh=rel(dh_o, dh_f), worst=worst)


def check_dh(D):
    """The DH recomputed from the hourly series must match the one printed by simulation.py."""
    for k in COL:
        s, ser = D[sc(k, HW)]
        mine = float(hourly_dh(ser)[0].sum())
        if abs(mine - s["DH_RE2020"]) > max(1.0, 0.02 * s["DH_RE2020"]):
            print(f"  [!] {k}: DH recomputed {mine:.1f} against {s['DH_RE2020']:.1f} in the summary")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zone", default="H1c", help="RE2020 climate zone (default H1c)")
    ap.add_argument("--parallel", type=int, default=16, help="simulations at the same time")
    ap.add_argument("--analyse", metavar="BATCH", help="skip the simulations, analyse this batch folder")
    ap.add_argument("--spinup-check", type=int, metavar="DAYS",
                    help="only run open and film (typical year) with this spin-up; used by the energy figure")
    a = ap.parse_args()

    if a.spinup_check:
        cmd = [sys.executable, os.path.join(ROOT, "run_scenarios.py"), sc("open", TYP), sc("film", TYP),
               "--parallel", "2", "--out", os.path.join(RESULTS, "spinup_check"), "--",
               "--zone", a.zone, "--spinup", str(a.spinup_check)]
        raise SystemExit(subprocess.run(cmd, env=dict(os.environ, PYTHONIOENCODING="utf-8"), cwd=ROOT).returncode)

    batch = os.path.abspath(a.analyse) if a.analyse else run_batch(a.zone, a.parallel)
    D = load(batch)
    zone = D[sc("open", TYP)][0]["zone"]
    folder = os.path.join(HERE, f"figures_{zone}")
    os.makedirs(folder, exist_ok=True)
    style()
    check_dh(D)
    figs = [fig_humidity(D, zone, folder), fig_heatwave(D, zone, folder),
            fig_energy(D, zone, folder), fig_ventilation(D, zone, folder)]
    path, res = report(D, batch, zone, figs, folder)
    print(f"\nFigures : {os.path.relpath(folder, ROOT)}  ({', '.join(figs)})")
    print(f"Report  : {os.path.relpath(path, ROOT)}")
    print(f"RH amplitude -{res['amp_red']:.0f} %   DH {res['dh']:+.0f} %   heating open vs film, longest spin-up {res["worst"]:.1f} %")


if __name__ == "__main__":
    main()
