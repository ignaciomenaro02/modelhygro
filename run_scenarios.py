# -*- coding: utf-8 -*-
"""
run_scenarios.py
================
Run several named scenarios (see scenarios.py) and compare them.

    python run_scenarios.py                       # list the scenarios
    python run_scenarios.py base rammed_earth     # run these, one after the other
    python run_scenarios.py --all                 # run every scenario
    python run_scenarios.py --all --parallel 3    # up to 3 at the same time
    python run_scenarios.py hemp rammed_earth -- --zone H3 --start 01/01 --end 01/01
                                                  # flags after "--" go to every simulation

Everything is saved in  results/batch_<date>_<time>/ :
one folder per scenario (with its own PDFs, run_config.txt, summary.json) plus
comparison.csv, a table with the key indicators of all the scenarios.
"""

import os, sys, csv, json, argparse, subprocess
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DIR)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from scenarios import SCENARIOS

COLUMNS = ["scenario", "zone", "mode", "start", "end", "walls", "ach_overvent", "heating",
           "T_max_out", "T_max_room", "T_min_room", "DH_RE2020", "DH_28C",
           "comfort_ok_pct", "comfort_hot_pct", "comfort_cold_pct",
           "heating_need_kWh_m2", "RH_mean", "RH_daily_amplitude", "pct_RH_above_70",
           "pct_RH_below_30", "latent_release_kWh", "latent_absorb_kWh",
           "spinup_days", "folder"]


def overlay_figure(batch_dir, folders):
    """Overlay of the scenarios: daily-mean indoor RH and hourly RH / T in the hottest and coldest week."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {}
    for name, folder in folders.items():
        path = os.path.join(batch_dir, folder, "series.csv")
        if os.path.isfile(path):
            data[name] = np.genfromtxt(path, delimiter=",", names=True)
    if len(data) < 2:
        return
    first  = next(iter(data.values()))
    n_days = int(first["day"].max()) + 1
    win    = min(7, n_days)
    T_day  = first["T_out"][:n_days * 24].reshape(n_days, 24).mean(axis=1)
    mean_w = np.convolve(T_day, np.ones(win) / win, mode="valid")
    weeks  = [("Semaine la + chaude", int(np.argmax(mean_w))), ("Semaine la + froide", int(np.argmin(mean_w)))]

    fig = plt.figure(figsize=(17, 11), constrained_layout=True)
    gs  = fig.add_gridspec(3, 2)
    ax0 = fig.add_subplot(gs[0, :])
    daily = lambda x: x[:n_days * 24].reshape(n_days, 24).mean(axis=1)
    ax0.plot(daily(first["RH_out"]), color="#AAAAAA", lw=1.2, ls="--", label="HR extérieure")
    for name, d in data.items():
        ax0.plot(daily(d["RH_room"]), lw=1.6, label=name)
    ax0.set_ylabel("HR moyenne journalière [%]"); ax0.set_xlabel("Jour de la simulation")
    ax0.legend(fontsize=9, ncol=3); ax0.grid(True, alpha=0.2)
    for c, (title, w) in enumerate(weeks):
        sl  = slice(w * 24, (w + win) * 24)
        hrs = np.arange(win * 24)
        axr, axt = fig.add_subplot(gs[1, c]), fig.add_subplot(gs[2, c])
        axr.plot(hrs, first["RH_out"][sl], color="#AAAAAA", lw=1.0, ls="--")
        axt.plot(hrs, first["T_out"][sl],  color="#AAAAAA", lw=1.0, ls="--")
        for name, d in data.items():
            axr.plot(hrs, d["RH_room"][sl], lw=1.6, label=name)
            axt.plot(hrs, d["T_room"][sl],  lw=1.6, label=name)
        axr.set_title(f"{title} (jours {w + 1} à {w + win})", fontweight="bold")
        axr.set_ylabel("HR pièce [%]"); axt.set_ylabel("T pièce [°C]"); axt.set_xlabel("Heures")
        axr.grid(True, alpha=0.2); axt.grid(True, alpha=0.2)
    fig.suptitle(f"Comparaison des scénarios — {os.path.basename(batch_dir)}", fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(batch_dir, "comparison_series.pdf"), dpi=150)
    plt.close(fig)


def list_scenarios():
    print("Available scenarios (scenarios.py):\n")
    for name, sc in SCENARIOS.items():
        print(f"  {name:24s} {sc.get('note', '')}")
    print("\nRun:  python run_scenarios.py <name> [<name> ...]    or    --all")


def run_one(name, batch_dir, extra, quiet):
    env = dict(os.environ, SIM_RESULTS_DIR=batch_dir, PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, os.path.join(DIR, "simulation.py"), "--scenario", name] + extra
    if quiet:
        with open(os.path.join(batch_dir, f"{name}.log"), "w", encoding="utf-8") as log:
            code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    else:
        print(f"\n{'=' * 70}\n  SCENARIO: {name}\n{'=' * 70}", flush=True)
        code = subprocess.run(cmd, env=env).returncode
    print(f"  [{'OK' if code == 0 else 'FAILED'}] {name}", flush=True)
    return name, code


def main():
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("names", nargs="*", help="scenario names")
    ap.add_argument("--all", action="store_true", help="run every scenario")
    ap.add_argument("--parallel", type=int, default=1, help="simulations at the same time")
    args = ap.parse_args(argv)

    if not args.names and not args.all:
        list_scenarios()
        return
    names = list(SCENARIOS) if args.all else args.names
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        raise SystemExit(f"Unknown scenario(s): {unknown}. Run without arguments to list them.")

    batch_dir = os.path.join(DIR, "results", "batch_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(batch_dir)
    print(f"Batch folder: {os.path.relpath(batch_dir, DIR)}   ({len(names)} scenarios, "
          f"{args.parallel} at a time)")

    quiet = args.parallel > 1
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        results = list(pool.map(lambda n: run_one(n, batch_dir, extra, quiet), names))

    # ── comparison table ──────────────────────────────────────────────────────
    rows, folders = [], {}
    for name, code in results:
        if code != 0:
            continue
        for sub in sorted(os.listdir(batch_dir)):
            path = os.path.join(batch_dir, sub, "summary.json")
            if os.path.isfile(path):
                with open(path, encoding="utf-8") as f:
                    s = json.load(f)
                if s.get("scenario") == name:
                    folders[name] = sub
                    s["walls"] = " | ".join(f"{k[0]}: {v}" for k, v in s["walls"].items())
                    rows.append({c: s.get(c, "") for c in COLUMNS})
    if rows:
        overlay_figure(batch_dir, folders)
        with open(os.path.join(batch_dir, "comparison.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter=";")
            w.writeheader(); w.writerows(rows)
        print("\n" + "=" * 100)
        print(f"{'scenario':24s} {'T max':>6s} {'DH RE2020':>10s} {'comfort%':>9s} {'heat kWh/m2':>11s} "
              f"{'RH mean':>8s} {'RH ampl':>8s} {'RH>70%':>7s} {'RH<30%':>7s} {'lat kWh':>8s}")
        print("-" * 100)
        f1 = lambda v, n=1: "" if v in ("", None) else f"{float(v):.{n}f}"
        for r in rows:
            print(f"{r['scenario']:24s} {f1(r['T_max_room']):>6s} {f1(r['DH_RE2020'], 0):>10s} "
                  f"{f1(r['comfort_ok_pct'], 0):>9s} {f1(r['heating_need_kWh_m2']):>11s} "
                  f"{f1(r['RH_mean'], 0):>8s} {f1(r['RH_daily_amplitude']):>8s} "
                  f"{f1(r['pct_RH_above_70']):>7s} {f1(r['pct_RH_below_30']):>7s} "
                  f"{f1(float(r['latent_release_kWh'] or 0) + float(r['latent_absorb_kWh'] or 0), 0):>8s}")
        print("=" * 100)
        print(f"Comparison table: {os.path.relpath(os.path.join(batch_dir, 'comparison.csv'), DIR)}")
    failed = [n for n, c in results if c != 0]
    if failed:
        print(f"Failed scenarios: {failed} (see the .log files / run them one by one)")


if __name__ == "__main__":
    main()
