# modelhygro

Coupled heat and moisture (HAM) model of a room with hygroscopic walls — PhD thesis of Luis Ignacio MENA
(CIFRE LTDS-ENTPE / UCBL / BATISERF).

The walls are solved in 1D with capillary pressure and temperature as potentials (implicit finite
differences); the room is a well-mixed air node coupled to the walls at every time step. The RE2020
conventions (weather, occupancy, internal gains, heating season, summer-comfort indicator DH) are
implemented to compare hygroscopic bio- and geo-based walls on regulatory indicators.

## Files

| File | Content |
|---|---|
| `simulation.py` | Configuration (section 1), run and results of one simulation |
| `scenarios.py` | Named variants of the configuration (`--scenario name`, combined with `+`) |
| `run_scenarios.py` | Runs several scenarios and writes a comparison table |
| `room_simulation.py` | Room heat and moisture balances, coupling with the walls |
| `wall_class.py` | 1D wall solver: coupled heat and moisture transfer, state `[Pc ; T]` |
| `wall_layer.py` | Wall mesh and material properties per layer |
| `wall_config.py` | Wall, window, thermal-bridge and opening descriptions |
| `materials_library.py` | Hygrothermal properties of the materials |
| `library.py` | Physical constants and functions (saturation pressure, Kelvin law…) |
| `weather.py`, `solar.py` | Weather files and solar irradiance on the surfaces |
| `heating.py` | RE2020 occupancy, internal gains and heating scenario |
| `sources.py` | Occupants, equipment, lighting, ventilation, HVAC |
| `re2020.py` | RE2020 indicators (official DH, heating need) |
| `validation/` | Benchmarks: NF EN 15026 Annex B, HAMSTAD benchmark 2 |

## Running

Python 3 with `numpy`, `scipy`, `matplotlib`, `pandas` and `openpyxl` (to read the RE2020 weather workbook).

    python simulation.py                                   # configuration of section 1
    python simulation.py --scenario poster_hemp             # a named scenario
    python simulation.py --zone H1c --start 01/01 --end 01/01
    python run_scenarios.py hemp_open hemp_film             # several scenarios + comparison table
    python validation/en15026.py                            # EN 15026 benchmark

Results are written to `results/` (one folder per run), or to the folder given by the environment
variable `SIM_RESULTS_DIR`.

## Data

The RE2020 weather workbook (`scenarios_meteorologiques_th-bc_th-d_re2020.xlsx`) is required for the
RE2020 climates. Its licence allows regulatory use only: it is not distributed with this repository.
