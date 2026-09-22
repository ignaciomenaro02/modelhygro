# Hygroscopic walls — evidence for the thesis statement (H1c)

Generated 2026-09-21 17:36 from `etude_hygro\results\batch_20260921_171450`. Figures (French): `etude_hygro\figures_H1c`.

> Hygroscopic materials stabilise the indoor humidity and reduce overheating during a heat wave; their effect on the heating energy is small (< 3 %) in this configuration.

## Set-up

- 5 × 5 × 2.5 m room, 30 cm hempcrete on the four walls, windows S 1.5 m² and N 0.5 m², 0.5 1/h hygiene ventilation + 0.1 1/h infiltration, night overventilation 4 1/h from May to September, 1 occupant, 730 days of spin-up.
- `open`: bare interior surface. `film`: PE film (Sd = 100 m) inside; the only difference with `open` is the moisture exchange with the room. `closed`: both faces closed to vapour (classic thermal model).
- Heat wave: Th-D weather, free-floating room. Typical year: Th-BC weather, ideal heater, RE2020 setpoints 19/16 °C, the same fixed season (1 October → 30 April) for every wall.

## 1. Indoor humidity (typical year, heated)

| Wall | RH daily amplitude [pts] | Hours RH > 70 % [%] | Hours RH < 30 % [%] | RH mean [%] |
|---|---|---|---|---|
| open | 7.2 | 2.8 | 0.0 | 54 |
| film | 18.1 | 13.1 | 3.7 | 54 |
| closed | 18.0 | 11.9 | 3.8 | 54 |

The daily RH amplitude is 60 % lower with open walls than with the film.

## 2. Heat wave (Th-D, free-floating room)

| Wall | DH RE2020 [°C·h] | T max room [°C] | RH daily amplitude [pts] |
|---|---|---|---|
| open | 28 | 31.0 | 8.1 |
| film | 91 | 32.1 | 22.1 |
| closed | 176 | 33.0 | 21.9 |

Open walls against the film: DH -69 %, T max -1.1 K.

## 3. Heating energy at equal service (typical year)

| Wall | Heating need [kWh/m²] | Bbio heating part (2·B_ch) | Hours heating | Too cold [%] |
|---|---|---|---|---|
| open | 28.2 | 56.5 | 2223 | 8.0 |
| film | 28.5 | 56.9 | 2344 | 8.8 |
| closed | 27.8 | 55.7 | 2300 | 8.0 |

| Comparison | Deviation of the annual heating need |
|---|---|
| Ouverte / film (spin-up 365 j) | +4.9 % |
| Ouverte / film (spin-up 730 j) | -0.8 % |
| Ouverte / film (spin-up 1460 j) | -3.4 % |
| Fermée / film (spin-up 730 j) | -2.2 % |
| Ouverte / fermée (spin-up 730 j) | +1.4 % |

Open against film for a spin-up increasing from 365 days: +4.9 %, -0.8 %, -3.4 %. The heating need of the open walls is still decreasing with the spin-up, so the effect is not converged and its sign is not established.

## 4. Ventilation sensitivity (open vs film)

| Case | Ventilation | DH open | DH film | T max open | T max film |
|---|---|---|---|---|---|
| heat wave | night overventilation +0 1/h | 3310 | 3476 | 36.1 | 37.1 |
| heat wave | night overventilation +4 1/h | 28 | 91 | 31.0 | 32.1 |
| heat wave | night overventilation +8 1/h | 1 | 7 | 29.9 | 30.7 |

| Case | Hygiene ventilation | RH amplitude open | RH amplitude film | Heating open | Heating film | Difference |
|---|---|---|---|---|---|---|
| typical year | 0.3 1/h | 6.7 | 20.5 | 19.7 | 20.1 | -1.7 % |
| typical year | 0.5 1/h | 7.2 | 18.1 | 28.2 | 28.5 | -0.8 % |
| typical year | 1 1/h | 8.4 | 16.6 | 50.2 | 50.3 | -0.1 % |

## Check of the statement against these simulations

| Part of the statement | Evidence | Consistent |
|---|---|---|
| stabilise the indoor humidity | daily RH amplitude 60 % lower than with the film | yes |
| reduce overheating in a heat wave | DH -69 % and T max -1.1 K against the film | yes |
| heating energy effect < 3 % | open against film with the longest spin-up: -3.4 % (range over spin-ups: -3.4 to +4.9 %) | no |

Suggested wording, consistent with the numbers above: hygroscopic walls reduce the daily indoor RH amplitude by 60 %, reduce the summer discomfort of a heat wave with night ventilation (DH -69 %, T max -1.1 K), and change the heating need by a few percent only (-3.4 to +4.9 % depending on the spin-up, not converged).

## Limits (what these results do not prove)

- Only the sensible heating need at an ideal emitter is counted. Ventilation is fixed: humidity-controlled ventilation, the main route by which hygroscopic materials save energy in the literature, is not modelled.
- Conductivity and vapour permeability do not depend on the moisture content, there is no hysteresis and no liquid transport; floor and ceiling are internal mass only.
- The latent exchange between walls and room air is not added to the room-air heat balance (it would count the same energy twice). A closed-box energy-conservation test of the coupling is still pending.
- `closed` also differs from `film` on the exterior face, so `film` is the clean reference for the room-side effect.
- One climate zone, one wall build-up, one occupancy scenario. The comfort indicators of the heat wave depend on the night overventilation (section 4); the RE2020 DH is only regulatory in Th-D mode.
- Residual dependence on the initial state: compare the two spin-up lengths in section 3.
