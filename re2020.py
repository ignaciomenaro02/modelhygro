# -*- coding: utf-8 -*-
"""
re2020.py
=========
RE2020 energy performance indicators for French residential buildings.

Computes
--------
BBio    Bioclimatic needs indicator  [sans unité, points]
        Measures the building's intrinsic efficiency in limiting heating,
        cooling and lighting needs. Lower is better.

Cep     Primary energy consumption   [kWh_ep/(m²·an)]
        Total primary energy for heating, cooling, DHW, lighting, auxiliaries.

Cep,nr  Non-renewable primary energy [kWh_ep/(m²·an)]
        Same but excluding renewable sources (on-site PV, etc.).

DH      Summer comfort indicator     [°C·h]
        Degree-hours of discomfort above 28°C. < 1250 for RE2020 compliance.

RE2020 thresholds (mainland France, individual house — arrêté 4 août 2021)
---------------------------------------------------------------------------
BBio_max    : 63 points  (climate zone H1a — adjusted per zone)
Cep_max     : 90 kWh_ep/(m²·an)  (all uses)
Cep,nr_max  : 70 kWh_ep/(m²·an)
DH_max      : 1 250 °C·h

Climate zone multipliers for BBio_max (indicative)
----------------------------------------------------
H1a (Paris, Strasbourg)   : × 1.0  → BBio_max = 63
H1b (Lyon, Dijon)         : × 1.0  → 63
H1c (Limoges)             : × 1.0  → 63
H2a (Nantes, Brest)       : × 0.95 → 60
H2b (Bordeaux)            : × 0.85 → 54
H2c (Perpignan inland)    : × 0.80 → 50
H2d (Ajaccio)             : × 0.75 → 47
H3  (Nice, Marseille)     : × 0.70 → 44

Note: The official BBio calculation (Th-BCE engine) is more complex.
This module provides a simplified engineering estimate suitable for
early-design comparisons and sensitivity analyses.

Usage
-----
    from re2020 import RE2020Evaluator
    ev = RE2020Evaluator(floor_area=80.0, climate_zone='H1b')
    ev.add_heating_kWh(1200.0)
    ev.add_cooling_kWh(150.0)
    ev.add_discomfort_hours(T_room_series, dt=3600)
    report = ev.report()
    ev.print_report(report)
"""

import numpy as np


# ── RE2020 reference values ───────────────────────────────────────────────────

# BBio_max by climate zone (indicative — see official tables for exact values)
_BBIO_MAX = {
    'H1a': 63, 'H1b': 63, 'H1c': 63,
    'H2a': 60, 'H2b': 54, 'H2c': 50, 'H2d': 47,
    'H3':  44,
}

# Cep thresholds [kWh_ep/(m²·an)] — all uses, non-renewable
_CEP_MAX   = 90    # all uses
_CEPNR_MAX = 70    # non-renewable

# Summer comfort threshold [°C·h] above 28°C
_DH_MAX = 1250

# Primary energy conversion factors (RE2020 / décret 2021)
_F_EP = {
    'electricity': 2.3,    # [kWh_ep / kWh_final]
    'gas':         1.0,
    'wood':        0.6,
    'district':    0.6,
}

# RE2020 BBio coefficients (αchauff, αrefr, αéclairage)
_ALPHA_HEAT  = 1.0
_ALPHA_COOL  = 1.0
_ALPHA_LIGHT = 1.0


class RE2020Evaluator:
    """
    Accumulates energy balance results and produces RE2020 indicators.

    Parameters
    ----------
    floor_area   : float   Habitable floor area SHON_RT [m²].
    climate_zone : str     French climate zone ('H1a', 'H1b', …, 'H3').
    building_type: str     'individual' or 'collective'.
    """

    def __init__(
        self,
        floor_area   : float = 80.0,
        climate_zone : str   = 'H1b',
        building_type: str   = 'individual',
    ):
        self.A     = floor_area
        self.zone  = climate_zone
        self.btype = building_type

        # Two different energies are tracked, because RE2020 uses them for
        # two different indicators:
        #
        #   • "besoin" (need)  = the THERMAL energy the building demands,
        #                        before the heating/cooling system.  Used for BBio.
        #   • "final" energy   = the energy actually bought (need ÷ COP/efficiency).
        #                        Converted to primary energy for Cep.
        #
        # Accumulators [kWh_final] — energy bought from the grid/network
        self._E_heat_kWh    = 0.0   # final heating energy
        self._E_cool_kWh    = 0.0   # final cooling energy
        self._E_dhw_kWh     = 0.0   # domestic hot water
        self._E_light_kWh   = 0.0   # lighting
        self._E_aux_kWh     = 0.0   # auxiliaries (fans, pumps)

        # Accumulators [kWh] — thermal NEED (besoin) for the BBio indicator
        self._B_heat_kWh    = 0.0   # heating need delivered to the room
        self._B_cool_kWh    = 0.0   # cooling need extracted from the room

        # Carrier info for primary energy conversion
        self._carrier_heat  = 'electricity'
        self._carrier_cool  = 'electricity'
        self._carrier_dhw   = 'electricity'

        # Discomfort degree-hours [°C·h]
        self._DH            = 0.0

        # Renewable production on site [kWh_ep]
        self._E_pv_ep       = 0.0

    # ── Energy input methods ───────────────────────────────────────────────────

    def set_carrier(self, heating='electricity', cooling='electricity', dhw='electricity'):
        """Set the energy carrier for each use."""
        self._carrier_heat = heating
        self._carrier_cool = cooling
        self._carrier_dhw  = dhw

    def add_heating_kWh(self, kWh_final: float, carrier: str = None):
        """Accumulate final heating energy [kWh]."""
        self._E_heat_kWh += kWh_final
        if carrier:
            self._carrier_heat = carrier

    def add_cooling_kWh(self, kWh_final: float, carrier: str = None):
        """Accumulate final cooling energy [kWh]."""
        self._E_cool_kWh += kWh_final
        if carrier:
            self._carrier_cool = carrier

    def add_heating_need_kWh(self, kWh_thermal: float):
        """Accumulate heating NEED (besoin) — thermal energy into the room [kWh]."""
        self._B_heat_kWh += kWh_thermal

    def add_cooling_need_kWh(self, kWh_thermal: float):
        """Accumulate cooling NEED (besoin) — thermal energy out of the room [kWh]."""
        self._B_cool_kWh += kWh_thermal

    def add_dhw_kWh(self, kWh_final: float, carrier: str = None):
        """Accumulate domestic hot water energy [kWh]."""
        self._E_dhw_kWh += kWh_final
        if carrier:
            self._carrier_dhw = carrier

    def add_lighting_kWh(self, kWh: float):
        """Accumulate lighting energy [kWh]."""
        self._E_light_kWh += kWh

    def add_auxiliaries_kWh(self, kWh: float):
        """Accumulate ventilation/pump auxiliary energy [kWh]."""
        self._E_aux_kWh += kWh

    def add_pv_kWh_ep(self, kWh_ep: float):
        """Subtract on-site renewable production [kWh_ep]."""
        self._E_pv_ep += kWh_ep

    def add_discomfort_hours(self, T_room_series: np.ndarray, dt: int = 3600):
        """
        Compute and accumulate summer discomfort degree-hours [°C·h].

        Parameters
        ----------
        T_room_series : array of float   Room temperature [°C] at each step.
        dt            : int              Time step [s].
        """
        T_arr = np.asarray(T_room_series, dtype=float)
        excess = np.maximum(T_arr - 28.0, 0.0)
        self._DH += float(excess.sum()) * (dt / 3600.0)   # [°C·h]

    # ── Indicator calculation ──────────────────────────────────────────────────

    def _to_primary(self, kWh_final: float, carrier: str) -> float:
        """Convert final energy to primary energy [kWh_ep]."""
        return kWh_final * _F_EP.get(carrier, 1.0)

    def _bbio(self) -> dict:
        """
        Simplified BBio indicator [points].

        Definition (RE2020):
            BBio = (alpha_heat * B_heat + alpha_cool * B_cool
                    + alpha_light * B_light) / floor_area

        where B_heat, B_cool, B_light are the building's intrinsic *needs*
        (besoins) — i.e. the energy the envelope+climate demand, BEFORE any
        system efficiency (COP/eta).  BBio therefore rates the building shell
        itself; it does not depend on whether you heat with a heat pump or a
        joule heater.

        Step by step:
          1. Take the accumulated thermal needs B_heat / B_cool [kWh].
             (If the simulation only fed final energy, fall back to it so the
             indicator is still defined — slightly optimistic, see note below.)
          2. Add the lighting need (already a need, not divided by an efficiency).
          3. Normalise per square metre of floor area.
          4. Compare to the per-zone BBio_max.
        """
        # 1–2. Needs per square metre [kWh/(m²·an)].
        #      Prefer the true thermal need; fall back to final energy if the
        #      need was never supplied (keeps old scripts working).
        B_heat  = (self._B_heat_kWh or self._E_heat_kWh) / self.A
        B_cool  = (self._B_cool_kWh or self._E_cool_kWh) / self.A
        B_light = self._E_light_kWh / self.A

        # 3. Weighted sum (the alpha weights are 1.0 in this simplified model).
        BBio = (_ALPHA_HEAT  * B_heat
                + _ALPHA_COOL  * B_cool
                + _ALPHA_LIGHT * B_light)

        # 4. Per-zone reference threshold.
        BBio_max = _BBIO_MAX.get(self.zone, 63)
        return {
            'BBio'      : round(BBio, 1),
            'BBio_max'  : BBio_max,
            'compliant' : BBio <= BBio_max,
            'B_heat'    : round(B_heat,  1),
            'B_cool'    : round(B_cool,  1),
            'B_light'   : round(B_light, 1),
        }

    def _cep(self) -> dict:
        """
        Cep and Cep,nr [kWh_ep/(m²·an)].
        """
        ep_heat  = self._to_primary(self._E_heat_kWh,  self._carrier_heat)
        ep_cool  = self._to_primary(self._E_cool_kWh,  self._carrier_cool)
        ep_dhw   = self._to_primary(self._E_dhw_kWh,   self._carrier_dhw)
        ep_light = self._to_primary(self._E_light_kWh, 'electricity')
        ep_aux   = self._to_primary(self._E_aux_kWh,   'electricity')

        Cep_total = (ep_heat + ep_cool + ep_dhw + ep_light + ep_aux - self._E_pv_ep) / self.A
        Cepnr     = Cep_total   # simplified: assume no renewable subtraction unless PV set

        return {
            'Cep'          : round(Cep_total, 1),
            'Cep_max'      : _CEP_MAX,
            'Cepnr'        : round(Cepnr, 1),
            'Cepnr_max'    : _CEPNR_MAX,
            'compliant_Cep': Cep_total <= _CEP_MAX,
            'ep_heat'      : round(ep_heat  / self.A, 1),
            'ep_cool'      : round(ep_cool  / self.A, 1),
            'ep_dhw'       : round(ep_dhw   / self.A, 1),
            'ep_light'     : round(ep_light / self.A, 1),
        }

    # ── Report ─────────────────────────────────────────────────────────────────

    def report(self) -> dict:
        """Return a dictionary of all RE2020 indicators."""
        bbio = self._bbio()
        cep  = self._cep()
        return {
            'floor_area_m2'  : self.A,
            'climate_zone'   : self.zone,
            **bbio,
            **cep,
            'DH_discomfort'  : round(self._DH, 0),
            'DH_max'         : _DH_MAX,
            'compliant_DH'   : self._DH <= _DH_MAX,
            'E_heat_kWh_m2'  : round(self._E_heat_kWh  / self.A, 1),
            'E_cool_kWh_m2'  : round(self._E_cool_kWh  / self.A, 1),
            'E_total_kWh_m2' : round((self._E_heat_kWh + self._E_cool_kWh +
                                       self._E_dhw_kWh  + self._E_light_kWh +
                                       self._E_aux_kWh) / self.A, 1),
        }

    def print_report(self, report: dict = None):
        """Print a formatted RE2020 compliance summary."""
        if report is None:
            report = self.report()

        tick  = lambda ok: '✅' if ok else '❌'

        print("\n" + "═"*56)
        print("  RE2020 PERFORMANCE SUMMARY")
        print("═"*56)
        print(f"  Building floor area : {report['floor_area_m2']:.0f} m²")
        print(f"  Climate zone        : {report['climate_zone']}")
        print("─"*56)
        print(f"  {'Indicator':<30} {'Value':>8}  {'Limit':>8}  {'OK'}")
        print("─"*56)
        print(f"  {'BBio [points]':<30} {report['BBio']:>8.1f}  "
              f"{report['BBio_max']:>8}  {tick(report['compliant'])}")
        print(f"  {'Cep [kWh_ep/(m²·an)]':<30} {report['Cep']:>8.1f}  "
              f"{report['Cep_max']:>8}  {tick(report['compliant_Cep'])}")
        print(f"  {'Cep,nr [kWh_ep/(m²·an)]':<30} {report['Cepnr']:>8.1f}  "
              f"{report['Cepnr_max']:>8}  {tick(report['Cepnr'] <= report['Cepnr_max'])}")
        print(f"  {'DH discomfort [°C·h]':<30} {report['DH_discomfort']:>8.0f}  "
              f"{report['DH_max']:>8}  {tick(report['compliant_DH'])}")
        print("─"*56)
        print(f"  Final energy breakdown  [kWh/(m²·an)]")
        print(f"    Heating   : {report['E_heat_kWh_m2']:>6.1f}")
        print(f"    Cooling   : {report['E_cool_kWh_m2']:>6.1f}")
        print(f"    Total     : {report['E_total_kWh_m2']:>6.1f}")
        print("═"*56 + "\n")
