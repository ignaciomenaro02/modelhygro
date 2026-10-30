# -*- coding: utf-8 -*-
"""
en15026_lemire_sign.py
======================
EN 15026 Annex B test with the heat balance written exactly as Eq. (2) of Le Mire, Bui, Fabbri & Force (2025),
Construction and Building Materials 490, 141850:

    coefficient of grad T : k + (cp,v·T~ - Lv)·delta_p·(phi·dPsat/dT - Pv·ln(phi)/T)
    coefficient of grad Pc: delta_l·cp,l·T - (cp,v·T~ - Lv)·delta_p·rho_v/rho_l

instead of the model's (+Lv, as in Bui et al. 2025, Eq. 5). Everything else is unchanged.

    python validation/en15026_lemire_sign.py
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import library as lib
import wall_class
import en15026


class WallLeMire(wall_class.Wall):
    def _coeffs(self, T, RH):
        C11, R11, R12, C22, R22, R21, k0, k1 = super()._coeffs(T, RH)
        lay = self.layer
        RHc = np.maximum(RH, 1e-3)
        X = lay.delta_p(T, RHc) * (RHc * lib.dPsat(T, RHc) - lib.Pv(T, RHc) * np.log(RHc) / T)
        Y = lay.delta_p(T, RHc) * lib.rhoV(T, RHc) / lib.rhoL
        h = lib.CpV * (T - 273.15) - lib.Lv                       # Le Mire et al. (2025), Eq. (2)
        R22 = lay.k(T, RHc) + h * X
        R21 = lay.delta_l(T, RHc) * lib.CpL * T - h * Y
        return C11, R11, R12, C22, R22, R21, k0, k1


if __name__ == "__main__":
    en15026.Wall = WallLeMire                                     # run() builds the wall through this name
    x, out = en15026.run()
    print("Heat balance of Le Mire et al. (2025), Eq. (2), with (cp,v·T~ - Lv):")
    en15026.check(x, out)
