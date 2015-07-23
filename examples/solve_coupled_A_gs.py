#!/usr/bin/env python

"""
Iteratively solve leaf temp, ci, gs and An, following Maetra

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (23.07.2015)"
__email__ = "mdekauwe@gmail.com"

import sys
import numpy as np
import os
import math

from fit_farquhar_model.farquhar_model import FarquharC3
from fit_farquhar_model.stomtal_conductance_models import StomtalConductance
from fit_farquhar_model.leaf_energy_balance import LeafEnergyBalance


def main(tair, par, vpd, wind, leaf_width, leaf_absorptance, pressure, g0, g1,
         D0, Vcmax25, Jmax25, Rd25, Eaj, Eav, deltaSj, deltaSv, Hdv, Hdj,
         Q10, Ca):

    # Ratio of Gbh:Gbc
    GBHGBC = 1.32
    deg2kelvin = 273.15

    F = FarquharC3(peaked_Jmax=True, peaked_Vcmax=False, model_Q10=True)
    S = StomtalConductance(g0=g0, g1=g1, D0=D0)
    L = LeafEnergyBalance(leaf_width, leaf_absorptance)

    # set initialise values
    Cs = Ca  # start at Ca
    Tleaf = tair
    Tleaf_K = Tleaf + deg2kelvin
    dleaf = vpd
    iter_max = 100


    print "start"
    print Cs, Tleaf, dleaf
    print

    iter = 0
    while True:

        (An, Acn, Ajn) = F.calc_photosynthesis(Ci=Cs, Tleaf=Tleaf_K, Par=par,
                                               Jmax25=Jmax25, Vcmax25=Vcmax25,
                                               Q10=Q10, Eaj=Eaj,
                                               Eav=Eav, deltaSj=deltaSj,
                                               deltaSv=deltaSv, Rd25=Rd25,
                                               Hdv=Hdv, Hdj=Hdj)
        gs = S.leuning(dleaf, An, Cs)


        (new_tleaf, et, gbH, gv) = L.calc_leaf_temp(Tleaf, tair, gs, par, dleaf,
                                                  pressure, wind)

        # update Cs and VPD
        gbc = gbH / GBHGBC
        Cs = Ca - An / gbc
        dleaf = et * pressure / gv

        print Cs, Tleaf, dleaf, An, gs

        if math.fabs(Tleaf - new_tleaf) < 0.02:
            break

        if iter > iter_max:
            raise Exception('No convergence!')

        Tleaf = new_tleaf
        Tleaf_K = Tleaf + deg2kelvin
        iter += 1

    # Now recalculate new An and gs based on resolved vpd, ci, tleaf
    (An, Acn, Ajn) = F.calc_photosynthesis(Ci=Cs, Tleaf=Tleaf_K, Par=par,
                                           Jmax25=Jmax25, Vcmax25=Vcmax25,
                                           Q10=Q10, Eaj=Eaj,
                                           Eav=Eav, deltaSj=deltaSj,
                                           deltaSv=deltaSv, Rd25=Rd25,
                                           Hdv=Hdv, Hdj=Hdj)
    gs = S.leuning(dleaf, An, Cs)

    print
    print "End"
    print Cs, Tleaf, dleaf, An, gs




if __name__ == '__main__':

    # gs stuff
    g0 = 0.01
    g1 = 9.0
    D0 = 1.5 # kpa

    # A stuff
    Vcmax25 = 30.0
    Jmax25 = Vcmax25 * 2.0
    Rd25 = 2.0
    Eaj = 30000.0
    Eav = 60000.0
    deltaSj = 650.0
    deltaSv = 650.0
    Hdv = 200000.0
    Hdj = 200000.0
    Q10 = 2.0
    par = 1500.0
    tair = 20.0
    vpd = 2.0
    wind = 2.5
    leaf_width = 0.02
    leaf_absorptance = 0.86 # leaf absorptance of solar radiation [0,1]
    pressure = 101.0
    Ca = 400.0
    main(tair, par, vpd, wind, leaf_width, leaf_absorptance, pressure, g0, g1,
         D0, Vcmax25, Jmax25, Rd25, Eaj, Eav, deltaSj, deltaSv, Hdv, Hdj,
         Q10, Ca)
