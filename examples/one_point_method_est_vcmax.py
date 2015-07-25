#!/usr/bin/env python

"""
Estimate Vcmax using the one-point method

See manuscript for details.

That's all folks.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (15.07.2015)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import numpy as np

from fit_farquhar_model.farquhar_model import FarquharC3

class OnePointVcmax(object):

    """ Bernacchi temperature paramaters are the default here"""
    def __init__(self, Eag=37830.0, gamstar25=42.75):
        self.Eag = Eag
        self.gamstar25 = gamstar25

    def est_vcmax(self, model, Photo, Rd, Ci, Tleaf_k):

        Km = model.calc_michaelis_menten_constants(Tleaf_k)
        gamma_star = model.arrh(self.gamstar25, self.Eag, Tleaf_k)

        return (Photo + Rd) * (Ci + Km) / (Ci - gamma_star)


def main():
    #
    ## Generate some data to estimate V_hat_cmax from
    #
    deg2kelvin = 273.15
    model = FarquharC3(peaked_Jmax=True, peaked_Vcmax=True, model_Q10=True)
    Ci = np.arange(0, 1500, 150)

    curve = 1
    Tleaf = 25.0
    Tleaf += deg2kelvin
    Jmax25 = 150.0
    Vcmax25 = Jmax25 / 1.6
    Rd25 = 2.0
    Eaj = 30000.0
    Eav = 60000.0
    deltaSj = 650.0
    deltaSv = 650.0
    Hdv = 200000.0
    Hdj = 200000.0
    Q10 = 2.0

    (An, Acn, Ajn) = model.calc_photosynthesis(Ci=Ci, Tleaf=Tleaf, Par=None,
                                              Jmax=None, Vcmax=None,
                                              Jmax25=Jmax25, Vcmax25=Vcmax25,
                                              Rd=None, Q10=Q10, Eaj=Eaj,
                                              Eav=Eav, deltaSj=deltaSj,
                                              deltaSv=deltaSv, Rd25=Rd25,
                                              Hdv=Hdv, Hdj=Hdj)

    Rd = model.calc_resp(Tleaf, Q10, Rd25, Tref=25.0)

    #
    ## Use Ci at 300
    #
    V = OnePointVcmax()
    Vcmax_est = V.est_vcmax(model, An[2], Rd, Ci[2], Tleaf)


    print Vcmax25, Vcmax_est



if __name__ == '__main__':
    main()
