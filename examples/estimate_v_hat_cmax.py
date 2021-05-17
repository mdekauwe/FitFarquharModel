#!/usr/bin/env python

"""
Estimate V_hat_cmax using the one point method

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


def estimate_one_point_vcmax(m, Photo, Rd, Ci, Tk, gamstar25, Eag):

    # Using Bernacchi temp dependancies
    Km = m.calc_michaelis_menten_constants(Tk)
    gamma_star = m.arrh(gamstar25, Eag, Tk)

    if Rd is None:
        # Assume Rd 1.5% of Vcmax following Collatz et al. (1991)
        return Photo / ((Ci - gamma_star) / (Ci + Km) - 0.015)
    else:
        return (Photo + Rd) * (Ci + Km) / (Ci - gamma_star)



# Generate some data to estimate V_hat_cmax from
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
Eag = 37830.0
gamstar25 = 42.75
(An, Acn, Ajn) = model.calc_photosynthesis(Ci=Ci, Tleaf=Tleaf, Par=None,
                                          Jmax=None, Vcmax=None,
                                          Jmax25=Jmax25, Vcmax25=Vcmax25,
                                          Rd=None, Q10=Q10, Eaj=Eaj,
                                          Eav=Eav, deltaSj=deltaSj,
                                          deltaSv=deltaSv, Rd25=Rd25,
                                          Hdv=Hdv, Hdj=Hdj)

Rd = model.calc_resp(Tleaf, Q10, Rd25, Tref=25.0)

# Use Ci at 300
vc_hat_cmax = estimate_one_point_vcmax(model, An[2], Rd, Ci[2], Tleaf,
                                       gamstar25, Eag)


print(Vcmax25, vc_hat_cmax)
