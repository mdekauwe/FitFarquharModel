#!/usr/bin/env python

"""
Generate some synthetic data to test against...
(note adding a little bit of noise)

That's all folks.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (13.08.2012)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import csv

from fit_farquhar_model.farquhar_model import FarquharC3

fname = "data/example.csv"
fp = open(fname, "w")
wr = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_NONE,  escapechar=' ')
wr.writerow(["Curve", "Tleaf", "Ci", "Photo", "Species", "Season", "Leaf", "fitgroup"])
deg2kelvin = 273.15
model = FarquharC3(peaked_Jmax=True, peaked_Vcmax=True, model_Q10=True)
Ci = np.arange(0, 1500, 150)

curve = 1
for Tleaf in np.arange(15.0, 40.0, 5.0):
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
    add_noise = True



    (An, Acn, Ajn) = model.calc_photosynthesis(Ci=Ci, Tleaf=Tleaf, Par=None,
                                              Jmax=None, Vcmax=None,
                                              Jmax25=Jmax25, Vcmax25=Vcmax25,
                                              Rd=None, Q10=Q10, Eaj=Eaj,
                                              Eav=Eav, deltaSj=deltaSj,
                                              deltaSv=deltaSv, Rd25=Rd25,
                                              Hdv=Hdv, Hdj=Hdj)

    Rd = model.calc_resp(Tleaf, Q10, Rd25, Tref=25.0)
    for i in range(len(An)):
        if add_noise:
            noise = np.random.normal(0.0, 2.0)
        else:
            noise = 0.0

        row = [curve, Tleaf-deg2kelvin, Ci[i], An[i] + Rd + noise, "Potatoes",\
               "Summer", 1, 1]
        wr.writerow(row)
    curve += 1

fp.close()
