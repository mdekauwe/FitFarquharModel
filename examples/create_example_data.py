#!/usr/bin/env python

"""
Generate some synthetic data to test against...

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
fp = open(fname, "wb")
wr = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_NONE,  escapechar=' ')
wr.writerow(["Curve", "Tleaf", "Ci", "Photo", "Species", "Season", "Leaf"])
deg2kelvin = 273.15
model = FarquharC3(peaked_Jmax=True, peaked_Vcmax=True)
Ci = np.arange(0, 1500, 150)

curve = 1
for Tleaf in np.arange(15.0, 35.0, 5.0):
    Tleaf += deg2kelvin
    Jmax25 = 150.0
    Vcmax25 = Jmax25 / 1.6
    r25 = 0.5
    Eaj = 30000.0
    Eav = 60000.0
    deltaSj = 620.0
    deltaSv = 620.0
    Hdv = 200000.0
    Hdj = 200000.0
    Q10 = 1.5
    
    Rd = model.resp(Tleaf, Q10, r25, Tref=25.0)
    (An, Acn, Ajn) = model.calc_photosynthesis(Ci, Tleaf, Jmax25=Jmax25, 
                                           Vcmax25=Vcmax25, Eaj=Eaj, Eav=Eav, 
                                           deltaSj=deltaSj, deltaSv=deltaSv, 
                                           r25=r25, Q10=Q10, Hdv=Hdv, Hdj=Hdj)
    
    for i in xrange(len(An)):
        row = [curve, Tleaf-deg2kelvin, Ci[i], An[i]+Rd, "Potatoes", "Summer", 1]
        wr.writerow(row) 
    curve += 1     
fp.close()