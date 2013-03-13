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


deg2kelvin = 273.15
model = FarquharC3(peaked_Jmax=True, peaked_Vcmax=True)
#Par = np.arange(0, 1500, 150)
Ci = np.arange(0, 1500, 150)
Par = np.ones(len(Ci)) * (800.)
Tleaf = 30.0
Tleaf += deg2kelvin
Jmax25 = 150.0
Vcmax25 = Jmax25 / 1.6
r25 = 0.5
Eaj = 30000.0
Eav = 60000.0
deltaSj = 650.0
deltaSv = 650.0
Hdv = 200000.0
Hdj = 200000.0
Q10 = 2.0

Rd = model.resp(Tleaf, Q10, r25, Tref=25.0)
(An, Acn, Ajn) = model.calc_photosynthesis(Ci, Tleaf, Par=Par, Jmax25=Jmax25, 
                                       Vcmax25=Vcmax25, Eaj=Eaj, Eav=Eav, 
                                       deltaSj=deltaSj, deltaSv=deltaSv, 
                                       r25=r25, Q10=Q10, Hdv=Hdv, Hdj=Hdj)



plt.plot(Ci, An, "k-", label="$A_n$", lw=5)
plt.plot(Ci, Acn, "b-", label="$A_c$", lw=1)
plt.plot(Ci, Ajn, "g-", label="$A_j$", lw=1)
plt.legend(numpoints=1, loc="best")
plt.show()