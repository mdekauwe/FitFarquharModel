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

from fit_farquhar_model.farquhar_model import FarquharC3


deg2kelvin = 273.15
model = FarquharC3(peaked_Jmax=True, peaked_Vcmax=True)
Ci = np.arange(0, 1500)
Tleaf = 30.0 + deg2kelvin
Jmax25 = 150.0
Vcmax25 = Jmax25 / 1.6
r25 = 1.0
Eaj = 30000.0
Eav = 60000.0
deltaSj = 620.0
deltaSv = 620.0
Hdv = 200000.0
Hdj = 200000.0
An, Acn, Ajn = model.calc_photosynthesis(Ci, Tleaf, Jmax25, Vcmax25, Eaj, Eav, 
                                         deltaSj, deltaSv, r25, Hdv, Hdj)
plt.plot(Ci, An)
plt.plot(Ci, Acn)
plt.plot(Ci, Ajn)
plt.show()
        