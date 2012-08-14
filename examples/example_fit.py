#!/usr/bin/env python

"""
Example fit to some synthetic data...

That's all folks.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (13.08.2012)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import numpy as np

from fit_farquhar_model.farquhar_model import FarquharC3
from fit_farquhar_model.fit_model import FitJmaxVcmaxRd, FitEaDels

##############################
# Fit Jmax, Vcmax + Rd
##############################
ofname = "fitting_results.txt"
ofname25 = "params_at_25.txt"
results_dir = "results"
data_dir = "data"
plot_dir = "plots"
model = FarquharC3(peaked_Jmax=True, peaked_Vcmax=True)
##############################
F = FitJmaxVcmaxRd(model, ofname, ofname25, results_dir, data_dir, plot_dir)
F.main(print_to_screen=False)     

##############################
# Fit Eaj, Eav, delSj + delSv
##############################
infname = "test_ea_fit.csv"
ofname = "ea_results.txt"
results_dir = "results"
data_dir = "data"
model = FarquharC3()
############################
F2 = FitEaDels(model, infname, ofname, results_dir, data_dir)
F2.main(print_to_screen=False)

