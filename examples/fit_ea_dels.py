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
from fit_farquhar_model.fit_model import FitEaDels


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
F2.main(print_to_screen=False, species_loop=False)

