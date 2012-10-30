#!/usr/bin/env python

"""
Example to fit Jeff's data...

That's all folks.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (31.10.2012)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import csv
import numpy as np

from fit_farquhar_model.farquhar_model import FarquharC3
from fit_farquhar_model.fit_model import FitK25EaDels


##############################
# Fit Eaj, Eav, delSj + delSv
##############################
infname = "results/normalised_results.csv"
ofname = "ea_results.csv"
results_dir = "results"
data_dir = "data"
model = FarquharC3()
peaked = True
############################

F2 = FitK25EaDels(model, infname, ofname, results_dir, data_dir, peaked=peaked)
F2.main(print_to_screen=False)
