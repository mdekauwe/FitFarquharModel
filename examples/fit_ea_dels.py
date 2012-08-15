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

def read_data(fname, delimiter=","):
    """ Read the A-Ci data. 
    
    Expects a format of:
    -> Curve, Tleaf, Ci, Photo, Species, Season, Leaf
    """
    data = np.recfromcsv(fname, delimiter=delimiter, names=True, 
                         case_sensitive=True)
    return data

##############################
# Fit Eaj, Eav, delSj + delSv
##############################
infname = "results/normalised_results.csv"
ofname = "ea_results.csv"
results_dir = "results"
data_dir = "data"
model = FarquharC3()
############################
F2 = FitEaDels(model, infname, ofname, results_dir, data_dir)
F2.main(print_to_screen=False, loop_id="Leaf")

# OK what are the real values??
Eaj = 30000.0
Eav = 60000.0
deltaSj = 650.0
deltaSv = 650.0
  

fit = read_data("results/ea_results.csv")
print "Truth - Jmax", Eaj, deltaSj
print "Fit - Jmax", fit["Ea"][0], fit["delS"][0]
print
print "Truth - Vcmax", Eav, deltaSv
print "Fit - Vcmax", fit["Ea"][1], fit["delS"][1]

