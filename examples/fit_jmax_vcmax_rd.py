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
from fit_farquhar_model.fit_model import FitJmaxVcmaxRd

def read_data(fname, delimiter=","):
    """ Read the A-Ci data. 
    
    Expects a format of:
    -> Curve, Tleaf, Ci, Photo, Species, Season, Leaf
    """
    data = np.recfromcsv(fname, delimiter=delimiter, names=True, 
                         case_sensitive=True)
    return data


##############################
# Fit Jmax, Vcmax + Rd
##############################
ofname = "fitting_results.csv"
results_dir = "results"
data_dir = "data"
plot_dir = "plots"
model = FarquharC3()
##############################
F = FitJmaxVcmaxRd(model, ofname, results_dir, data_dir, plot_dir)
F.main(print_to_screen=False)     


# OK what are the real values??
fit = read_data("results/fitting_results.csv")
deg2kelvin = 273.15
index = 0
for Tleaf in np.arange(15.0, 40.0, 5.0):
    Tleaf += deg2kelvin
    Jmax25 = 150.0
    Vcmax25 = Jmax25 / 1.6
    Eaj = 30000.0
    Eav = 60000.0
    deltaSj = 650.0
    deltaSv = 650.0
    Hdv = 200000.0
    Hdj = 200000.0
    Q10 = 2.0
    r25 = 0.5
    
    Vcmax = model.peaked_arrh(Vcmax25, Eav, Tleaf, deltaSv, Hdv)
    Jmax = model.peaked_arrh(Jmax25, Eaj, Tleaf, deltaSj, Hdj)
    Rd = model.resp(Tleaf, Q10, r25, Tref=25.0)
    
    print "              Tleaf     Jmax      Vcmax        Rd"
    print "Truth - curve", Tleaf-deg2kelvin, Jmax, Vcmax, Rd
    print "Fit - curve 1", fit["Tav"][index], fit["Jmax"][index], \
                           fit["Vcmax"][index], fit["Rd"][index]
    print "Fit - curve 2", fit["Tav"][index+1], fit["Jmax"][index+1], \
                           fit["Vcmax"][index+1], fit["Rd"][index+1]
    print
    
    index +=2