#!/usr/bin/env python

"""
Example fit to some synthetic data...fitting all the curves in a single step

That's all folks.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (02.06.2014)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import numpy as np

from fit_farquhar_model.farquhar_model import FarquharC3
from fit_farquhar_model.fit_dummy_version_ear_const import FitMe

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
model = FarquharC3(peaked_Jmax=True, peaked_Vcmax=True, model_Q10=False)
##############################
F = FitMe(model, ofname, results_dir, data_dir, plot_dir)
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
    Ear = 34000.0
    deltaSj = 650.0
    deltaSv = 650.0
    Hdv = 200000.0
    Hdj = 200000.0
    Q10 = 2.0
    R25 = 2.0
    
    Vcmax = model.peaked_arrh(Vcmax25, Eav, Tleaf, deltaSv, Hdv)
    Jmax = model.peaked_arrh(Jmax25, Eaj, Tleaf, deltaSj, Hdj)
    Rd = R25 * Q10**(((Tleaf - deg2kelvin) - 25.0) / 10.0)

    print "\t\t   %s    %s      %s     %s" % ("Tleaf", "Jmax", "Vcmax", "Rd")
    print "%s\t %.5f %.5f %.5f %.5f" % ("Truth - curve", 
                                         Tleaf-deg2kelvin, Jmax, Vcmax, Rd)
    
    idx1 = 'Vcmax25_1'
    idx2 = 'Jfac'
    idx3 = 'Rdfac' 
    v = fit[idx1] 
    Jmax25 = v * fit[idx2]
    Rd25 = v * fit[idx3]
    
    Vcmax = model.peaked_arrh(v, fit['Eav'], Tleaf, fit['delSv'], Hdv)
    Jmax = model.peaked_arrh(Jmax25, fit['Eaj'], Tleaf, fit['delSj'], Hdj)
    Rd = model.calc_resp(Tleaf, Q10, Rd25, Ear=Ear, Tref=25.0)
    
    print "%s\t  %.5f %.5f %.5f %.5f" % ("Fit - curve", 
                                         Tleaf-deg2kelvin,
                                         Jmax, Vcmax, Rd)
    print
    
    index +=1