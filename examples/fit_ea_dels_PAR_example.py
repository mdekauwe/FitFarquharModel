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
import csv
import numpy as np

from fit_farquhar_model.farquhar_model import FarquharC3
from fit_farquhar_model.fit_model import FitEaDels

##############################
# Fit Eaj, Eav, delSj + delSv
##############################
infname = "results/normalised_results.csv"
ofname = "ea_results.csv"
results_dir = "results"
data_dir = "data"
model = FarquharC3()
############################


# ************************
# Need to add the groupby column...(JUST FOR THE EXAMPLE!)
# REMOVE THIS SECTION IF RUNNING FOR REAL
def read_data(fname, delimiter=","):
    """ Read the A-Ci data.

    Expects a format of:
    -> Curve, Tleaf, Ci, Photo, Species, Season, Leaf
    """
    data = np.recfromcsv(fname, delimiter=delimiter, names=True,
                         case_sensitive=True)
    return data

data = read_data(infname)
header = ["Jmax", "Vcmax", "Jnorm", "Vnorm", "Rd", "Tav", \
          "Tarrh", "R2", "n", "Species", "Leaf", "Curve", \
          "Filename", "fitgroup"]
if os.path.isfile(infname):
    os.remove(infname)
fp = open(infname, 'w')            
wr = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_NONE, escapechar=' ')
wr.writerow(header)
for row in data:
    new_row = []
    for col in row:
        new_row.append(col)
    new_row.append(1)
    wr.writerow(new_row)
fp.close()
# REMOVE DOWN TO HERE


F2 = FitEaDels(model, infname, ofname, results_dir, data_dir)
F2.main(print_to_screen=False)
