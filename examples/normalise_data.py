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

from fit_farquhar_model.normalise import Normalise


###############################
#Normalise data
#############################
fname = "fitting_results.csv"
ofname1 = "values_at_Tnorm.csv"
ofname2 = "normalised_results.csv"
results_dir = "results"
plot_dir = "plots"
tnorm = 25.0
#############################
N = Normalise(fname, ofname1, ofname2, results_dir, plot_dir, tnorm)
N.main()

