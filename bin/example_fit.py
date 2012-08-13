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

from FitFarquharModel import fit_vcmax_jmax_and_rd as fitvjr




ofname = "fitting_results.txt"
ofname25 = "params_at_25.txt"
results_dir = "results"
data_dir = "data"
plot_dir = "plots"

F = fitvjr.FitMe(ofname, ofname25, results_dir, data_dir, plot_dir)
F.main(print_to_screen=False)     