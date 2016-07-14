#!/usr/bin/env python

"""
Fit Jmax, Vcmax using measured Rdark, assuming a relationship to Rd.

That's all folks.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (14.07.2016)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob

from fit_farquhar_model.farquhar_model import FarquharC3
from fit_farquhar_model.fit_model import FitJmaxVcmaxRd
from fit_farquhar_model.fit_model import FitJmaxVcmaxKnownRdark

def main():

    ofname = "fitting_results_%s.csv" % ("Mercado")
    results_dir = "results"
    data_dir = "data"
    plot_dir = "plots"

    # Normal way
    model = FarquharC3(peaked_Jmax=True, peaked_Vcmax=True, model_Q10=True)
    F = FitJmaxVcmaxRd(model, ofname, results_dir, data_dir, plot_dir)
    F.main(print_to_screen=True)

    # Using Rdark measurement, assuming Rd = Rdark * 0.6; assuming Q10
    ofname = "fitting_results_%s.csv" % ("Mercado_Rdark")
    model = FarquharC3(peaked_Jmax=True, peaked_Vcmax=True, model_Q10=True)
    F = FitJmaxVcmaxKnownRdark(model, ofname, results_dir, data_dir, plot_dir)
    F.main(print_to_screen=True)

    # Using Rdark measurement, assuming Rd = Rdark * 0.6; assuming no Tdependancy
    ofname = "fitting_results_%s.csv" % ("Mercado_Rdark")
    model = FarquharC3(peaked_Jmax=True, peaked_Vcmax=True, measured_Rd=True)
    F = FitJmaxVcmaxKnownRdark(model, ofname, results_dir, data_dir, plot_dir)
    F.main(print_to_screen=True)

if __name__ == "__main__":

    main()
