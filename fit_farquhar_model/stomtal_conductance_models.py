#!/usr/bin/env python

"""
Various stomatal conductance models

Reference:
==========
* Leuning, R.: A critical appraisal of a combined stomatal- photosynthesis
  model for C3 plants, Plant Cell Environ., 18, 339–355, 1995.
* Medlyn, B. E., Duursma, R. A., Eamus, D., Ellsworth, D. S., Pren- tice,
  I. C., Barton, C. V. M., Crous, K. Y., De Angelis, P., Free- man, M., and
  Wingate, L.: Reconciling the optimal and empirical approaches to modelling
  stomatal conductance, Global Change Biol., 17, 2134–2144, 2011.

"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (23.07.2015)"
__email__ = "mdekauwe@gmail.com"

import math

class StomtalConductance(object):

    def __init__(self, g0=None, g1=None, D0=None):
        self.g0 = g0
        self.g1 = g1
        self.D0 = D0

    def medlyn(self, vpd, An, Ci):
        gs = self.g0 + 1.6 * (1.0 + self.g1 / math.sqrt(vpd)) * An / Ci

        return gs

    def leuning(self, vpd, An, Ci):

        # co2 compensation point, obv shouldn't assume this, but it will do for
        # now
        gamma = 0.0

        arg1 = self.g1 * An
        arg2 = (Ci - gamma)
        arg3 = 1.0 + vpd / self.D0
        gs = self.g0 + arg1 / (arg2 * arg3)

        return gs

if __name__ == '__main__':

    Ci = 400. * 0.7
    vpd = 1.5
    An = 15.0
    g0_c3 = 0.01
    g1_c3 = 9.0
    D0_c3 = 1.5 # kpa

    S = StomtalConductance(g0=g0_c3, g1=g1_c3, D0=D0_c3)
    gs = S.leuning(vpd, An, Ci)
    print gs

    S.g1=3.37
    gs = S.medlyn(vpd, An, Ci)
    print gs
