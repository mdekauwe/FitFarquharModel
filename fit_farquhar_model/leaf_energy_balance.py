#!/usr/bin/env python

"""
Calculate the leaf energy balance based on Leuning.

Reference:
==========
* Leuning et al. (1995) Leaf nitrogen, photosynthesis, conductance and
  transpiration: scaling from leaves to canopies

"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (20.12.2013)"
__email__ = "mdekauwe@gmail.com"


import math
import sys
from isothermal_penman_monteith import PenmanMonteith

class LeafEnergyBalance(object):
    """
        Calculate the net leaf enegy balance, given Tleaf and gs
        - only works for single sided leaves!
    """

    def __init__(self, leaf_width, leaf_absorptance):

        # constants
        self.sigma = 5.6704E-08  # stefan boltzmann constant, (w m-2 k-4)
        self.emissivity_leaf = 0.99   # emissivity of leaf (-)
        self.cp = 1010.0         # specific heat of dry air (j kg-1 k-1)
        self.h2olv0 = 2.501e6    # latent heat H2O (J kg-1)
        self.h2omw = 18E-3       # mol mass H20 (kg mol-1)
        self.air_mass = 29.0E-3     # mol mass air (kg mol-1)
        self.umol_to_j = 4.57    # conversion from J to umol quanta
        self.dheat = 21.5E-6     # molecular diffusivity for heat
        self.DEG_TO_KELVIN = 273.15
        self.RGAS = 8.314
        self.leaf_width = leaf_width
        self.leaf_absorptance = leaf_absorptance

    def calc_leaf_temp(self, tleaf=None, tair=None, gs=None, par=None, vpd=None,
                       pressure=None, wind=None, leaf_width=None):

        P = PenmanMonteith(self.leaf_width, self.leaf_absorptance)

        tleaf_k = tleaf + self.DEG_TO_KELVIN
        tair_k = tair + self.DEG_TO_KELVIN

        air_density = pressure * 1000.0 / (287.058 * tair_k)
        cmolar = pressure * 1000.0 / (self.RGAS * tair_k)
        rnet_iso = P.calc_rnet(pressure, par, tair, tair_k, tleaf_k, vpd)

        (grn, gh, gbH, gv) = P.calc_conductances(tair_k, tleaf, tair, pressure,
                                                wind, gs, cmolar)
        (et, le_et) = P.calc_et(tleaf, tair, gs, vpd, pressure, wind, par,
                                    gh, gv, rnet_iso)

        # D6 in Leuning
        Y = 1.0 / (1.0 + grn / gbH)

        # sensible heat exchanged between leaf and surroundings
        sensible_heat = Y * (rnet_iso - le_et)

        # leaf-air temperature difference recalculated from energy balance.
        # (same equation as above!)
        new_Tleaf = (tair + sensible_heat /
                     (self.cp * air_density * (gbH / cmolar)))

        return (new_Tleaf, et, gbH, gv)


if __name__ == '__main__':

    tleaf = 21.5
    tair = 20.0
    gs = 0.15
    par = 1000
    vpd = 2.0
    pressure = 101.0
    wind = 2.0
    leaf_width = 0.02
    leaf_absorptance = 0.5 # leaf absorptance of solar radiation [0,1]


    L = LeafEnergyBalance(leaf_width, leaf_absorptance)
    new_Tleaf, et, gbh, gv = L.calc_leaf_temp(tleaf, tair, gs, par, vpd, pressure,
                                    wind, leaf_width)

    print new_Tleaf, et, et*18*0.001*86400.
