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

class LeafEnergyBalance(object):
    """
        Calculate the net leaf enegy balance, given Tleaf and gs
        - only works for single sided leaves!
    """

    def __init__(self):

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

    def main(self, tleaf=None, tair=None, gs=None, par=None, vpd=None,
                    pressure=None, wind=None, leaf_width=None,
                    leaf_absorptance=None):

        tleaf_k = tleaf + self.DEG_TO_KELVIN
        tair_k = tair + self.DEG_TO_KELVIN
        esat = self.calc_esat(tair, pressure)
        esat_inc = self.calc_esat(tair + 0.1, pressure)

        # density of dry air
        air_density = pressure * 1000.0 / (287.058 * tair_k)
        cmolar = pressure * 1000.0 / (self.RGAS * tair_k)

        rnet_iso = self.calc_rnet(esat, leaf_absorptance, par, tair_k, tleaf_k,
                                  vpd)

        # radiation conductance (mol m-2 s-1)
        (grad, gbh,
         gbhr, gw, gv) = self.calc_conductances(tair_k, tleaf, tair, pressure,
                                            wind, leaf_width, gs, cmolar)


        (et, lambda_et) = self.calc_isothermal_transp(tair, vpd, pressure, esat,
                                                      esat_inc, rnet_iso,
                                                      grad, gbh, gbhr, gw)

        # D6 in Leuning
        Y = 1.0 / (1.0 + grad / gbh)

        # sensible heat exchanged between leaf and surroundings
        sensible_heat = Y * (rnet_iso - lambda_et)

        # leaf-air temperature difference recalculated from energy balance.
        # (same equation as above!)
        new_Tleaf = (tair + sensible_heat /
                     (self.cp * air_density * (gbh / cmolar)))

        return (new_Tleaf, et, gbh, gv)

    def calc_isothermal_transp(self, tair, vpd, pressure, esat, esat_inc,
                                rnet_iso, grad, gbh, gbhr, gw):
        """ Isothermal form of the Penman-Monteith equation """

        kpa_2_pa = 1000.

        # latent heat of water vapour at air temperature (j mol-1)
        lhv = (self.h2olv0 - 2.365e3 * tair) * self.h2omw

        # (pa k-1)
        slope = (esat_inc - esat) / 0.1

        # psychrometric constant
        gamma = self.cp * self.air_mass * pressure * 1000.0 / lhv

        # Y cancels in eqn 10
        arg1 = (slope * rnet_iso + (vpd * kpa_2_pa) * gbh * self.cp *
                self.air_mass)
        arg2 = slope + gamma * gbhr / gw
        et = arg1 / arg2

        # latent heat loss
        LE_et = et

        # et units = mol m-2 s-1,
        # multiply by 18 (grams)* 0.001 (grams to kg) * 86400.
        # to get to kg m2 d-1 or mm d-1
        return et / lhv, LE_et

    def calc_conductances(self, tair_k, tleaf, tair, pressure, wind, leaf_width,
                          gs, cmolar):
        """
        Leuning 1995, appendix E
        """

        # radiation conductance (mol m-2 s-1)
        grad = ((4.0 * self.sigma * tair_k**3 * self.emissivity_leaf) /
                (self.cp * self.air_mass))

        # boundary layer conductance for 1 side of leaf from forced convection
        gbhw = 0.003 * math.sqrt(wind / leaf_width) * cmolar

        # grashof number
        grashof_num = 1.6e8 * math.fabs(tleaf - tair) * leaf_width**3

        # boundary layer conductance for free convection
        gbhf = 0.5 * self.dheat * (grashof_num**0.25) / leaf_width * cmolar

        # total conductance to heat
        gbh = 2.0 * (gbhf + gbhw)

        # total conductance to heat for one side of the leaf
        gbh = gbhw + gbhf

        # ... for hypostomatous leaves only g^H should be doubled and the
        # single-sided value used for gbw

        # heat and radiative conductance
        gbhr = 2.0 * (gbh + grad)

        # boundary layer conductance for water (mol m-2 s-1)
        gbw = 1.075 * gbh
        gw = gs * gbw / (gs + gbw)

        # total conductance for water vapour
        gsv = 1.57 * gs
        gv = (gbw * gsv) / (gbw + gsv)

        return (grad, gbh, gbhr, gw, gv)

    def calc_rnet(self, esat, leaf_absorptance, par, tair_k, tleaf_k, vpd):

        kpa_2_pa = 1000.0
        umol_m2_s_to_W_m2 = 2.0 / self.umol_to_j

        par *= umol_m2_s_to_W_m2

        # atmospheric water vapour pressure (Pa)
        ea = esat - (vpd * kpa_2_pa)

        # eqn D4
        emissivity_atm = 0.642 * (ea / tair_k)**(1.0 / 7.0)

        rlw_down = emissivity_atm * self.sigma * tair_k**4
        rlw_up = self.emissivity_leaf * self.sigma * tleaf_k**4
        isothermal_net_lw = rlw_up - rlw_down

        # isothermal net radiation
        return (leaf_absorptance * par - isothermal_net_lw)

    def calc_esat(self, temp, pressure):
        """
        saturation vapor pressure (kPa)

        Values of saturation vapour pressure from the Tetens formula are
        within 1 Pa of the exact values.

        but see Jones 1992 too.
        """
        Tk = temp + self.DEG_TO_KELVIN
        A = 17.27
        T_star = 273.0
        T_dash = 36.0
        es_T_star = 0.611
        kpa_2_pa = 1000.0

        esat = es_T_star * math.exp(A * (Tk - T_star) / (Tk - T_dash))
        esat *= kpa_2_pa

        return esat


if __name__ == '__main__':

    tleaf = 21.5
    tair = 20.0
    gs = 0.15
    par = 1000
    vpd = 2.0
    pressure = 101.0
    wind = 2.0
    leaf_width = 0.02
    leaf_absorptance = 0.86 # leaf absorptance of solar radiation [0,1]


    L = LeafEnergyBalance()
    new_Tleaf, et, gbh, gv = L.main(tleaf, tair, gs, par, vpd, pressure,
                                    wind, leaf_width, leaf_absorptance)

    print new_Tleaf, et, et*18*0.001*86400.
