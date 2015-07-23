#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Iteratively solve leaf temp, ci, gs and An.

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (13.08.2012)"
__email__ = "mdekauwe@gmail.com"

import sys
import numpy as np
import os

def medlyn(g0, g1, vpd, A, Ci):
    gs = g0 + 1.6 * (1.0 + g1 / np.sqrt(vpd)) * A / Ci

    return gs

# The net leaf energy balance, given that we know Tleaf, gs
def calc_leaf_energy_balance(Tleaf = 21.5, Tair = 20, 
                              gs = 0.15,
                              PPFD = 1500, VPD = 2, Patm = 101,
                              Wind = 2, Wleaf = 0.02,
                              StomatalRatio = 1,   # 2 for amphistomatous
                              LeafAbs = 0.5,   # in shortwave range, much less than PAR
                              returnwhat=c("balance","fluxes")){


  returnwhat <- match.arg(returnwhat)

  # Constants
  Boltz <- 5.67 * 10^-8     # w M-2 K-4
  Emissivity <- 0.95        # -
  LatEvap <- 2.54           # MJ kg-1
  CPAIR <- 1010.0           # J kg-1 K-1

  H2OLV0 <- 2.501e6         # J kg-1
  H2OMW <- 18e-3            # J kg-1
  AIRMA <- 29.e-3           # mol mass air (kg/mol)
  AIRDENS <- 1.204          # kg m-3
  UMOLPERJ <- 4.57
  DHEAT <- 21.5e-6          # molecular diffusivity for heat



  # Density of dry air
  AIRDENS <- Patm*1000/(287.058 * Tk(Tair))

  # Latent heat of water vapour at air temperature (J mol-1)
  LHV <- (H2OLV0 - 2.365E3 * Tair) * H2OMW

  # Const s in Penman-Monteith equation  (Pa K-1)
  SLOPE <- (esat(Tair + 0.1) - esat(Tair)) / 0.1

  # Radiation conductance (mol m-2 s-1)
  Gradiation <- 4.*Boltz*Tk(Tair)^3 * Emissivity / (CPAIR * AIRMA)

  # See Leuning et al (1995) PC&E 18:1183-1200 Appendix E
  # Boundary layer conductance for heat - single sided, forced convection
  CMOLAR <- Patm*1000 / (8.314 * Tk(Tair))   # .Rgas() in package...
  Gbhforced <- 0.003 * sqrt(Wind/Wleaf) * CMOLAR

  # Free convection
  GRASHOF <- 1.6E8 * abs(Tleaf-Tair) * (Wleaf^3) # Grashof number
  Gbhfree <- 0.5 * DHEAT * (GRASHOF^0.25) / Wleaf * CMOLAR

  # Total conductance to heat (both leaf sides)
  Gbh <- 2*(Gbhfree + Gbhforced)

  # Heat and radiative conductance
  Gbhr <- Gbh + 2*Gradiation

  # Boundary layer conductance for water (mol m-2 s-1)
  Gbw <- StomatalRatio * 1.075 * Gbh  # Leuning 1995
  gw <- gs*Gbw/(gs + Gbw)

  # Longwave radiation
  # (positive flux is heat loss from leaf)
  Rlongup <- Emissivity*Boltz*Tk(Tleaf)^4

  # Rnet
  Rsol <- 2*PPFD/UMOLPERJ   # W m-2
  Rnet <- LeafAbs*Rsol - Rlongup   # full

  # Isothermal net radiation (Leuning et al. 1995, Appendix)
  ea <- esat(Tair) - 1000*VPD
  ema <- 0.642*(ea/Tk(Tair))^(1/7)
  Rnetiso <- LeafAbs*Rsol - (1 - ema)*Boltz*Tk(Tair)^4 # isothermal net radiation

  # Isothermal version of the Penmon-Monteith equation
  GAMMA <- CPAIR*AIRMA*Patm*1000/LHV
  ET <- (1/LHV) * (SLOPE * Rnetiso + 1000*VPD * Gbh * CPAIR * AIRMA) / (SLOPE + GAMMA * Gbhr/gw)

  # Latent heat loss
  lambdaET <- LHV * ET

  # Heat flux calculated using Gradiation (Leuning 1995, Eq. 11)
  Y <- 1/(1 + Gradiation/Gbh)
  H2 <- Y*(Rnetiso - lambdaET)

  # Heat flux calculated from leaf-air T difference.
  # (positive flux is heat loss from leaf)
  H <- -CPAIR * AIRDENS * (Gbh/CMOLAR) * (Tair - Tleaf)

  # Leaf-air temperature difference recalculated from energy balance.
  # (same equation as above!)
  Tleaf2 <- Tair + H2/(CPAIR * AIRDENS * (Gbh/CMOLAR))

  # Difference between input Tleaf and calculated, this will be minimized.
  EnergyBal <- Tleaf - Tleaf2

  if(returnwhat == "balance")return(EnergyBal)

  if(returnwhat == "fluxes"){

    l <- data.frame(ELEAFeb=1000*ET, Gradiation=Gradiation, Rsol=Rsol, Rnetiso=Rnetiso, Rlongup=Rlongup, H=H, lambdaET=lambdaET, gw=gw, Gbh=Gbh, H2=H2, Tleaf2=Tleaf2)
    return(l)
  }
}



def main(tair, vpd, wind, leaf_width, leaf_absorptance):











if __name__ == '__main__':

    tair = 25.0
    vpd = 1.0
    wind = 2.0
    leaf_width = 0.02
    leaf_absorptance = 0.86 # leaf absorptance of solar radiation [0,1]

    main(tair, vpd, wind, leaf_width, leaf_absorptance)
