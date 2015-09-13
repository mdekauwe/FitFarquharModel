#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model of C3 photosynthesis, this is passed to fitting function and we are
optimising Jmax25, Vcmax25, Rd, Eaj, Eav, deltaS

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (13.08.2012)"
__email__ = "mdekauwe@gmail.com"

import sys
import numpy as np
import os


class FarquharC3(object):
    """
    Rate of photosynthesis in a leaf depends on the the rates of
    carboxylation (Ac) and the regeneration of ribulose-1,5-bisphosphate (RuBP)
    catalysed by the enzyme RUBISCO (Aj). This class returns the net leaf
    photosynthesis (An) which is the minimum of this two limiting processes
    less the rate of mitochondrial respiration in the light (Rd). We are
    ignoring the the "export" limitation (Ap) which could occur under high
    levels of irradiance.

    Model assumes conductance between intercellular space and the site of
    carboxylation is zero. The models parameters Vcmax, Jmax, Rd along with
    the calculated values for Kc, Ko and gamma star all vary with temperature.
    The parameters Jmax and Vcmax are typically fitted with a temperature
    dependancy function, either an exponential Arrheniuous or a peaked
    function, i.e. the Arrhenious function with a switch off point.


    All calculations in Kelvins...

    References:
    -----------
    * De Pury and Farquhar, G. D. (1997) Simple scaling of photosynthesis from
      leaves to canopies without the errors of big-leaf models. Plant Cell and
      Environment, 20, 537-557.
    * Farquhar, G.D., Caemmerer, S. V. and Berry, J. A. (1980) A biochemical
      model of photosynthetic CO2 assimilation in leaves of C3 species. Planta,
      149, 78-90.
    * Medlyn, B. E., Dreyer, E., Ellsworth, D., Forstreuter, M., Harley, P.C.,
      Kirschbaum, M.U.F., Leroux, X., Montpied, P., Strassemeyer, J.,
      Walcroft, A., Wang, K. and Loustau, D. (2002) Temperature response of
      parameters of a biochemically based model of photosynthesis. II.
      A review of experimental data. Plant, Cell and Enviroment 25, 1167-1179.
    """

    def __init__(self, peaked_Jmax=False, peaked_Vcmax=False, Oi=210.0, 
                 gamstar25=42.75, Kc25=404.9, Ko25=278.4, Ec=79430.0,
                 Eo=36380.0, Eag=37830.0, theta_hyperbol=0.9995,
                 theta_J=0.7, force_vcmax_fit_pts=None,
                 alpha=None, quantum_yield=0.3, absorptance=0.8,
                 change_over_pt=None, model_Q10=False):
        """
        Parameters
        ----------
        Oi : float
            intercellular concentration of O2 [mmol mol-1]
        gamstar25 : float
            CO2 compensation point - base rate at 25 deg C / 298 K [umol mol-1]
        Kc25 : float
            Michaelis-Menten coefficents for carboxylation by Rubisco at
            25degC [umol mol-1] or 298 K
        Ko25: float
            Michaelis-Menten coefficents for oxygenation by Rubisco at
            25degC [mmol mol-1]. Note value in Bernacchie 2001 is in mmol!!
            or 298 K
        Ec : float
            Activation energy for carboxylation [J mol-1]
        Eo : float
            Activation energy for oxygenation [J mol-1]
        Eag : float
            Activation energy at CO2 compensation point [J mol-1]
        RGAS : float
            Universal gas constant [J mol-1 K-1]
        theta_hyperbol : float
            Curvature of the light response.
            See Peltoniemi et al. 2012 Tree Phys, 32, 510-519
        theta_J : float
            Curvature of the light response
        alpha : float
            Leaf quantum yield (initial slope of the A-light response curve)
            [mol mol-1]
        peaked_Jmax : logical
            Use the peaked Arrhenius function (if true)
        peaked_Vcmax : logical
            Use the peaked Arrhenius function (if true)

        force_vcmax_fit_pts : None or npts
            Force Ac fit for first X points
        change_over_pt : None or value of Ci
            Explicitly set the transition point between Aj and Ac.

        """
        self.peaked_Jmax = peaked_Jmax
        self.peaked_Vcmax = peaked_Vcmax
        self.deg2kelvin = 273.15
        self.RGAS = 8.314
        self.Oi = Oi
        self.gamstar25 = gamstar25
        self.Kc25 = Kc25
        self.Ko25 = Ko25
        self.Ec = Ec
        self.Eo = Eo
        self.Eag = Eag
        self.theta_hyperbol = theta_hyperbol
        self.theta_J = theta_J
        if alpha is not None:
            self.alpha = alpha
        else:
            self.alpha = quantum_yield * absorptance # (Medlyn et al 2002)

        self.force_vcmax_fit_pts = force_vcmax_fit_pts
        self.change_over_pt = change_over_pt
        self.model_Q10 = model_Q10

    def calc_photosynthesis(self, Ci=None, Tleaf=None, Par=None, Jmax=None,
                            Vcmax=None, Jmax25=None, Vcmax25=None, Rd=None,
                            Rd25=None, Q10=None, Eaj=None, Eav=None,
                            deltaSj=None, deltaSv=None, Hdv=200000.0,
                            Hdj=200000.0, Ear=None):
        """
        Parameters
        ----------
        Ci : float
            intercellular CO2 concentration [umol mol-1]
        Tleaf : float
            leaf temp [deg K]

        * Optional args:
        Jmax : float
            potential rate of electron transport at measurement temperature
            [deg K]
        Vcmax : float
            max rate of rubisco activity at measurement temperature [deg K]
        Jmax25 : float
            potential rate of electron transport at 25 deg or 298 K
        Vcmax25 : float
            max rate of rubisco activity at 25 deg or 298 K
        Rd : float
            Day "light" respiration [umol m-2 time unit-1]
        Q10 : float
            ratio of respiration at a given temperature divided by respiration
            at a temperature 10 degrees lower
        Eaj : float
            activation energy for the parameter [J mol-1]
        Eav : float
            activation energy for the parameter [J mol-1]
        deltaSj : float
            entropy factor [J mol-1 K-1)
        deltaSv : float
            entropy factor [J mol-1 K-1)
        HdV : float
            Deactivation energy for Vcmax [J mol-1]
        Hdj : float
            Deactivation energy for Jmax [J mol-1]
        Rd25 : float
            Estimate of respiration rate at the reference temperature 25 deg C
             or 298 K [deg K]
        Par : float
            PAR [umol m-2 time unit-1]. Default is not to supply PAR, with
            measurements taken under light saturation.

        Returns:
        --------
        An : float
            Net leaf assimilation rate [umol m-2 s-1]
        Acn : float
            Net rubisco-limited leaf assimilation rate [umol m-2 s-1]
        Ajn : float
            Net RuBP-regeneration-limited leaf assimilation rate [umol m-2 s-1]
        """
        self.check_supplied_args(Jmax, Vcmax, Rd, Jmax25, Vcmax25, Rd25)

        # calculate temp dependancies of Michaelis–Menten constants for CO2, O2
        Km = self.calc_michaelis_menten_constants(Tleaf)

        # Effect of temp on CO2 compensation point
        gamma_star = self.arrh(self.gamstar25, self.Eag, Tleaf)

        # Calculations at 25 degrees C or the measurement temperature
        if Rd25 is not None:
            Rd = self.calc_resp(Tleaf, Q10, Rd25, Ear)

        # Calculate temperature dependancies on Vcmax and Jmax
        if Vcmax25 is not None:
            # Effect of temperature on Vcmax and Jamx
            if self.peaked_Vcmax:
                Vcmax = self.peaked_arrh(Vcmax25, Eav, Tleaf, deltaSv, Hdv)
            else:
                Vcmax = self.arrh(Vcmax25, Eav, Tleaf)

        if Jmax25 is not None:
            if self.peaked_Jmax:
                Jmax = self.peaked_arrh(Jmax25, Eaj, Tleaf, deltaSj, Hdj)
            else:
                Jmax = self.arrh(Jmax25, Eaj, Tleaf)

        # actual rate of electron transport, a function of absorbed PAR
        if Par is not None:
            J = self.quadratic(a=self.theta_J, b=-(self.alpha * Par + Jmax),
                               c=self.alpha * Par * Jmax)
        # All measurements are calculated under saturated light!!
        else:
            J = Jmax

        # Rubisco carboxylation limited rate of photosynthesis
        Ac = self.assim(Ci, gamma_star, a1=Vcmax, a2=Km)

        # Light-limited rate of photosynthesis allowed by RuBP regeneration
        Aj = self.assim(Ci, gamma_star, a1=J/4.0, a2=2.0*gamma_star)

        # option for the user to specify the transition point
        if self.change_over_pt is not None:
            A = np.where(Ci<self.change_over_pt, Ac, Aj)
        else:
            # Photosynthesis estimated using hyperbolic minimum of Ac and Aj to
            # effectively smooth over discontinuity when moving from light/electron
            # transport limited to rubisco limited photosynthesis
            # except if Ci < 150 in which case it is always Rubisco limited
            arg = ((Ac + Aj - \
                   np.sqrt((Ac + Aj)**2 - 4.0 * self.theta_hyperbol * Ac * Aj)) /
                  (2.0 * self.theta_hyperbol))

            # By default we assume a everything under Ci<150 is Ac limited
            A = np.where(Ci < 150.0, Ac, arg)

            # Specifically for Angelica's data...force Ac fit through the first
            # X points.
            if self.force_vcmax_fit_pts is not None:
                indx = self.force_vcmax_fit_pts - 1 # indexed from zero
                A = np.where(Ci <= Ci[indx] , Ac, A)
                indx += 1 # use all the rest for Aj limited...
                A = np.where(Ci >= Ci[indx] , Aj, A)

            # Force the fit through at least the final point-ish
            elif self.force_vcmax_fit_pts is None:
                A = np.where(Ci > np.max(Ci) - 10.0, Aj, A)
            else:
                err_msg = "error fitting, are you suppling the correct args?"
                raise AttributeError, err_msg

        # net assimilation rates.
        An = A - Rd
        Acn = Ac - Rd
        Ajn = Aj - Rd

        return An, Acn, Ajn

    def check_supplied_args(self, Jmax, Vcmax, Rd, Jmax25, Vcmax25, Rd25):
        """ Check the user supplied arguments, either they supply the values
        at 25 deg C, or the supply Jmax and Vcmax at the measurement temp. It
        is of course possible they accidentally supply both or a random
        combination, so raise an exception if so

        Parameters
        ----------
        Jmax : float
            potential rate of electron transport at measurement temperature
            [deg K]
        Vcmax : float
            max rate of rubisco activity at measurement temperature [deg K]
        Rd : float
            Day "light" respiration [umol m-2 time unit-1]
        Jmax25 : float
            potential rate of electron transport at 25 deg or 298 K
        Vcmax25 : float
            max rate of rubisco activity at 25 deg or 298 K
        Rd25 : float
            Estimate of respiration rate at the reference temperature 25 deg C
             or 298 K [deg K]

        Returns
        -------
        Nothing
        """
        try:
            if (Rd25 is not None and Jmax25 is not None and
                Vcmax25 is not None and Vcmax is None and
                Jmax is None and Rd is None):

                return
            elif (Rd25 is None and Jmax25 is None and
                  Vcmax25 is None and Vcmax is not None and
                  Jmax is not None and Rd is not None):

                return

        except AttributeError:
            err_msg = "Supplied arguments are a mess!"
            raise AttributeError, err_msg

    def calc_michaelis_menten_constants(self, Tleaf):
        """ Michaelis-Menten constant for O2/CO2, Arrhenius temp dependancy
        Parameters:
        ----------
        Tleaf : float
            leaf temperature [deg K]

        Returns:
        Km : float

        """
        Kc = self.arrh(self.Kc25, self.Ec, Tleaf)
        Ko = self.arrh(self.Ko25, self.Eo, Tleaf)

        Km = Kc * (1.0 + self.Oi / Ko)

        return Km

    def arrh(self, k25, Ea, Tk):
        """ Temperature dependence of kinetic parameters is described by an
        Arrhenius function.

        Parameters:
        ----------
        k25 : float
            rate parameter value at 25 degC or 298 K
        Ea : float
            activation energy for the parameter [J mol-1]
        Tk : float
            leaf temperature [deg K]

        Returns:
        -------
        kt : float
            temperature dependence on parameter

        References:
        -----------
        * Medlyn et al. 2002, PCE, 25, 1167-1179.
        """
        return k25 * np.exp((Ea * (Tk - 298.15)) / (298.15 * self.RGAS * Tk))

    def peaked_arrh(self, k25, Ea, Tk, deltaS, Hd):
        """ Temperature dependancy approximated by peaked Arrhenius eqn,
        accounting for the rate of inhibition at higher temperatures.

        Parameters:
        ----------
        k25 : float
            rate parameter value at 25 degC or 298 K
        Ea : float
            activation energy for the parameter [J mol-1]
        Tk : float
            leaf temperature [deg K]
        deltaS : float
            entropy factor [J mol-1 K-1)
        Hd : float
            describes rate of decrease about the optimum temp [J mol-1]

        Returns:
        -------
        kt : float
            temperature dependence on parameter

        References:
        -----------
        * Medlyn et al. 2002, PCE, 25, 1167-1179.

        """
        arg1 = self.arrh(k25, Ea, Tk)
        arg2 = 1.0 + np.exp((298.15 * deltaS - Hd) / (298.15 * self.RGAS))
        arg3 = 1.0 + np.exp((Tk * deltaS - Hd) / (Tk * self.RGAS))

        return arg1 * arg2 / arg3

    def assim(self, Ci, gamma_star, a1, a2):
        """calculation of photosynthesis with the limitation defined by the
        variables passed as a1 and a2, i.e. if we are calculating vcmax or
        jmax limited assimilation rates.

        Parameters:
        ----------
        Ci : float
            intercellular CO2 concentration.
        gamma_star : float
            CO2 compensation point in the abscence of mitochondrial respiration
        a1 : float
            variable depends on whether the calculation is light or rubisco
            limited.
        a2 : float
            variable depends on whether the calculation is light or rubisco
            limited.

        Returns:
        -------
        assimilation_rate : float
            assimilation rate assuming either light or rubisco limitation.
        """
        return a1 * (Ci - gamma_star) / (a2 + Ci)

    def calc_resp(self, Tleaf=None, Q10=None, Rd25=None, Ear=None, Tref=25.0):
        """ Calculate leaf respiration accounting for temperature dependence.

        Parameters:
        ----------
        Rd25 : float
            Estimate of respiration rate at the reference temperature 25 deg C
            or or 298 K
        Tref : float
            reference temperature
        Q10 : float
            ratio of respiration at a given temperature divided by respiration
            at a temperature 10 degrees lower
        Ear : float
            activation energy for the parameter [J mol-1]
        Returns:
        -------
        Rt : float
            leaf respiration

        References:
        -----------
        Tjoelker et al (2001) GCB, 7, 223-230.
        """
        if self.model_Q10:
            Rd = Rd25 * Q10**(((Tleaf - self.deg2kelvin) - Tref) / 10.0)
        else:
            Rd = self.arrh(Rd25, Ear, Tleaf)

        return Rd

    def quadratic(self, a=None, b=None, c=None):
        """ minimilist quadratic solution as root for J solution should always
        be positive, so I have excluded other quadratic solution steps. I am
        only returning the smallest of the two roots

        Parameters:
        ----------
        a : float
            co-efficient
        b : float
            co-efficient
        c : float
            co-efficient

        Returns:
        -------
        val : float
            positive root
        """

        d = b**2 - 4.0 * a * c # discriminant
        # if < 0.0 then an imaginary root was found
        d = np.where(np.logical_or(d<=0, np.any(np.isnan(d))), -999.9, d)
        root1 = np.where(d>0.0, (-b - np.sqrt(d)) / (2.0 * a), d)
        #root2 = np.where(d>0.0, (-b + np.sqrt(d)) / (2.0 * a), d)

        return root1
