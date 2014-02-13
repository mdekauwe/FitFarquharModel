#!/usr/bin/env python

"""
Using a series of A-Ci curves, the Farquhar model and the pymc libs fit Jmax25, 
Vcmax25, Rd25 and their temperature dependancies. *Docstrings are wrong as I am 
testing different ways of fitting this*...testing constant factors between 
leaves for J/V ratios.

Currently...

Jmax25, Vcmax25 & Rd25 vary by leaf due to varying leaf N, but trailing fitting
a single Vcmax25 value per leaf and a constant factor between this and Jmax25
and Rd25. Thus we are using a dummy variables to fit Vcmax25 values per leaf,
where N leaves might be 3+, but we are fitting all the curves per species to 
resolves the temperature dependancies.

That's all folks.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (03.02.2014)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import numpy as np
import csv
from lmfit import minimize, Parameters
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from math import fabs
import pymc

class FitMe(object):
    """
    Basic fitting class, contains some generic methods which are used by the
    fitting routines, e.g. plotting, file reporting etc.
    
    """
    def __init__(self, results_dir=None, data_dir=None, plot_dir=None, 
                 trace_dir=None, delimiter=","):
        """
        Parameters
        ----------
        results_dir : string
            output directory path for the result to be written
        data_dir : string
            input directory path where measured A-Ci files live
        plot_dir : string
            directory to save plots of various fitting routines
        """
        self.trace_dir = trace_dir
        self.results_dir = results_dir
        self.data_dir = data_dir 
        self.plot_dir = plot_dir    
        self.deg2kelvin = 273.15
        self.delimiter = delimiter
        
        
    def main(self, MCMC, infname_tag="*.csv"):   
        """ Loop over all our A-Ci measured curves and fit the Farquhar model
        parameters to these data. 
        
        Parameters
        ----------
        
        """
        for fname in glob.glob(os.path.join(self.data_dir, infname_tag)):
            df = self.read_data(fname)
            # sort each curve by Ci...
            df = self.sort_curves_by_ci(df)
            for group in np.unique(df["fitgroup"]):
                df_group = df[df["fitgroup"]==group]
                df_group.index = range(len(df_group)) # need to reindex slice
                ofname = os.path.join(self.results_dir, "%s_%s.csv" % \
                                (df_group["Species"][0], group))
                (df_group) = self.setup_dummy_variable(df_group)
                
                trace_ofname = os.path.join(self.trace_dir, 
                                            "mcmc_%s.pickle" % (str(group)))
                
                MC = MCMC.call_mcmc(df_group, trace_ofname)
                MCMC.save_fits(MC, os.path.join(self.results_dir, 
                                        "mcmc_fit_result_%s" % (str(group))))
               
    def sort_curves_by_ci(self, df):
        """ Sort curves by Ci (low to high) helps with output plotting,
            shouldn't matter to fitting, but it makes more sense to be sorted"""
        df_sorted = pd.DataFrame()
        for curve_num in np.unique(df["Curve"]):
            curve_df = df[df["Curve"]==curve_num]
            curve_df = curve_df.sort(['Ci'], ascending=True)
            df_sorted = df_sorted.append(curve_df)
        df_sorted.index = range(len(df_sorted)) # need to reindex slice
        
        return df_sorted

    def read_data(self, fname):
        """ Reads in the A-Ci data if infile_type="aci" is true, otherwise this
        reads in the fitted results... 
        
        For A-Ci data, code expects a format of:
        -> Curve, Tleaf, Ci, Photo, Species, Season, Leaf
        
        Parameters
        ----------
        fname : string
            input file name, expecting csv file.
        
        Returns: 
        --------
        data : array
            numpy array containing the data
        """
        
        df = pd.read_csv(fname, sep=self.delimiter, header=0)
        
        # change temperature to kelvins
        df["Tleaf"] += self.deg2kelvin
            
        return df
    
    def setup_dummy_variable(self, df):
        """ Setup lmfit Parameters object
        
        Parameters
        ----------
        df : dataframe
            dataframe containing all the A-Ci curves.
        
        Returns
        -------
        params : object
            lmfit object containing parameters to fit
        """
        
        
        # Need to loop over all the leaves, fitting separate Jmax25, Vcmax25 
        # and Rd25 parameter values by leaf
        
        for leaf_num in np.unique(df["Leaf"]):
            
            # Need to build dummy variable identifier for each leaf.
            col_id = "f_%d" % (leaf_num)
            temp = df["Leaf"]
            temp = np.where(temp==leaf_num, 1.0, 0.0)
            df[col_id] = temp
       
        return df
 
 
class FarquharMCMC(object):
    """ General(ish) MCMC fitting class, idea is that the user will subclass at 
    the very least the priors method and adjust as they feel fit.
    """
    def __init__(self, call_model=None, iterations=None, burn=None, thin=None,
                 progress_bar=True):
        
        self.iterations = iterations
        self.burn = burn
        self.thin = thin
        self.call_model = call_model
        self.progress_bar = progress_bar
    
    def call_mcmc(self, df, trace_ofname):   
        
        # optimise starting point for chain
        print "Optimising starting position for chain..."
        model = self.make_model(df)
        map = pymc.MAP(model)
        map.fit()
        
        # Use these parameter estimates in the fitting
        MC = pymc.MCMC(map.variables, db='pickle', dbname=trace_ofname)
        
        MC.sample(iter=self.iterations, burn=self.burn, thin=self.thin, 
                  progress_bar=self.progress_bar)
        
        return MC
        
            
    
    def save_fits(self, MC, ofname, variables=None):
        """ write parameter fits, CI to csv file """
        
        if variables is None:
             MC.write_csv(ofname)
        else:
            # only works in git version
            MC.write_csv(ofname, variables=vars)
    
    def set_priors(self, df):
        """ default priors
        
        When setting normals I am assuming that sigma = range / 4 to set these 
        priors
        """
        # mu=25, range=(5-50)
        Vcvals = [pymc.TruncatedNormal('Vcmax25_%d' % (i), \
                  mu=25.0, tau=1.0/11.25**2, a=0.0, b=650.0) \
                  for i in np.unique(df["Leaf"])]
        
        
        # mu=1.8, range=(0.8-2.8)
        Jfac = pymc.TruncatedNormal('Jfac', mu=1.8, tau=1.0/0.5**2, \
                                    a=0.0, b=5.0)
        
        # broad prior
        Rdfac = pymc.Uniform('Rdfac', lower=0.005, upper=0.05)
        
        # mu=40000, range=(20000-60000)
        Eaj = pymc.TruncatedNormal('Eaj', mu=40000.0, tau=1.0/10000.0**2, 
                                    a=0.0, b=199999.9)
        
        # mu=60000, range=(40000-80000)
        Eav = pymc.TruncatedNormal('Eav', mu=60000.0, tau=1.0/10000.0**2, 
                                    a=0.0, b=199999.9)
        
        # mu=34000, range=(20000-60000)
        Ear = pymc.TruncatedNormal('Ear', mu=34000.0, tau=1.0/10000.0**2, 
                                    a=0.0, b=199999.9)
       
        # mu=640, range=(620-660)       
        delSj = pymc.TruncatedNormal('delSj', mu=640.0, tau=1.0/10.0**2, \
                                      a=300.0, b=800.0)
        
        # mu=640, range=(620-660)     
        delSv = pymc.TruncatedNormal('delSv', mu=640.0, tau=1.0/10.0**2, \
                                      a=300.0, b=800.0)
        
        """
        log_mu = np.log(25.0)
        log_sigma = np.log(11.25)
        log_tau = 1.0/log_sigma**2
        Vcvals = [pymc.Lognormal('Vcmax25_%d' % (i), mu=log_mu, tau=log_tau)\
                  for i in np.unique(df["Leaf"])]
        
        log_mu = np.log(1.8)
        log_sigma = np.log(0.5)
        log_tau = 1.0/log_sigma**2
        Jfac = pymc.Lognormal('Jfac', mu=log_mu, tau=log_tau)
        
        Rdfac = pymc.Uniform('Rdfac', lower=0.005, upper=0.05)
        
        log_mu = np.log(40000.0)
        log_sigma = np.log(20000.0)
        log_tau = 1.0/log_sigma**2
        Eaj = pymc.Lognormal('Eaj', mu=log_mu, tau=log_tau)
        
        log_mu = np.log(60000.0)
        log_sigma = np.log(20000.0)
        log_tau = 1.0/log_sigma**2
        Eav = pymc.Lognormal('Eav', mu=log_mu, tau=log_tau)
        
        log_mu = np.log(34000)
        log_sigma = np.log(15000.0)
        log_tau = 1.0/log_sigma**2
        Ear = pymc.Lognormal('Ear', mu=log_mu, tau=log_tau)
        
        log_mu = np.log(640.0)
        log_sigma = np.log(50.0)
        log_tau = 1.0/log_sigma**2
        delSj = pymc.Lognormal('delSj', mu=log_mu, tau=log_tau)
        
        log_mu = np.log(640.0)
        log_sigma = np.log(50.0)
        log_tau = 1.0/log_sigma**2
        delSv = pymc.Lognormal('delSv', mu=log_mu, tau=log_tau)
        """
        
        return Vcvals, Jfac, Rdfac, Eaj, Eav, Ear, delSj, delSv
    
    def make_model(self, df):
        """ Setup 'model factory' - which exposes various attributes to PYMC 
        call """
       
        (Vcvals, Jfac, Rdfac, Eaj, Eav, Ear, delSj, delSv) = self.set_priors(df)
        
        @pymc.deterministic
        def func(Vcvals=Vcvals, Jfac=Jfac, Rdfac=Rdfac, Eaj=Eaj, Eav=Eav, 
                 Ear=Ear, delSj=delSj, delSv=delSv): 
            
            # Need to build dummy variables such that each leaf has access
            # to its corresponding PAR, temperature and Ci data.
            # For each curve create a column of 1's and 0's, which indicate
            # the leaves Ci, temp data.
            
            # These parameter values need to be arrays
            Jmax25 = np.zeros(len(df))
            Vcmax25 = np.zeros(len(df))
            Rd25 = np.zeros(len(df))
            
            # Need to build dummy variables.
            for index, i in enumerate(np.unique(df["Leaf"])):
                col_id = "f_%d" % (i)
                Vcmax25 += Vcvals[index] * df[col_id]
                Jmax25 += Vcvals[index] * Jfac * df[col_id]
                Rd25 += Vcvals[index] * Rdfac * df[col_id]
                #print Vcvals[index], Jfac, Rdfac, Eaj, Eav, Ear   
            Hdv = 200000.0
            Hdj = 200000.0
            if hasattr(df, "Par"):
                (An, Anc, Anj) = self.call_model(Ci=df["Ci"], Tleaf=df["Tleaf"], 
                                           Par=df["Par"], Jmax=None, Vcmax=None, 
                                           Jmax25=Jmax25, Vcmax25=Vcmax25, 
                                           Rd=None, Q10=None, Eaj=Eaj, Eav=Eav, 
                                           deltaSj=delSj, deltaSv=delSv, 
                                           Rd25=Rd25, Ear=Ear, Hdv=Hdv, Hdj=Hdj)
            else:
                (An, Anc, Anj) = self.call_model(Ci=df["Ci"], Tleaf=df["Tleaf"], 
                                           Par=None, Jmax=None, Vcmax=None, 
                                           Jmax25=Jmax25, Vcmax25=Vcmax25, 
                                           Rd=None, Q10=None, Eaj=Eaj, Eav=Eav, 
                                           deltaSj=delSj, deltaSv=delSv, 
                                           Rd25=Rd25, Ear=Ear, Hdv=Hdv, Hdj=Hdj)
            return An
        
        obs = df["Photo"]
        # Standard deviation is modelled with a Uniform prior
        obs_sigma = pymc.Uniform("obs_sigma", lower=0.0, upper=100.0, value=0.1)
        
        @pymc.deterministic
        def precision(obs_sigma=obs_sigma):
            # Precision, based on standard deviation
            return 1.0/obs_sigma**2
        
        like = pymc.Normal('like', mu=func, tau=precision, value=obs, 
                           observed=True)
        
        #obs_sigma = 0.0001 # assume obs are perfect
        #like = pymc.Normal('like', mu=func, tau=1.0/obs_sigma**2, value=obs, 
        #                   observed=True)
        
        return locals()
 
    def make_plots(self, df, MC):
        """ Make some plots to show how good our fitted model is to the data 
        
        * Plots A-Ci model fits vs. data
        * Residuals between fit and measured A
        
        Parameters
        ----------
        df : dataframe
            input A-Ci curve information 
        curve_num : int
            unique identifier to distinguish A-Ci curve
        An_fit : array
            best model fit using optimised parameters, Net leaf assimilation 
            rate [umol m-2 s-1]
        Anc_fit : array
            best model fit using optimised parameters, Net rubisco-limited leaf 
            assimilation rate [umol m-2 s-1]
        Anj_fit : array
            best model fit using optimised parameters, Net 
            RuBP-regeneration-limited leaf assimilation rate [umol m-2 s-1]
        result : object
            fitting result, param, std. error etc.
        """
        species = df["Species"][0]
        season = df["Season"][0]
        season = "all"
        leaf = df["Leaf"][0]
        
        # Get RMSE of total fit.
        
        Jmax25 = np.zeros(len(df))
        Vcmax25 = np.zeros(len(df))
        Rd25 = np.zeros(len(df))
        
        # Need to build dummy variables.
        for index, i in enumerate(np.unique(df["Leaf"])):
            col_id = "f_%d" % (i)
            Vcmax25 += MC.stats()['Vcmax25_%d' % (i)]['mean'] * df[col_id]
            Jmax25 += (MC.stats()['Vcmax25_%d' % (i)]['mean'] * 
                       MC.stats()['Jfac']['mean'] * df[col_id])
            Rd25 += (MC.stats()['Vcmax25_%d' % (i)]['mean'] * 
                       MC.stats()['Rdfac']['mean'] * df[col_id])           
        Eaj = MC.stats()['Eaj']['mean']
        delSj = MC.stats()['delSj']['mean']
        Eav = MC.stats()['Eav']['mean']
        delSv = MC.stats()['delSv']['mean']
        Ear = MC.stats()['Ear']['mean']
        Hdv = 200000.00000000
        Hdj = 200000.00000000
        
        if hasattr(df, "Par"):
            (An, Anc, Anj) = self.call_model(Ci=df["Ci"], Tleaf=df["Tleaf"], 
                                       Par=df["Par"], Jmax=None, Vcmax=None, 
                                       Jmax25=Jmax25, Vcmax25=Vcmax25, 
                                       Rd=None, Q10=None, Eaj=Eaj, Eav=Eav, 
                                       deltaSj=delSj, deltaSv=delSv, 
                                       Rd25=Rd25, Ear=Ear, Hdv=Hdv, Hdj=Hdj)
        else:
            (An, Anc, Anj) = self.call_model(Ci=df["Ci"], Tleaf=df["Tleaf"], 
                                       Par=None, Jmax=None, Vcmax=None, 
                                       Jmax25=Jmax25, Vcmax25=Vcmax25, 
                                       Rd=None, Q10=None, Eaj=Eaj, Eav=Eav, 
                                       deltaSj=delSj, deltaSv=delSv, 
                                       Rd25=Rd25, Ear=Ear, Hdv=Hdv, Hdj=Hdj)
        rmse = np.sqrt(np.mean((df["Photo"] - An)**2))
        
        print "**** RMSE = %f" % (rmse)
        print
        
        for curve_num in np.unique(df["Curve"]):
            curve_df = df[df["Curve"]==curve_num]
            i = curve_df["Leaf"].values[0]
            
            col_id = "f_%d" % (i)
            
            
            Vcmax25 = MC.stats()['Vcmax25_%d' % (i)]['mean']
            Rd25 = MC.stats()['Rdfac']['mean'] *  Vcmax25
            Jmax25 = MC.stats()['Jfac']['mean'] *  Vcmax25            
            Eaj = MC.stats()['Eaj']['mean']
            delSj = MC.stats()['delSj']['mean']
            Eav = MC.stats()['Eav']['mean']
            delSv = MC.stats()['delSv']['mean']
            Ear = MC.stats()['Ear']['mean']
            Hdv = 200000.00000000
            Hdj = 200000.00000000
            if hasattr(curve_df, "Par"):
                (An, Anc, Anj) = self.call_model(Ci=curve_df["Ci"], Tleaf=curve_df["Tleaf"], 
                                           Par=curve_df["Par"], Jmax=None, Vcmax=None, 
                                           Jmax25=Jmax25, Vcmax25=Vcmax25, Rd=None, 
                                           Q10=None, Eaj=Eaj, Eav=Eav, 
                                           deltaSj=delSj, deltaSv=delSv, Rd25=Rd25, 
                                           Ear=Ear, Hdv=Hdv, Hdj=Hdj)
            else:
                (An, Anc, Anj) = self.call_model(Ci=curve_df["Ci"], Tleaf=curve_df["Tleaf"], 
                                           Par=None, Jmax=None, Vcmax=None, 
                                           Jmax25=Jmax25, Vcmax25=Vcmax25, Rd=None, 
                                           Q10=None, Eaj=Eaj, Eav=Eav, 
                                           deltaSj=delSj, deltaSv=delSv, Rd25=Rd25, 
                                           Ear=Ear, Hdv=Hdv, Hdj=Hdj)
            residuals = curve_df["Photo"] - An
              
            ofname = "%s/%s_%s_%s_%s_fit_and_residual.png" % \
                     (self.plot_dir, species, season, leaf, curve_num)
            
        
            
            plt.rcParams['figure.subplot.hspace'] = 0.05
            plt.rcParams['figure.subplot.wspace'] = 0.05
            plt.rcParams['font.size'] = 10
            plt.rcParams['legend.fontsize'] = 10
            plt.rcParams['xtick.labelsize'] = 10.0
            plt.rcParams['ytick.labelsize'] = 10.0
            plt.rcParams['axes.labelsize'] = 10.0
        
            fig = plt.figure() 
        
            ax = fig.add_subplot(211)
            ax2 = fig.add_subplot(212)
        
            ax.plot(curve_df["Ci"], curve_df["Photo"], 
                    ls="", lw=1.5, marker="o", c="black")
            ax.plot(curve_df["Ci"], An, '-', c="red", linewidth=2, 
                    label="An")
            ax.plot(curve_df["Ci"], Anc, '--', c="green", linewidth=1, 
                    label="Anc")
            ax.plot(curve_df["Ci"], Anj, '--', c="blue", linewidth=1, 
                    label="Anj")
            ax.set_ylabel("Assimilation Rate")
            ax.axes.get_xaxis().set_visible(False)
            ax.set_xlim(0, 1600)
            ax.legend(numpoints=1, loc="best")

            ax2.plot(curve_df["Ci"], residuals, "ko")
            ax2.axhline(y=0.0, ls="--", color='black')
            ax2.set_xlabel('Ci')
            ax2.set_ylabel("Residuals (Obs$-$Fit)")
            ax2.set_xlim(0, 1500)
            ax2.set_ylim(10,-10)
        
            fig.savefig(ofname)
            plt.close(fig)
        

    

if __name__ == "__main__":

    results_dir = "../examples/results_pymc"
    data_dir = "../examples/data_pymc"
    plot_dir = "../examples/plots_pymc"
    trace_dir = "../examples/traces"
    
    # photosynthesis model
    from farquhar_model import FarquharC3
    model = FarquharC3(peaked_Jmax=True, peaked_Vcmax=True, model_Q10=False)
    call_model = model.calc_photosynthesis
    iterations = 50000
    burn = 25000
    thin = 5

    # Bayesian fitting class initialisation
    MCMC = FarquharMCMC(call_model=call_model, iterations=iterations, burn=burn, 
                        thin=thin, progress_bar=True)


    F = FitMe(results_dir, data_dir, plot_dir, trace_dir)
    mc = F.main(MCMC)
