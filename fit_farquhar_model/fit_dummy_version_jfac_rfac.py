#!/usr/bin/env python

"""
Using the Levenberg-Marquardt algorithm  to fit Jmax25, Vcmax25, Rd25, Eaj, Eav, 
Ear, deltaSj and deltaSv.

Jmax25, Vcmax25 & Rd25 are fit seperately by leaf, thus accounting for 
differences in leaf N. At the same time, Eaj, Eav, Ear, deltaSj and deltaSv are 
fit together for the same species. To achieve this we utilise dummy variables,
it will become more obvious below.

The steps here are:
    1. Define a search grid to pick the starting point of the minimiser, in an
       attempt to avoid issues relating to falling into a local minima. 
    2. Try and fit the parameters 
    
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
from lmfit import minimize, Parameters, conf_interval, report_fit, report_ci
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
from math import fabs

class FitMe(object):
    """
    Basic fitting class, contains some generic methods which are used by the
    fitting routines, e.g. plotting, file reporting etc.
    
    
    Error bar fitting issue - from lmfit documentation...

    In some cases, it may not be possible to estimate the 
    errors  and correlations. For example, if a variable 
    actually has no  practical effect on the fit, it will 
    likely cause the  covariance matrix to be singular, 
    making standard errors  impossible to estimate. Placing 
    bounds on varied Parameters  makes it more likely that 
    errors cannot be estimated, as  being near the maximum or 
    minimum value makes the covariance 
    matrix singular. In these cases, the errorbars attribute 
    of the fit result (Minimizer object) will be False. 
    """
    def __init__(self, model=None, ofname=None, results_dir=None, 
                 data_dir=None, plot_dir=None, num_iter=10, peaked=True, 
                 delimiter=","):
        """
        Parameters
        ----------
        model : object
            model that we are fitting measurements against...
        ofname : string
            output filename for writing fitting result.
        results_dir : string
            output directory path for the result to be written
        data_dir : string
            input directory path where measured A-Ci files live
        plot_dir : string
            directory to save plots of various fitting routines
        num_iter : int
            number of different attempts to refit code
        """
        
        self.results_dir = results_dir
        if ofname is not None:
            self.ofname = os.path.join(self.results_dir, ofname)
        self.data_dir = data_dir 
        self.plot_dir = plot_dir    
        self.farq = model.calc_photosynthesis
        self.succes_count = 0
        self.nfiles = 0
        self.deg2kelvin = 273.15
        self.num_iter = num_iter
        self.delimiter = delimiter
        
        self.peaked = peaked
        self.high_number = 99999.9
        
    def main(self, print_to_screen, infname_tag="*.csv"):   
        """ Loop over all our A-Ci measured curves and fit the Farquhar model
        parameters to these data. 
        
        Parameters
        ----------
        print_to_screen : logical
            print fitting result to screen? Default is no!
        """
        # open files and write header information
        ofile = self.open_output_files()
        writer = csv.writer(ofile, delimiter=',', quoting=csv.QUOTE_NONE, 
                            escapechar=' ')
                        
        for fname in glob.glob(os.path.join(self.data_dir, infname_tag)):
            df = self.read_data(fname)
            
            # sort data by curve first...
            df_sorted = pd.DataFrame()
            for curve_num in np.unique(df["Curve"]):
                curve_df = df[df["Curve"]==curve_num]
                curve_df = curve_df.sort(['Ci'], ascending=True)
                df_sorted = df_sorted.append(curve_df)
            df_sorted.index = range(len(df_sorted)) # need to reindex slice
            
            hdr_written = False
            for group in np.unique(df_sorted["fitgroup"]):
                dfr = df_sorted[df_sorted["fitgroup"]==group]
                dfr.index = range(len(dfr)) # need to reindex slice
                (params, dfr) = self.setup_model_params(dfr)
                # Test sensitivity, are we falling into local mins?
                lowest_rmse = self.high_number
                for i, iter in enumerate(xrange(self.num_iter)): 
                    print group, i
                    # pick new initial parameter guesses, but dont rebuild 
                    # params object
                    if i > 0:
                        params = self.change_param_values(dfr, params)
                
                    result = minimize(self.residual, params, args=(dfr,))
                    #report_fit(params, show_correl=False)
                    
                    #ci = conf_interval(result, trace=False)
                    #report_ci(ci)
                    
                    
                    (An, Anc, Anj) = self.forward_run(result, dfr)
                    # Successful fit?
                    # See comment above about why errorbars might not be 
                    # resolved.
                    #if result.errorbars and self.check_params(result):
                    if result.errorbars:
                        rmse = np.sqrt(np.mean((dfr["Photo"] - An)**2))
                        if rmse < lowest_rmse:
                            lowest_rmse = rmse
                            best_result = result
                        
                # Pick the best fit...
                if lowest_rmse < self.high_number and best_result.errorbars:
                    if print_to_screen:
                        self.print_fit_to_screen(best_result)
                        
                    hdr_written = self.report_fits(writer, best_result, 
                                                   os.path.basename(fname), 
                                                   dfr, An, hdr_written)
                    self.make_plots(dfr, An, Anc, Anj, best_result)
                else:
                    print "Fit failed, fitgroup = %d" % (group)          
        # tidy up
        ofile.close()
         
    def open_output_files(self):
        """
        Opens output file for recording fit information
        
        Returns: 
        --------
        fp : object
            file pointer
        """
        if os.path.isfile(self.ofname):
            os.remove(self.ofname)
        
        try:
            ofile = open(self.ofname, 'wb')
        except IOError:
            raise IOError("Can't open %s file for write" % self.ofname)     
        
        return ofile
        
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
    
    def setup_model_params(self, df):
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
        
        params = Parameters()
        # Need to loop over all the leaves, fitting separate Jmax25, Vcmax25 
        # and Rd25 parameter values by leaf
        
        for leaf_num in np.unique(df["Leaf"]):
             
            (Jmax25_guess, Vcmax25_guess, 
             Rd25_guess, Eaj_guess, 
             Eav_guess, Ear_guess, 
             delSj_guess, delSv_guess) = self.pick_random_starting_point()
            
            #params.add('Jmax25_%d' % (leaf_num), value=Jmax25_guess, min=0.0, 
            #            max=600.0)
            params.add('Vcmax25_%d' % (leaf_num), value=Vcmax25_guess , min=0.0, 
                        max=600.0)
            
            #params.add('Rdfac_%d' % (leaf_num), value=0.015, min=0.005, max=0.03) 
            #params.add('Rd25_%d' % (leaf_num), value=Rd25_guess, 
            #           expr='rdfac_%d * Vcmax25_%d' % (leaf_num, leaf_num))      
            
            #params.add('Rd25_%d' % (leaf_num), value=Rd25_guess, min=0.0, 
            #            max=6.0)      
            
            
            # Need to build dummy variable identifier for each leaf.
            col_id = "f_%d" % (leaf_num)
            temp = df["Leaf"]
            temp = np.where(temp==leaf_num, 1.0, 0.0)
            df[col_id] = temp
        
        params.add('Rdfac', value=0.015, min=0.005, max=0.05) 
        params.add('Jfac',value = 2.0, min = 0.5, max = 3.0)
        
        # Temp dependancy values do not vary by leaf, so only need one set of 
        # params.
        params.add('Eaj', value=Eaj_guess, min=20000.0, max=199999.9)
        params.add('delSj', value=delSj_guess, min=300.0, max=800.0)  
        #params.add('Hdj', value=200000.0, vary=False)
        params.add('Eav', value=Eav_guess, min=20000.0, max=199999.9)
        params.add('delSv', value=delSv_guess, min=300.0, max=800.0)  
        #params.add('Hdv', value=200000.0, vary=False)
        #params.add('Ear', value=Ear_guess, min=20000.0, max=199999.9)
            
        return params, df
    
    def change_param_values(self, df, params):
        """ pick new guesses for parameter values """
        for leaf_num in np.unique(df["Leaf"]):
            
            (Jfac_guess, Vcmax25_guess, 
             Rdfac_guess, Eaj_guess, 
             Eav_guess, 
             delSj_guess, delSv_guess) = self.pick_random_starting_point()
            
            #params['Jmax25_%d' % (leaf_num)].value = Jmax25_guess
            params['Vcmax25_%d' % (leaf_num)].value = Vcmax25_guess
            #params['Rd25_%d' % (leaf_num)].value = Rd25_guess
            #params['Rdfac_%d'% (leaf_num)].value = Rdfac_guess
        params['Rdfac'].value = Rdfac_guess
        params['Jfac'].value = Jfac_guess
        params['Eaj'].value = Eaj_guess
        params['Eav'].value = Eav_guess
        #params['Ear'].value = Ear_guess
        params['delSj'].value = delSj_guess
        params['delSv'].value = delSv_guess
        
        return params
        
    def pick_random_starting_point(self):
        """ random pick starting point for parameter values 
        
        Parameters
        ----------
        
        Returns: 
        --------
        retval * 3 : float
            Three starting guesses for Jmax, Vcmax and Rd
        """
        #Jmax25 = np.random.uniform(5.0, 550) 
        Jfac = np.random.uniform(0.5, 3.0)
        Vcmax25 = np.random.uniform(5.0, 350) 
        Rdfac = np.random.uniform(0.005, 0.03) 
        #Rd25 = np.random.uniform(0.0, 5.0)
        Eaj = np.random.uniform(20000.0, 80000.0)
        Eav = np.random.uniform(20000.0, 80000.0)
        #Ear = np.random.uniform(20000.0, 80000.0)
        delSj = np.random.uniform(550.0, 700.0)
        delSv = np.random.uniform(550.0, 700.0)
        if not self.peaked:
            delSj = None
            delSv = None
        
        return Jfac, Vcmax25, Rdfac, Eaj, Eav, delSj, delSv
    
                  
    def residual(self, params, df):
        """ simple function to quantify how good the fit was for the fitting
        routine. 
        
        Parameters
        ----------
        params : object
            List of parameters to be fit, initial guess, ranges etc. This is
            an lmfit object
        df: dataframe
            df containing all the A-Ci curve and temp data 
        
        Returns: 
        --------
        residual : array
            residual of fit between model and obs, based on current parameter
            set
        """
        # Need to employ dummy variables to fit model parameters
        # e.g. Jmax = Jmax_leaf1 * f1 + Jmax_leaf2 * f2 etc
        # where f1=1 for matching leaf data and 0 elsewhere, ditto f2.
         
        # These parameter values need to be arrays
        Jmax25 = np.zeros(len(df))
        Vcmax25 = np.zeros(len(df))
        Rd25 = np.zeros(len(df))
        
        # Need to build dummy variables.
        for i in np.unique(df["Leaf"]):
            col_id = "f_%d" % (i)
            
            
            Vcmax25 += params['Vcmax25_%d' % (i)].value * df[col_id]
            #Rd25 += params['Rd25_%d' % (i)].value * df[col_id]
            #Rd25 += (params['Rdfac_%d' % (i)].value * 
            #         params['Vcmax25_%d' % (i)].value * df[col_id])
            Rd25 += (params['Rdfac'].value * 
                     params['Vcmax25_%d' % (i)].value * df[col_id])
            #Jmax25 += params['Jmax25_%d' % (i)].value * df[col_id]
            Jmax25 += (params['Jfac'].value * 
                       params['Vcmax25_%d' % (i)].value * df[col_id])
            
            
        Eaj = params['Eaj'].value
        delSj = params['delSj'].value
        #Hdj = params['Hdj'].value
        Eav = params['Eav'].value
        delSv = params['delSv'].value
        #Hdv = params['Hdv'].value
        #Ear = params['Ear'].value
        Ear = 20000.0
        Hdv = 200000.00000000
        Hdj = 200000.00000000
        if hasattr(df, "Par"):
            (An, Anc, Anj) = self.farq(Ci=df["Ci"], Tleaf=df["Tleaf"], 
                                       Par=df["Par"], Jmax=None, Vcmax=None, 
                                       Jmax25=Jmax25, Vcmax25=Vcmax25, Rd=None, 
                                       Q10=None, Eaj=Eaj, Eav=Eav, 
                                       deltaSj=delSj, deltaSv=delSv, Rd25=Rd25, 
                                       Ear=Ear, Hdv=Hdv, Hdj=Hdj)
        else:
            (An, Anc, Anj) = self.farq(Ci=df["Ci"], Tleaf=df["Tleaf"], 
                                       Par=None, Jmax=None, Vcmax=None, 
                                       Jmax25=Jmax25, Vcmax25=Vcmax25, Rd=None, 
                                       Q10=None, Eaj=Eaj, Eav=Eav, 
                                       deltaSj=delSj, deltaSv=delSv, Rd25=Rd25, 
                                       Ear=Ear, Hdv=Hdv, Hdj=Hdj)
        
        return (df["Photo"] - An)
    
    def forward_run(self, result, df):
        """ Run farquhar model with fitted parameters and return result 
        
        Parameters
        ----------
        result : object
            fitting result, param, std. error etc.
         df: dataframe
            df containing all the A-Ci curve and temp data 
        
        Returns
        --------
        An : float
            Net leaf assimilation rate [umol m-2 s-1]
        Acn : float
            Net rubisco-limited leaf assimilation rate [umol m-2 s-1]
        Ajn : float
            Net RuBP-regeneration-limited leaf assimilation rate [umol m-2 s-1]
        """
        Jmax25 = np.zeros(len(df))
        Vcmax25 = np.zeros(len(df))
        Rd25 = np.zeros(len(df))

        for i in np.unique(df["Leaf"]):
            col_id = "f_%d" % (i)
            
            Vcmax25 += result.params['Vcmax25_%d' % (i)].value * df[col_id]
            #Rd25 += result.params['Rd25_%d' % (i)].value * df[col_id]
            Rd25 += (result.params['Rdfac'].value * 
                     result.params['Vcmax25_%d' % (i)].value * df[col_id])
            Jmax25 += (result.params['Jfac'].value * 
                       result.params['Vcmax25_%d' % (i)].value * df[col_id])
            Jmax25 += result.params['Jmax25_%d' % (i)].value * df[col_id]
                     
        Eaj = result.params['Eaj'].value
        delSj = result.params['delSj'].value
        #Hdj = result.params['Hdj'].value
        Eav = result.params['Eav'].value
        delSv = result.params['delSv'].value
        #Hdv = result.params['Hdv'].value
        #Ear = result.params['Ear'].value
        Ear = 20000.0
        Hdv = 200000.00000000
        Hdj = 200000.00000000
        if hasattr(df, "Par"):
            (An, Anc, Anj) = self.farq(Ci=df["Ci"], Tleaf=df["Tleaf"], 
                                       Par=df["Par"], Jmax=None, Vcmax=None, 
                                       Jmax25=Jmax25, Vcmax25=Vcmax25, Rd=None, 
                                       Q10=None, Eaj=Eaj, Eav=Eav, 
                                       deltaSj=delSj, deltaSv=delSv, Rd25=Rd25, 
                                       Ear=Ear, Hdv=Hdv, Hdj=Hdj)
        else:
            (An, Anc, Anj) = self.farq(Ci=df["Ci"], Tleaf=df["Tleaf"], 
                                       Par=None, Jmax=None, Vcmax=None, 
                                       Jmax25=Jmax25, Vcmax25=Vcmax25, Rd=None, 
                                       Q10=None, Eaj=Eaj, Eav=Eav, 
                                       deltaSj=delSj, deltaSv=delSv, Rd25=Rd25, 
                                       Ear=Ear, Hdv=Hdv, Hdj=Hdj)
        
        return (An, Anc, Anj)
    
    def report_fits(self, writer, result, fname, df, An_fit, 
                    hdr_written=False):
        """ Save fitting results to a file... 
        
        Parameters
        ----------
        result: object
            fitting result, param, std. error etc.
        fname : string
            filename to append to output file
        df : object
            dataframe containing all the A-Ci curve information
        An_fit : array
            best model fit using optimised parameters, Net leaf assimilation 
            rate [umol m-2 s-1]
        hdr_written : logical
            Flag to stop the header being rewritten when in a loop    
        
        """
        
        
        remaining_header = ["Tav", "Var", "R2", "SSQ", "MSE", "DOF", "n", \
                            "Species", "Season", "Leaf", "Filename", \
                            "Topt_J", "Topt_V", "id"]
        
        pearsons_r = stats.pearsonr(df["Photo"], An_fit)[0]
        diff_sq = (df["Photo"]-An_fit)**2
        ssq = np.sum(diff_sq)
        mean_sq_err = np.mean(diff_sq)
        row = []
        header = []
        for name, par in result.params.items():
            header.append("%s" % (name))
            header.append("%s" % ("SE"))
            row.append("%s" % (par.value))
            row.append("%s" % (par.stderr))
        row.append("%s" % (np.mean(df["Tleaf"] - self.deg2kelvin)))
        row.append("%s" % ((df["Photo"]-An_fit).var()))
        row.append("%s" % (pearsons_r**2))
        row.append("%s" % (ssq))
        row.append("%s" % (mean_sq_err))
        row.append("%s" % (len(An_fit)-1))
        row.append("%s" % (len(An_fit)))
        row.append("%s" % (df["Species"][0]))
        row.append("%s" % (df["Season"][0]))
        row.append("%s" % (df["Leaf"][0]))
        row.append("%s" % (fname))
        
        Hdv = 200000.00000000
        Hdj = 200000.00000000
        Topt_J = (self.calc_Topt(Hdj, 
                                 result.params["Eaj"].value, 
                                 result.params["delSj"].value))
        Topt_V = (self.calc_Topt(Hdv, 
                                 result.params["Eav"].value, 
                                 result.params["delSv"].value))
        #Topt_J = (self.calc_Topt(result.params["Hdj"].value, 
        #                         result.params["Eaj"].value, 
        #                         result.params["delSj"].value))
        #Topt_V = (self.calc_Topt(result.params["Hdv"].value, 
        #                         result.params["Eav"].value, 
        #                         result.params["delSv"].value))
        row.append("%f" % (Topt_J))
        row.append("%f" % (Topt_V))
        row.append("%s%s%s" % (str(df["Species"][0]), \
                               str(df["Season"][0]), \
                               str(df["Leaf"][0])))
        
        header = header + remaining_header
        if not hdr_written:
            writer.writerow(header)
            hdr_written = True
        writer.writerow(row)
        
        return hdr_written
        
                
    def print_fit_to_screen(self, result):
        """ Print the fitting result to the terminal 
        
        Parameters
        ----------
        result : object
            fitting result, param, std. error etc.
        """
        for name, par in result.params.items():
            print '%s = %.8f +/- %.8f ' % (name, par.value, par.stderr)
        print 
    
    def check_params(self, result, threshold=1.05):
        """ Check that fitted values aren't stuck against the "wall"
        
        Parameters
        ----------
        result : object
            fitting result, param, std. error etc.
        """
        bad = False
        for name, par in result.params.items():
            if name in ('Hdj', 'Hdv'):
                continue
            elif fabs(par.stderr - 0.00000000) < 0.00001:
                bad = True
                break
                
            #if (par.value * threshold > par.max or 
            #    fabs(par.stderr - 0.00000000) < 0.00001):
            #    bad = True
            #    break
            
        return bad

    
    
    def make_plots(self, df, An_fit, Anc_fit, Anj_fit, result):
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
        
        for curve_num in np.unique(df["Curve"]):
            curve_df = df[df["Curve"]==curve_num]
            #curve_df = curve_df.sort(['Ci'], ascending=True)#Sort for nice plots
            i = curve_df["Leaf"].values[0]
            
            col_id = "f_%d" % (i)
            
            #Jmax25 = result.params['Jmax25_%d' % (i)].value
            Vcmax25 = result.params['Vcmax25_%d' % (i)].value
            Jmax25 = result.params['Jfac'].value *  Vcmax25  
            Rd25 = result.params['Rdfac'].value *  Vcmax25   
            #Rd25 = result.params['Rdfac_%d' % (i)].value *  Vcmax25      
            #Rd25 = result.params['Rd25_%d' % (i)].value        
            Eaj = result.params['Eaj'].value
            delSj = result.params['delSj'].value
            #Hdj = result.params['Hdj'].value
            Eav = result.params['Eav'].value
            delSv = result.params['delSv'].value
            #Hdv = result.params['Hdv'].value
            #Ear = result.params['Ear'].value
            Ear = 20000.0
            Hdv = 200000.00000000
            Hdj = 200000.00000000
            if hasattr(curve_df, "Par"):
                (An, Anc, Anj) = self.farq(Ci=curve_df["Ci"], Tleaf=curve_df["Tleaf"], 
                                           Par=curve_df["Par"], Jmax=None, Vcmax=None, 
                                           Jmax25=Jmax25, Vcmax25=Vcmax25, Rd=None, 
                                           Q10=None, Eaj=Eaj, Eav=Eav, 
                                           deltaSj=delSj, deltaSv=delSv, Rd25=Rd25, 
                                           Ear=Ear, Hdv=Hdv, Hdj=Hdj)
            else:
                (An, Anc, Anj) = self.farq(Ci=curve_df["Ci"], Tleaf=curve_df["Tleaf"], 
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
            ax.set_ylabel("Net Assimilation Rate")
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
    
    def calc_Topt(self, Hd, Ha, delS, RGAS=8.314):
        """ Calculate the temperature optimum 
        
        Parameters
        ----------
        Hd : float
            describes rate of decrease about the optimum temp [KJ mol-1]
        Ha : float
            activation energy for the parameter [kJ mol-1]
        delS : float
            entropy factor [J mol-1 K-1)
        RGAS : float
            Universal gas constant [J mol-1 K-1]
        
        Returns
        --------
        Topt : float
            optimum temperature [deg C]
        
        Reference
        ----------
        * Medlyn, B. E., Dreyer, E., Ellsworth, D., Forstreuter, M., Harley, 
          P.C., Kirschbaum, M.U.F., Leroux, X., Montpied, P., Strassemeyer, J., 
          Walcroft, A., Wang, K. and Loustau, D. (2002) Temperature response of 
          parameters of a biochemically based model of photosynthesis. II. 
          A review of experimental data. Plant, Cell and Enviroment 25, 
          1167-1179.
        """
        return (Hd / (delS - RGAS * np.log(Ha / (Hd - Ha)))) - self.deg2kelvin
            

if __name__ == "__main__":

    ofname = "fitting_results.csv"
    results_dir = "/Users/mdekauwe/Desktop/results"
    data_dir = "/Users/mdekauwe/Desktop/data"
    plot_dir = "/Users/mdekauwe/Desktop/plots"
    
    #results_dir = "results"
    #data_dir = "data"
    #plot_dir = "plots"
    from farquhar_model import FarquharC3
    model = FarquharC3(peaked_Jmax=True, peaked_Vcmax=True, model_Q10=False)
    
    F = FitMe(model, ofname, results_dir, data_dir, plot_dir)
    F.main(print_to_screen=True) 