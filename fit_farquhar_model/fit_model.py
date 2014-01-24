#!/usr/bin/env python

"""
Using the Levenberg-Marquardt algorithm  to fit Jmax, Vcmax, Rd, Eaj, Eav,
deltaSj and deltaSv.

The steps here are:
    1. Define a search grid to pick the starting point of the minimiser, in an
       attempt to avoid issues relating to falling into a local minima. 
    2. Try and fit the parameters 
    
That's all folks.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (13.08.2012)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import numpy as np
import csv
from lmfit import minimize, Parameters, printfuncs, conf_interval, conf_interval2d
from scipy import stats
import matplotlib.pyplot as plt


class FitMe(object):
    """
    Basic fitting class, contains some generic methods which are used by the
    fitting routines, e.g. plotting, file reporting etc. This is intended
    to be subclased.
    """
    def __init__(self, model=None, ofname=None, results_dir=None, 
                 data_dir=None, plot_dir=None, Niter=500):
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
        Niter : int
            number of different attempts to refit code
        """
        
        self.results_dir = results_dir
        if ofname is not None:
            self.ofname = os.path.join(self.results_dir, ofname)
        self.data_dir = data_dir 
        self.plot_dir = plot_dir    
        self.call_model = model.calc_photosynthesis
        self.succes_count = 0
        self.nfiles = 0
        self.deg2kelvin = 273.15
        self.Niter = Niter
        
    def open_output_files(self, ofname):
        """
        Opens output file for recording fit information
        
        Parameters
        ----------
        ofname : string
            output file name
        
        Returns: 
        --------
        fp : object
            file pointer
        """
        if os.path.isfile(ofname):
            os.remove(ofname)
            
        try:
            fp = open(ofname, 'wb')
        except IOError:
            raise IOError("Can't open %s file for write" % ofname)     
        
        return fp
        
    def read_data(self, fname, infile_type="aci", delimiter=","):
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
        data = np.recfromcsv(fname, delimiter=delimiter, names=True, 
                                 case_sensitive=True)
        if infile_type == "norm":
            # using normalised temp data
            data["Tav"] = data["Tav"] + self.deg2kelvin
            data["Jnorm"] = np.exp(data["Jnorm"])
            data["Vnorm"] = np.exp(data["Vnorm"])
        elif infile_type == "meas":
            # using measured temp data.
            data["Tav"] = data["Tav"] + self.deg2kelvin
            data["Jmax"] = data["Jmax"]
            data["Vcmax"] = data["Vcmax"]  
        elif infile_type != "aci":
            raise IOError("Unknown file type in read??\n")   
              
        return data
                      
    def residual(self, params, data, obs):
        """ simple function to quantify how good the fit was for the fitting
        routine. Could use something better? RMSE?
        
        Parameters
        ----------
        params : object
            List of parameters to be fit, initial guess, ranges etc. This is
            an lmfit object
        data: array
            data to run farquhar model with
        obs : array
            A-Ci data to fit model against
        
        Returns: 
        --------
        residual : array
            residual of fit between model and obs, based on current parameter
            set
        """
        Jmax = params['Jmax'].value
        Vcmax = params['Vcmax'].value
        Rd = params['Rd'].value  
        
        if hasattr(data, "Par"):
             (An, Anc, Anj) = self.call_model(Ci=data["Ci"], Tleaf=data["Tleaf"],
                                              Par=data["Par"], Jmax=Jmax, 
                                              Vcmax=Vcmax, Rd=Rd)
        else:
            
            (An, Anc, Anj) = self.call_model(Ci=data["Ci"], Tleaf=data["Tleaf"], 
                                             Jmax=Jmax, Vcmax=Vcmax, Rd=Rd)
        return (obs - An)
    
    def report_fits(self, f, result, fname, data, An_fit):
        """ Save fitting results to a file... 
        
        Parameters
        ----------
        f : object
            file pointer
        result: object
            fitting result, param, std. error etc.
        obs : array
            A-Ci data to fit model against
        fname : string
            filename to append to output file
        data : object
            input A-Ci curve information
        An_fit : array
            best model fit using optimised parameters, Net leaf assimilation 
            rate [umol m-2 s-1]
        """
        pearsons_r = stats.pearsonr(data["Photo"], An_fit)[0]
        diff_sq = (data["Photo"]-An_fit)**2
        ssq = np.sum(diff_sq)
        mean_sq_err = np.mean(diff_sq)
        row = []
        for name, par in result.params.items():
            row.append("%s" % (par.value))
            row.append("%s" % (par.stderr))
        row.append("%s" % (np.mean(data["Tleaf"] - self.deg2kelvin)))
        row.append("%s" % ((data["Photo"]-An_fit).var()))
        row.append("%s" % (pearsons_r**2))
        row.append("%s" % (ssq))
        row.append("%s" % (mean_sq_err))
        row.append("%s" % (len(An_fit)-1))
        row.append("%s" % (len(An_fit)))
        row.append("%s" % (data["Species"][0]))
        row.append("%s" % (data["Season"][0]))
        row.append("%s" % (data["Leaf"][0]))
        row.append("%s" % (data["Curve"][0]))
        row.append("%s" % (fname))
        row.append("%s%s%s" % (str(data["Species"][0]), \
                               str(data["Season"][0]), \
                               str(data["Leaf"][0])))
        f.writerow(row)
        
        """
        fname2 = os.path.join(self.results_dir, 
                                "fitted_conf_int_j_v_rd.txt")
        f2 = open(fname2, "w")
        try:
            ci = conf_interval(result, sigmas=[0.95])
            
            print >>f2, "\t\t95%\t\t0.00%\t\t95%"
            for k, v in ci.iteritems():
                print >>f2,"%s\t\t%f\t%f\t%f" % (k, round(ci[k][0][1], 3), \
                                                 round(ci[k][1][1], 3), \
                                                 round(ci[k][2][1], 3))
        
        except ValueError:
            print >>f2, "Oops!  Some problem fitting confidence intervals..."    
        f2.close()
        """
    def forward_run(self, result, data):
        """ Run farquhar model with fitted parameters and return result 
        
        Parameters
        ----------
        result : object
            fitting result, param, std. error etc.
        data : object
            input A-Ci curve information
        
        Returns
        --------
        An : float
            Net leaf assimilation rate [umol m-2 s-1]
        Acn : float
            Net rubisco-limited leaf assimilation rate [umol m-2 s-1]
        Ajn : float
            Net RuBP-regeneration-limited leaf assimilation rate [umol m-2 s-1]
        """
        Jmax = result.params['Jmax'].value
        Vcmax = result.params['Vcmax'].value
        Rd = result.params['Rd'].value  
        if hasattr(data, "Par"):
             (An, Anc, Anj) = self.call_model(Ci=data["Ci"], Tleaf=data["Tleaf"],
                                              Par=data["Par"], Jmax=Jmax, 
                                              Vcmax=Vcmax, Rd=Rd)
        else:
            (An, Anc, Anj) = self.call_model(Ci=data["Ci"], Tleaf=data["Tleaf"], 
                                             Jmax=Jmax, Vcmax=Vcmax, Rd=Rd)
            
        return (An, Anc, Anj)
                
    def setup_model_params(self, jmax_guess=None, vcmax_guess=None, 
                           rd_guess=None, hd_guess=None, ea_guess=None, 
                           dels_guess=None):
        """ Setup lmfit Parameters object
        
        Parameters
        ----------
        jmax_guess : value
            initial parameter guess, if nothing is passed, i.e. it is None,
            then parameter is not fitted
        vcmax_guess : value
            initial parameter guess, if nothing is passed, i.e. it is None,
            then parameter is not fitted
        rd_guess : value
            initial parameter guess, if nothing is passed, i.e. it is None,
            then parameter is not fitted
        hd_guess : value
            initial parameter guess, if nothing is passed, i.e. it is None,
            then parameter is not fitted
        ea_guess : value
            initial parameter guess, if nothing is passed, i.e. it is None,
            then parameter is not fitted
        dels_guess : value
            initial parameter guess, if nothing is passed, i.e. it is None,
            then parameter is not fitted
        
        Returns
        -------
        params : object
            lmfit object containing parameters to fit
        """
        params = Parameters()
        if jmax_guess is not None:
            params.add('Jmax', value=jmax_guess, min=0.0)
        if vcmax_guess is not None:
            params.add('Vcmax', value=vcmax_guess, min=0.0)
        if rd_guess is not None:
            params.add('Rd', value=rd_guess, min=0.0)
        
        if ea_guess is not None:
            params.add('Ea', value=ea_guess, min=0.0)
        if hd_guess is not None:
            params.add('Hd', value=hd_guess, vary=False)
        if dels_guess is not None:
            params.add('delS', value=dels_guess, min=0.0, max=700.0)
        
        return params
    
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
    
    def make_plot(self, data, curve_num, An_fit, Anc_fit, Anj_fit, result):
        """ Make some plots to show how good our fitted model is to the data 
        
        * Plots A-Ci model fits vs. data
        * Residuals between fit and measured A
        
        Parameters
        ----------
        data : object
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
        species = data["Species"][0]
        season = data["Season"][0]
        season = "all"
        leaf = data["Leaf"][0]
        ofname = "%s/%s_%s_%s_%s_fit_and_residual.png" % \
                 (self.plot_dir, species, season, leaf, curve_num)
        residuals = data["Photo"] - An_fit  
        
        colour_list=['red', 'blue', 'green', 'yellow', 'orange', 'blueviolet',\
                     'darkmagenta', 'cyan', 'indigo', 'palegreen', 'salmon',\
                     'pink', 'darkgreen', 'darkblue']
        
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
        
        ax.plot(data["Ci"], data["Photo"], 
                ls="", lw=1.5, marker="o", c="black")
        ax.plot(data["Ci"], An_fit, '-', c="black", linewidth=3, 
                label="An-Rd")
        ax.plot(data["Ci"], Anc_fit, '-', c="red", linewidth=1, 
                label="Ac-Rd")
        ax.plot(data["Ci"], Anj_fit, '-', c="blue", linewidth=1, 
                label="Aj-Rd")
        ax.set_ylabel("A$_n$", weight="bold")
        ax.axes.get_xaxis().set_visible(False)
        ax.set_xlim(0, 1600)
        ax.legend(numpoints=1, loc="best")

        ax2.plot(data["Ci"], residuals, "ko")
        ax2.axhline(y=0.0, ls="--", color='black')
        ax2.set_xlabel('Ci', weight="bold")
        ax2.set_ylabel("Residuals (Obs$-$Fit)", weight="bold")
        ax2.set_xlim(0, 1500)
        ax2.set_ylim(10,-10)
        
        fig.savefig(ofname)
        plt.close(fig)
        
        """
        # Plots confidence regions for two fixed parameters.
        ofname = "%s/%s_%s_%s_%s_jmax_vcmax_conf_surface.png" % \
                 (self.plot_dir, species, season, leaf, curve_num)
        
        fig = plt.figure() 
        ax = fig.add_subplot(111)
        x, y, grid = conf_interval2d(result, 'Jmax', 'Vcmax', 30, 30)
        plt.contourf(x, y, grid, np.linspace(0,1,11))
        plt.xlabel('Jmax')
        plt.ylabel('Vcmax')
        plt.colorbar()
        fig.savefig(ofname)
        plt.clf()
        """
        
    def write_file_hdr(self, fname, header):  
        """ Write CSV file header 
        
        Parameters
        ----------
        fname : string
            output filename
        header : list/array
            List/array containing information for header for output CSV file
        
        Returns
        -------
        wr : object
            output file pointer
        """
        
        wr = csv.writer(fname, delimiter=',', quoting=csv.QUOTE_NONE, 
                        escapechar=' ')
        wr.writerow(header)
        
        return wr
        
    def tidy_up(self, fp=None):
        """ Clean up at the end of fitting 
        
        Parameters
        ----------
        fp : object
            file pointer
        """
        total_fits = float(self.succes_count) / self.nfiles * 100.0
        print "\nOverall fitted %.1f%% of the data\n" % (total_fits)
        fp.close()
    
    def pick_random_starting_point(self):
        """ random pick starting point for parameter values 
        
        Parameters
        ----------
        data : array
            model driving data
        grid_size : int
            hardwired, number of samples
        
        Returns: 
        --------
        retval * 3 : float
            Three starting guesses for Jmax, Vcmax and Rd
        """
        # Shuffle arrays so that our combination of parameters is random
        Vcmax = np.random.uniform(5.0, 350) 
        Jmax = np.random.uniform(5.0, 550) 
        Rd = np.random.uniform(0.0, 6.0)
        
        return Vcmax, Jmax, Rd
        
    def pick_starting_point(self, data, grid_size=100):
        """ Figure out a good starting parameter guess
        
        High-density grid search to overcome issues with ending up in a 
        local minima. Values that yield the lowest RMSE (without minimisation)
        are used as the starting point for the minimisation. 
        
        Parameters
        ----------
        data : array
            model driving data
        grid_size : int
            hardwired, number of samples
        
        Returns: 
        --------
        retval * 3 : float
            Three starting guesses for Jmax, Vcmax and Rd
        
        Reference:
        ----------
        Dubois et al (2007) Optimizing the statistical estimation of the 
        parameters of the Farquhar-von Caemmerer-Berry model of photosynthesis. 
        New Phytologist, 176, 402--414
        
        """
        
        # Shuffle arrays so that our combination of parameters is random
        Vcmax = np.linspace(5.0, 350, grid_size) 
        Jmax = np.linspace(5.0, 550, grid_size) 
        Rd = np.linspace(0.0, 6.0, grid_size)
        
        """
        grid_size = 50
        for v in np.linspace(5.0, 350, grid_size):
            for j in np.linspace(5.0, 550, grid_size):
                for r in np.linspace(1E-8, 10.5, grid_size):
        
                    (An, Anc, Anj) = self.call_model(data["Ci"], 
                                                data["Tleaf"], 
                                                Jmax=j, Vcmax=v, Rd=r)
                    rmse = np.sqrt(np.mean((data["Photo"]- An)**2))
                    for k in xrange(len(An)):
                        
    
                        print v, j, r, An[k], data["Ci"][k], rmse
        """
       
        p1, p2, p3 = np.ix_(Jmax, Vcmax, Rd)
        if hasattr(data, "Par"):
            (An, Anc, Anj) = self.call_model(Ci=data["Ci"][:,None,None,None], 
                                             Tleaf=data["Tleaf"][:,None,None,None], 
                                             Par=data["Par"][:,None,None,None],
                                             Jmax=p1, Vcmax=p2, Rd=p3)
        else:
            (An, Anc, Anj) = self.call_model(Ci=data["Ci"][:,None,None,None], 
                                             Tleaf=data["Tleaf"][:,None,None,None], 
                                             Jmax=p1, Vcmax=p2, Rd=p3)
        
        rmse = np.sqrt(((data["Photo"][:,None,None,None]- An)**2).mean(0))
        ndx = np.where(rmse.min()== rmse)
        (jidx, vidx, rdidx) = ndx
        
        return Vcmax[vidx].squeeze(), Jmax[jidx].squeeze(), Rd[rdidx].squeeze()
    
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
            
class FitJmaxVcmaxRd(FitMe):
    """ Fit the model parameters Jmax, Vcmax and Rd to the measured A-Ci data
    
    This is a subclass of the FitMe class above, it uses most of the same 
    methods
    """
    def __init__(self, model=None, ofname=None, results_dir=None, 
                 data_dir=None, plot_dir=None):
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
        """        
        FitMe.__init__(self, model, ofname, results_dir, data_dir, plot_dir)
        self.header = ["Jmax", "JSE", "Vcmax", "VSE", "Rd", "RSE", "Tav", \
                       "Var", "R2", "SSQ", "MSE", "DOF", "n", "Species", "Season", \
                       "Leaf", "Curve", "Filename", "id"]
           
    def main(self, print_to_screen, infname_tag="*.csv"):   
        """ Loop over all our A-Ci measured curves and fit the Farquhar model
        parameters to this data 
        
        Parameters
        ----------
        print_to_screen : logical
            print fitting result to screen? Default is no!
        """
        # open files and write header information
        fp = self.open_output_files(self.ofname)
        wr = self.write_file_hdr(fp, self.header)
    
        # Loop over all the measured data and fit the model params.
        for fname in glob.glob(os.path.join(self.data_dir, infname_tag)):
            data = self.read_data(fname, infile_type="aci")
            data["Tleaf"] += self.deg2kelvin
            for curve_num in np.unique(data["Curve"]):
                curve_data = data[np.where(data["Curve"]==curve_num)]
                (vcmax_guess, jmax_guess, 
                    rd_guess) = self.pick_starting_point(curve_data)
                params = self.setup_model_params(jmax_guess=jmax_guess, 
                                                 vcmax_guess=vcmax_guess, 
                                                 rd_guess=rd_guess)
                
                result = minimize(self.residual, params,  
                                  args=(curve_data, curve_data["Photo"]))
                    
                # Did we resolve the error bars during the fit? 
                #
                # From lmfit...
                #
                # In some cases, it may not be possible to estimate the errors 
                # and correlations. For example, if a variable actually has no 
                # practical effect on the fit, it will likely cause the 
                # covariance matrix to be singular, making standard errors 
                # impossible to estimate. Placing bounds on varied Parameters 
                # makes it more likely that errors cannot be estimated, as 
                # being near the maximum or minimum value makes the covariance 
                # matrix singular. In these cases, the errorbars attribute of 
                # the fit result (Minimizer object) will be False.
                if (result.errorbars and 
                    np.isnan(result.params['Jmax'].stderr) == False):
                    
                    self.succes_count += 1
                else:
                    
                    # Failed errobar fitting, going to try and mess with 
                    # starting poisition...
                    for i in xrange(self.Niter):
                        (vcmax_guess, jmax_guess, 
                         rd_guess) = self.pick_random_starting_point()
                    
                        params = self.setup_model_params(jmax_guess=jmax_guess, 
                                                     vcmax_guess=vcmax_guess, 
                                                     rd_guess=rd_guess)
                        result = minimize(self.residual, params,  
                                         args=(curve_data, curve_data["Photo"]))
                        if result.errorbars:
                            self.succes_count += 1
                            break
                
                if print_to_screen:
                    print fname, curve_num
                    self.print_fit_to_screen(result)
                    
                # Need to run the Farquhar model with the fitted params for
                # plotting...
                (An, Anc, Anj) = self.forward_run(result, curve_data)
                self.report_fits(wr, result, os.path.basename(fname), 
                                 curve_data, An)
                
                self.make_plot(curve_data, curve_num, An, Anc, Anj, result)
                self.nfiles += 1       
        self.tidy_up(fp)    
    

class FitEaDels(FitMe):
    """ Fit the model parameters Eaj, Eav, Dels to the measured A-Ci data"""
    def __init__(self, model=None, infname=None, ofname=None, results_dir=None, 
                 data_dir=None, peaked=True):
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
        """
        
        self.infname = infname
        FitMe.__init__(self, model=model, ofname=ofname, 
                       results_dir=results_dir, data_dir=data_dir)
       
        self.peaked = peaked
        if self.peaked:
            self.call_model = model.peaked_arrh
            self.header = ["Param", "Ea", "SE", "Hd", "SE", "delS", "delSSE", \
                            "Var", "R2", "SSQ", "MSE", "DOF", "n", "Topt",\
                            "id"]
        else:
            self.call_model = model.arrh
            self.header = ["Param", "Ea", "SE", "Var", "R2", "SSQ", "MSE", "DOF", \
                            "n", "Topt", "id"]
        
    def main(self, print_to_screen):   
        """ Loop over all our A-Ci measured curves and fit the Farquhar model
        parameters to this data 
        
        Parameters
        ----------
        print_to_screen : logical
            print fitting result to screen? Default is no!
        """
        all_data = self.read_data(self.infname, infile_type="norm")
        fp = self.open_output_files(self.ofname)
        wr = self.write_file_hdr(fp, self.header)
        
        # Loop over all the measured data and fit the model params.
        for id in np.unique(all_data["fitgroup"]):
            data = all_data[np.where(all_data["fitgroup"] == id)]
            
            # Fit Jmax vs T first
            if self.peaked:
                (ea_guess, 
                dels_guess) = self.pick_starting_point(data, data["Jnorm"])  
                params = self.setup_model_params(peaked=self.peaked, 
                                                 hd_guess=200000.0, 
                                                 ea_guess=ea_guess, 
                                                 dels_guess=dels_guess)
            else:
                params = Parameters()
                ea_guess = np.random.uniform(20000.0, 80000.0)
                params.add('Ea', value=ea_guess, min=0.0)
            
            result = minimize(self.residual, params, method="leastsq", 
                              args=(data, data["Jnorm"]))
            
            # Did we resolve the error bars during the fit? 
            #
            # From lmfit...
            #
            # In some cases, it may not be possible to estimate the errors 
            # and correlations. For example, if a variable actually has no 
            # practical effect on the fit, it will likely cause the 
            # covariance matrix to be singular, making standard errors 
            # impossible to estimate. Placing bounds on varied Parameters 
            # makes it more likely that errors cannot be estimated, as 
            # being near the maximum or minimum value makes the covariance 
            # matrix singular. In these cases, the errorbars attribute of 
            # the fit result (Minimizer object) will be False.
            if (result.errorbars and 
                np.isnan(result.params['Ea'].stderr) == False):
                
                self.succes_count += 1
            else:
                
                # Failed errobar fitting, going to try and mess with 
                # starting poisition...
                for i in xrange(self.Niter):
                    # Fit Jmax vs T first
                    if self.peaked:
                        (ea_guess, 
                         dels_guess) = self.pick_random_starting_point()
                        params = self.setup_model_params(peaked=self.peaked, 
                                                 hd_guess=200000.0, 
                                                 ea_guess=ea_guess, 
                                                 dels_guess=dels_guess)
                    else:
                        params = Parameters()
                        ea_guess = np.random.uniform(20000.0, 80000.0)
                        params.add('Ea', value=ea_guess, min=0.0)
                    
                    result = minimize(self.residual, params, method="leastsq", 
                              args=(data, data["Jnorm"]))
                    
                    if result.errorbars:
                        self.succes_count += 1
                        break
            
            if print_to_screen:
                self.print_fit_to_screen(result)
        
            (peak_fit) = self.forward_run(result, data)
            if self.peaked:
                Topt = (self.calc_Topt(result.params["Hd"].value, 
                                       result.params["Ea"].value, 
                                       result.params["delS"].value))
            else:
                Topt = -9999.9 # not calculated
            self.report_fits(wr, result, data, data["Jnorm"], peak_fit, 
                             "Jmax", Topt, id)
           
            # Fit Vcmax vs T next 
            if self.peaked:
                (ea_guess, 
                dels_guess) = self.pick_starting_point(data, data["Vnorm"])
                params = self.setup_model_params(peaked=self.peaked, 
                                                 hd_guess=200000.0, 
                                                 ea_guess=ea_guess, 
                                                 dels_guess=dels_guess)
            else:
                params = Parameters()
                if ea_guess is not None:
                    params = Parameters()
                    ea_guess = np.random.uniform(20000.0, 80000.0)
                    params.add('Ea', value=ea_guess, min=0.0)
                
            result = minimize(self.residual, params, method="leastsq", 
                              args=(data, data["Vnorm"]))
            
            # Did we resolve the error bars during the fit? 
            #
            # From lmfit...
            #
            # In some cases, it may not be possible to estimate the errors 
            # and correlations. For example, if a variable actually has no 
            # practical effect on the fit, it will likely cause the 
            # covariance matrix to be singular, making standard errors 
            # impossible to estimate. Placing bounds on varied Parameters 
            # makes it more likely that errors cannot be estimated, as 
            # being near the maximum or minimum value makes the covariance 
            # matrix singular. In these cases, the errorbars attribute of 
            # the fit result (Minimizer object) will be False.
            if (result.errorbars and 
                np.isnan(result.params['Ea'].stderr) == False):
                
                self.succes_count += 1
            else:
                
                # Failed errobar fitting, going to try and mess with 
                # starting poisition...
                for i in xrange(self.Niter):
                    # Fit Jmax vs T first
                    if self.peaked:
                        (ea_guess, 
                         dels_guess) = self.pick_random_starting_point()
                        params = self.setup_model_params(peaked=self.peaked, 
                                                 hd_guess=200000.0, 
                                                 ea_guess=ea_guess, 
                                                 dels_guess=dels_guess)
                    else:
                        params = Parameters()
                        ea_guess = np.random.uniform(20000.0, 80000.0)
                        params.add('Ea', value=ea_guess, min=0.0)
                    
                    result = minimize(self.residual, params, method="leastsq", 
                              args=(data, data["Vnorm"]))
                    
                    if result.errorbars:
                        self.succes_count += 1
                        break
            
            if print_to_screen:
                self.print_fit_to_screen(result)
            
        
            (peak_fit) = self.forward_run(result, data)
            if self.peaked:
                Topt = (self.calc_Topt(result.params["Hd"].value, 
                                       result.params["Ea"].value, 
                                       result.params["delS"].value))
            else:
                Topt = -9999.9 # not calculated
            
            self.report_fits(wr, result, data, data["Vnorm"], peak_fit, 
                             "Vcmax", Topt, id)
            
        fp.close()  
        
    def forward_run(self, result, data):
        """ Run peaked Arrhenius model with fitted parameters and return result 
        
        Parameters
        ----------
        result : object
            fitting result, param, std. error etc.
        data : object
            input A-Ci curve information
        
        Returns
        --------
        model_fit : array
            fitted result
        """
        Ea = result.params['Ea'].value
        
        if self.peaked:
            Hd = result.params['Hd'].value
            delS = result.params['delS'].value
        
            # First arg is 1.0, as this is calculated at the normalised temp
            model_fit = self.call_model(1.0, Ea, data["Tav"], delS, Hd)                                         
        else:
            model_fit = self.call_model(1.0, Ea, data["Tav"])     
        
        return model_fit
    
    def report_fits(self, f, result, data, obs, fit, pname, Topt, id):
        """ Save fitting results to a file... 
        
        Parameters
        ----------
        f : object
            file pointer
        result: object
            fitting result, param, std. error etc.
        data : object
            input A-Ci curve information
        obs : array
            A-Ci data to fit model against
        fit : array
            best model fit to optimised parameters
        fname : string
            filename to append to output file
        pname : string 
            param_name
        Topt : float
            Optimum temperature [deg C]
        """
        pearsons_r = stats.pearsonr(obs, fit)[0]
        diff_sq = (obs-fit)**2
        ssq = np.sum(diff_sq)
        mean_sq_err = np.mean(diff_sq)
        
        row = [pname]
        for name, par in result.params.items():
            row.append("%s" % (par.value))
            row.append("%s" % (par.stderr))
        row.append("%s" % ((obs-fit).var()))
        row.append("%s" % (pearsons_r**2))
        row.append("%s" % (ssq))
        row.append("%s" % (mean_sq_err))
        row.append("%s" % (len(obs)-1))
        row.append("%s" % (len(fit)))
        row.append("%s" % (Topt))
        row.append("%s" % (id))
        f.writerow(row)
        
    def residual(self, parameters, data, obs):
        """ simple function to quantify how good the fit was for the fitting
        routine. Could use something better? RMSE?
        
        Parameters
        ----------
        params : object
            List of parameters to be fit, initial guess, ranges etc. This is
            an lmfit object
        data: array
            data to run farquhar model with
        obs : array
            A-Ci data to fit model against
        
        Returns: 
        --------
        residual : array
            residual of fit between model and obs, based on current parameter
            set
        """
        Ea = parameters["Ea"].value
        
        if self.peaked:
            Hd = parameters["Hd"].value
            delS = parameters["delS"].value
            # First arg is 1.0, as this is calculated at the normalised temp
            model = self.call_model(1.0, Ea, data["Tav"], delS, Hd)                                         
        else:
            model = self.call_model(1.0, Ea, data["Tav"])   
            
        return (obs - model)
        
    def pick_random_starting_point(self):
        """ random pick starting point for parameter values 
        
        Parameters
        ----------
        data : array
            model driving data
        grid_size : int
            hardwired, number of samples
        
        Returns: 
        --------
        retval * 3 : float
            Three starting guesses for Jmax, Vcmax and Rd
        """
        Ea = np.random.uniform(20000.0, 80000.0)
        delS = np.random.uniform(550.0, 700.0)
        
        return Ea, delS
    
    
    def pick_starting_point(self, data, obs, grid_size=50):
        """ High-density grid search to overcome issues with ending up in a 
        local minima. Values that yield the lowest RMSE (without minimisation)
        are used as the starting point for the minimisation.
        
        Reference:
        ----------
        Dubois et al (2007) Optimizing the statistical estimation of the 
        parameters of the Farquhar-von Caemmerer-Berry model of photosynthesis. 
        New Phytologist, 176, 402--414
        """
        # Shuffle arrays so that our combination of parameters is random
        Hd = 200000.0
        Ea = np.linspace(20000.0, 80000.0, grid_size) 
        delS = np.linspace(550.0, 700.0, grid_size)
     
        p1, p2 = np.ix_(Ea, delS)
       
        if self.peaked:
            # First arg is 1.0, as this is calculated at the normalised temp
            model = self.call_model(1.0, p1, data["Tav"][:,None,None], p2, Hd)                                  
        
        rmse = np.sqrt(((obs[:,None,None]- model)**2).mean(0))
        ndx = np.where(rmse.min()== rmse)
        (idx, jdx) = ndx
        
        return Ea[idx].squeeze(), delS[jdx].squeeze()

    def setup_model_params(self, peaked, hd_guess, ea_guess, dels_guess):
        """ Setup lmfit Parameters object
        
        Parameters
        ----------
        
        
        Returns
        -------
        params : object
            lmfit object containing parameters to fit
        """
        params = Parameters()
        if peaked:
            params.add('Ea', value=ea_guess, min=0.0, max=199999.9)
            params.add('delS', value=dels_guess, min=0.0, max=800.0)  
            params.add('Hd', value=200000.0, vary=False)
        else:
            params.add('Ea', value=ea_guess, min=0.0, max=199999.9)
        
        return params
    
   
    
class FitK25EaDels(FitMe):
    """ Fit the model parameters K25j, K25v, Eaj, Eav, Dels to the measured 
        A-Ci data, i.e. not fitting normalised data!!!
    """
    def __init__(self, model=None, infname=None, ofname=None, results_dir=None, 
                 data_dir=None, peaked=True):
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
        """
        
        self.infname = infname
        FitMe.__init__(self, model=model, ofname=ofname, 
                       results_dir=results_dir, data_dir=data_dir)
       
        self.peaked = peaked
        if self.peaked:
            self.call_model = model.peaked_arrh
            self.header = ["Param", "K25", "SE", "Ea", "SE", "Hd", "SE", \
                            "delS", "delSSE", "Var", "R2", "SSQ", "MSE", "DOF", \
                            "n", "Topt", "ID"]
        else:
            self.call_model = model.arrh
            self.header = ["Param", "K25", "SE", "Ea", "SE", "Var", "R2", "SSQ", \
                            "MSE", "DOF", "n", "Topt", "ID"]
        
    def main(self, print_to_screen):   
        """ Loop over all our A-Ci measured curves and fit the Farquhar model
        parameters to this data 
        
        Parameters
        ----------
        print_to_screen : logical
            print fitting result to screen? Default is no!
        """
        all_data = self.read_data(self.infname, infile_type="meas")
        fp = self.open_output_files(self.ofname)
        wr = self.write_file_hdr(fp, self.header)
        
        # Loop over all the measured data and fit the model params.
        for id in np.unique(all_data["fitgroup"]):
            data = all_data[np.where(all_data["fitgroup"] == id)]
            
            # Fit Jmax vs T first
            if self.peaked:
                (k25_guess, 
                 ea_guess, 
                 dels_guess) = self.pick_starting_point(data, data["Vnorm"])
                params = self.setup_model_params(k25_guess, ea_guess, dels_guess)
            else:
                (k25_guess, 
                 ea_guess) = self.pick_starting_point(data, data["Vnorm"])
                params = self.setup_model_params(k25_guess, ea_guess)
            result = minimize(self.residual, params, method="leastsq", 
                              args=(data, data["Jmax"]))
            
            # Did we resolve the error bars during the fit? 
            #
            # From lmfit...
            #
            # In some cases, it may not be possible to estimate the errors 
            # and correlations. For example, if a variable actually has no 
            # practical effect on the fit, it will likely cause the 
            # covariance matrix to be singular, making standard errors 
            # impossible to estimate. Placing bounds on varied Parameters 
            # makes it more likely that errors cannot be estimated, as 
            # being near the maximum or minimum value makes the covariance 
            # matrix singular. In these cases, the errorbars attribute of 
            # the fit result (Minimizer object) will be False.
            if (result.errorbars and 
                np.isnan(result.params['K25'].stderr) == False):
                self.succes_count += 1
            else:
                
                # Failed errobar fitting, going to try and mess with 
                # starting poisition...
                for i in xrange(self.Niter):
                    # Fit Jmax vs T first
                    if self.peaked:
                        (k25_guess, ea_guess, 
                         dels_guess) = pick_random_starting_point() 
                        params = self.setup_model_params(k25_guess, ea_guess, 
                                                         dels_guess)
                    else:
                        (k25_guess, ea_guess) = pick_random_starting_point()
                        params = self.setup_model_params(k25_guess, ea_guess)
                    result = minimize(self.residual, params, method="leastsq", 
                                      args=(data, data["Jmax"]))
                    
                    if result.errorbars:
                        self.succes_count += 1
                        break
            
            if print_to_screen:
                self.print_fit_to_screen(result)
        
            (peak_fit) = self.forward_run(result, data)
            if self.peaked:
                Topt = (self.calc_Topt(result.params["Hd"].value, 
                                       result.params["Ea"].value, 
                                       result.params["delS"].value))
            else:
                Topt = -9999.9 # not calculated
            
            self.report_fits(wr, result, data, data["Jmax"], peak_fit, 
                             "Jmax", Topt, id)
           
            # Fit Vcmax vs T next 
            if self.peaked:
                (k25_guess, 
                 ea_guess, 
                 dels_guess) = self.pick_starting_point(data, data["Vnorm"])
                params = self.setup_model_params(k25_guess, ea_guess, dels_guess)
            else:
                (k25_guess, 
                 ea_guess) = self.pick_starting_point(data, data["Vnorm"])
                params = self.setup_model_params(k25_guess, ea_guess)
            result = minimize(self.residual, params, method="leastsq", 
                              args=(data, data["Vcmax"]))
            # Did we resolve the error bars during the fit? 
            #
            # From lmfit...
            #
            # In some cases, it may not be possible to estimate the errors 
            # and correlations. For example, if a variable actually has no 
            # practical effect on the fit, it will likely cause the 
            # covariance matrix to be singular, making standard errors 
            # impossible to estimate. Placing bounds on varied Parameters 
            # makes it more likely that errors cannot be estimated, as 
            # being near the maximum or minimum value makes the covariance 
            # matrix singular. In these cases, the errorbars attribute of 
            # the fit result (Minimizer object) will be False.
            if (result.errorbars and 
                np.isnan(result.params['K25'].stderr) == False):
                self.succes_count += 1
            else:
                
                # Failed errobar fitting, going to try and mess with 
                # starting poisition...
                for i in xrange(self.Niter):
                    # Fit Vcmax vs T next 
                    if self.peaked:
                        (k25_guess, ea_guess, 
                         dels_guess) = pick_random_starting_point() 
                        params = self.setup_model_params(k25_guess, ea_guess, 
                                                         dels_guess)
                    else:
                        (k25_guess, ea_guess) = pick_random_starting_point()
                        params = self.setup_model_params(k25_guess, ea_guess)
                    result = minimize(self.residual, params, method="leastsq", 
                                      args=(data, data["Vcmax"]))
                    
                    if result.errorbars:
                        self.succes_count += 1
                        break
            
            if print_to_screen:
                self.print_fit_to_screen(result)
        
            (peak_fit) = self.forward_run(result, data)
            if self.peaked:
                Topt = (self.calc_Topt(result.params["Hd"].value, 
                                       result.params["Ea"].value, 
                                       result.params["delS"].value))
            else:
                Topt = -9999.9 # not calculated
            
            self.report_fits(wr, result, data, data["Vcmax"], peak_fit, 
                             "Vcmax", Topt, id)
            
        fp.close()  
        
    def forward_run(self, result, data):
        """ Run peaked Arrhenius model with fitted parameters and return result 
        
        Parameters
        ----------
        result : object
            fitting result, param, std. error etc.
        data : object
            input A-Ci curve information
        
        Returns
        --------
        model_fit : array
            fitted result
        """
        Ea = result.params['Ea'].value
        K25 = result.params['K25'].value
        
        if self.peaked:
            Hd = result.params['Hd'].value
            delS = result.params['delS'].value
            model_fit = self.call_model(K25, Ea, data["Tav"], delS, Hd)                                         
        else:
            model_fit = self.call_model(K25, Ea, data["Tav"])     
        
        return model_fit
    
    def report_fits(self, f, result, data, obs, fit, pname, Topt, id):
        """ Save fitting results to a file... 
        
        Parameters
        ----------
        f : object
            file pointer
        result: object
            fitting result, param, std. error etc.
        data : object
            input A-Ci curve information
        obs : array
            A-Ci data to fit model against
        fit : array
            best model fit to optimised parameters
        fname : string
            filename to append to output file
        pname : string 
            param_name
        Topt : float
            Optimum temperature [deg C]
        """
        pearsons_r = stats.pearsonr(obs, fit)[0]
        diff_sq = (obs-fit)**2
        ssq = np.sum(diff_sq)
        mean_sq_err = np.mean(diff_sq)
        
        row = [pname]
        for name, par in result.params.items():
            row.append("%s" % (par.value))
            row.append("%s" % (par.stderr))
        row.append("%s" % ((obs-fit).var()))
        row.append("%s" % (pearsons_r**2))
        row.append("%s" % (ssq))
        row.append("%s" % (mean_sq_err))
        row.append("%s" % (len(fit)-1))
        row.append("%s" % (len(fit)))
        row.append("%s" % (Topt))
        row.append("%s" % (id))
        f.writerow(row)
        
    def residual(self, parameters, data, obs):
        """ simple function to quantify how good the fit was for the fitting
        routine. Could use something better? RMSE?
        
        Parameters
        ----------
        params : object
            List of parameters to be fit, initial guess, ranges etc. This is
            an lmfit object
        data: array
            data to run farquhar model with
        obs : array
            A-Ci data to fit model against
        
        Returns: 
        --------
        residual : array
            residual of fit between model and obs, based on current parameter
            set
        """
        Ea = parameters["Ea"].value
        K25 = parameters['K25'].value
        if self.peaked:
            Hd = parameters["Hd"].value
            delS = parameters["delS"].value
            model = self.call_model(K25, Ea, data["Tav"], delS, Hd)                                         
        else:
            model = self.call_model(K25, Ea, data["Tav"])   
            
        return (obs - model)

    def setup_model_params(self, k25_guess, ea_guess, dels_guess=None):
        """ Setup lmfit Parameters object
        
        Parameters
        ----------
        
        
        Returns
        -------
        params : object
            lmfit object containing parameters to fit
        """
        params = Parameters()
        if self.peaked:
            params.add('K25', value=k25_guess, min=0.0)
            params.add('Ea', value=ea_guess, min=0.0, max=199999.9)
            params.add('delS', value=dels_guess, min=0.0, max=800.0)  
            params.add('Hd', value=200000.0, vary=False)
        else:
            params = Parameters()
            params.add('K25', value=k25_guess, min=0.0)
            params.add('Ea', value=ea_guess, min=0.0, max=199999.9)
        
        return params
        
    def pick_starting_point(self, data, obs, grid_size=50):
        """ High-density grid search to overcome issues with ending up in a 
        local minima. Values that yield the lowest RMSE (without minimisation)
        are used as the starting point for the minimisation.
        
        Reference:
        ----------
        Dubois et al (2007) Optimizing the statistical estimation of the 
        parameters of the Farquhar-von Caemmerer-Berry model of photosynthesis. 
        New Phytologist, 176, 402--414
        """
        # Shuffle arrays so that our combination of parameters is random
        Hd = 200000.0
        Ea = np.linspace(20000.0, 80000.0, grid_size) 
        delS = np.linspace(550.0, 700.0, grid_size)
        k25 = np.linspace(50.0, 250.0, grid_size)
      
        if self.peaked:
            p1, p2, p3 = np.ix_(k25, Ea, delS)
            model = self.call_model(p1, p2, data["Tav"][:,None,None,None], p3, Hd)
            rmse = np.sqrt(((obs[:,None,None,None]- model)**2).mean(0))
            ndx = np.where(rmse.min()== rmse)
            (idx, jdx, kdx) = ndx 
            
            return k25[idx].squeeze(), Ea[jdx].squeeze(), delS[kdx].squeeze()                                  
        else:
            p1, p2 = np.ix_(k25, Ea)
            model = self.call_model(K25, Ea, data["Tav"][:,None,None])
            rmse = np.sqrt(((obs[:,None,None]- model)**2).mean(0))
            ndx = np.where(rmse.min()== rmse)
            (idx, jdx) = ndx 
        
            return k25[idx].squeeze(), Ea[jdx].squeeze()
    
    def pick_random_starting_point(self):
        """ random pick starting point for parameter values 
        
        Parameters
        ----------
        data : array
            model driving data
        grid_size : int
            hardwired, number of samples
        
        Returns: 
        --------
        retval * 3 : float
            Three starting guesses for Jmax, Vcmax and Rd
        """
        if self.peaked:
            Ea = np.random.uniform(20000.0, 80000.0) 
            delS = np.random.uniform(550.0, 700.0)
            k25 = np.random.uniform(50.0, 250.0)
        
            return K25, Ea, delS  
        else:
            Ea = np.random.uniform(20000.0, 80000.0) 
            delS = np.random.uniform(550.0, 700.0)
            k25 = np.random.uniform(50.0, 250.0)
        
            return K25, Ea
     