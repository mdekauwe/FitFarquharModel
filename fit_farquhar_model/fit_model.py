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
from lmfit import minimize, Parameters, printfuncs, conf_interval
from scipy import stats
import matplotlib.pyplot as plt


class FitMe(object):
    """
    Basic fitting class, contains some generic methods which are used by the
    fitting routines, e.g. plotting, file reporting etc. This is intended
    to be subclased.
    """
    def __init__(self, model=None, ofname=None, results_dir=None, 
                 data_dir=None, plot_dir=None, random_sample_grid=None):
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
        random_sample_grid : logical
            If this is true we use a random sampling procedure to define the
            best starting point for initial parameter guesses. If this is false
            then we use a much denser sampling method. Note this will be 
            considerably more computational expensive, up to you.
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
        self.random_sample_grid = random_sample_grid
    
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
        
    def read_data(self, fname, delimiter=","):
        """ Read the A-Ci data. 
        
        Expects a format of:
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
        (An, Anc, Anj) = self.call_model(data["Ci"], data["Tleaf"], Jmax=Jmax, 
                                         Vcmax=Vcmax, Rd=Rd)
        return (obs - An)
    
    def report_fits(self, f, result, fname, curve_data, An_fit):
        """ Save fitting results to a file... """
        pearsons_r = stats.pearsonr(curve_data["Photo"], An_fit)[0]
        row = []
        for name, par in result.params.items():
            row.append("%s" % (par.value))
            row.append("%s" % (par.stderr))
        row.append("%s" % (np.mean(curve_data["Tleaf"] - self.deg2kelvin)))
        row.append("%s" % (pearsons_r**2))
        row.append("%s" % (len(An_fit)))
        row.append("%s" % (curve_data["Species"][0]))
        row.append("%s" % (curve_data["Season"][0]))
        row.append("%s" % (curve_data["Leaf"][0]))
        row.append("%s" % (curve_data["Curve"][0]))
        row.append("%s" % (fname))
        row.append("%s%s%s" % (str(curve_data["Species"][0]), \
                               str(curve_data["Season"][0]), \
                               str(curve_data["Leaf"][0])))
        f.writerow(row)
       
    def forward_run(self, result, data):
        Jmax = result.params['Jmax'].value
        Vcmax = result.params['Vcmax'].value
        Rd = result.params['Rd'].value  
        (An_fit, Anc_fit, Anj_fit) = self.call_model(data["Ci"], data["Tleaf"], 
                                                     Jmax=Jmax, Vcmax=Vcmax, 
                                                     Rd=Rd)
        
        return (An_fit, Anc_fit, Anj_fit)
                
    def setup_model_params(self, jmax_guess=None, vcmax_guess=None, 
                           rd_guess=None, hd_guess=None, ea_guess=None, 
                           dels_guess=None):
        """ Setup parameters """
        params = Parameters()
        if jmax_guess is not None:
            params.add('Jmax', value=jmax_guess, min=0.0)
        if vcmax_guess is not None:
            params.add('Vcmax', value=vcmax_guess, min=0.0)
        if rd_guess is not None:
            params.add('Rd', value=rd_guess)
        if hd_guess is not None:
            params.add('Hd', value=hd_guess, vary=False)
        if ea_guess is not None:
            params.add('Ea', value=ea_guess, min=0.0)
        if dels_guess is not None:
            params.add('delS', value=dels_guess, min=0.0, max=700.0)
        
        return params
    
    def print_fit_to_screen(self, result):
        for name, par in result.params.items():
            print '%s = %.8f +/- %.8f ' % (name, par.value, par.stderr)
        print 
  
    
    def make_plot(self, curve_data, curve_num, An_fit, Anc_fit, Anj_fit):
        """ Make some plots to show how good our fitted model is to the data """
        species = curve_data["Species"][0]
        season = curve_data["Season"][0]
        season = "all"
        leaf = curve_data["Leaf"][0]
        ofname = "%s/%s_%s_%s_%s_fit_and_residual.png" % \
                 (self.plot_dir, species, season, leaf, curve_num)
        residuals = curve_data["Photo"] - An_fit  
        
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
        
        ax.plot(curve_data["Ci"], curve_data["Photo"], 
                ls="", lw=1.5, marker="o", c="black")
        ax.plot(curve_data["Ci"], An_fit, '-', c="black", linewidth=1, 
                label="An-Rd")
        ax.plot(curve_data["Ci"], Anc_fit, '--', c="red", linewidth=3, 
                label="Ac-Rd")
        ax.plot(curve_data["Ci"], Anj_fit, '--', c="blue", linewidth=3, 
                label="Aj-Rd")
        ax.set_ylabel("A$_n$", weight="bold")
        ax.axes.get_xaxis().set_visible(False)
        ax.set_xlim(0, 1600)
        ax.legend(numpoints=1, loc="best")

        ax2.plot(curve_data["Ci"], residuals, "ko")
        ax2.axhline(y=0.0, ls="--", color='black')
        ax2.set_xlabel('Ci', weight="bold")
        ax2.set_ylabel("Residuals (Obs$-$Fit)", weight="bold")
        ax2.set_xlim(0, 1500)
        ax2.set_ylim(10,-10)
        
        fig.savefig(ofname)
        plt.clf()    

    def write_file_hdr(self, fname, header):  
        wr = csv.writer(fname, delimiter=',', quoting=csv.QUOTE_NONE, 
                        escapechar=' ')
        wr.writerow(header)
        
        return wr
        
    def tidy_up(self, fp=None):
        total_fits = float(self.succes_count) / self.nfiles * 100.0
        print "\nOverall fitted %.1f%% of the data\n" % (total_fits)
        fp.close()
    
    def pick_starting_point(self, data, grid_size=500):
        """ Figure out a good starting parameter guess
        
        High-density grid search to overcome issues with ending up in a 
        local minima. Values that yield the lowest SSE (without minimisation)
        are used as the starting point for the minimisation. There is also the
        option to randomly sample, I think I ought to remove this now.
        
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
        Rd = np.linspace(1E-8, 10.5, grid_size)
        fits = np.zeros(0)
        
        if self.random_sample_grid:
            np.random.shuffle(Vcmax)
            np.random.shuffle(Jmax)
            np.random.shuffle(Rd)
        
            for i in xrange(len(Vcmax)):
                (An, Anc, Anj) = self.call_model(data["Ci"], data["Tleaf"], 
                                                 Jmax=Jmax[i], Vcmax=Vcmax[i], 
                                                 Rd=Rd[i])     
                # Save SSE
                fits = np.append(fits, np.sum((data["Photo"] - An)**2))
        else:
            # merge arrays, so we only have a single loop (for speed!)
            x = np.dstack((Vcmax,Jmax, Rd)).flatten()
            for i in xrange(0,len(x), 3):
                (An, Anc, Anj) = self.call_model(data["Ci"], data["Tleaf"], 
                                                 Jmax=x[i+1], Vcmax=x[i], 
                                                 Rd=x[i+2])     
                # Save SSE
                fits = np.append(fits, np.sum((data["Photo"] - An)**2))
        index = np.argmin(fits, 0) # smalles SSE
        
        return Vcmax[index], Jmax[index], Rd[index]
       
        

class FitJmaxVcmaxRd(FitMe):
    """ Fit the model parameters Jmax, Vcmax and Rd to the measured A-Ci data"""
    def __init__(self, model=None, ofname=None, results_dir=None, 
                 data_dir=None, plot_dir=None, random_sample_grid=None):
        FitMe.__init__(self, model, ofname, results_dir, data_dir, plot_dir, 
                       random_sample_grid)
        self.header = ["Jmax", "JSE", "Vcmax", "VSE", "Rd", "RSE", "Tav", \
                       "R2", "n", "Species", "Season", "Leaf", "Curve", \
                       "Filename", "id"]
                       
    def main(self, print_to_screen, deg25_range=[None, None]):   
        """ Loop over all our A-Ci measured curves and fit the Farquhar model
        parameters to this data """
        
        # open files and write header information
        fp = self.open_output_files(self.ofname)
        wr = self.write_file_hdr(fp, self.header)
    
        # Loop over all the measured data and fit the model params.
        for fname in glob.glob(os.path.join(self.data_dir, "*.csv")):
            data = self.read_data(fname)
            data["Tleaf"] += self.deg2kelvin
            for curve_num in np.unique(data["Curve"]):
                curve_data = data[np.where(data["Curve"]==curve_num)]
                (vcmax_guess, jmax_guess, 
                    rd_guess) = self.pick_starting_point(curve_data)
                
                params = self.setup_model_params(jmax_guess=jmax_guess, 
                                                 vcmax_guess=vcmax_guess, 
                                                 rd_guess=rd_guess)
                result = minimize(self.residual, params, engine="leastsq", 
                                  args=(curve_data, curve_data["Photo"]))
                
                if print_to_screen:
                    self.print_fit_to_screen(result)
                
                # Did we resolve the error bars during the fit? If yes then
                # move onto the next A-Ci curve
                if result.errorbars:
                    self.succes_count += 1
                
                # Need to run the Farquhar model with the fitted params for
                # plotting...
                (An, Anc, Anj) = self.forward_run(result, curve_data)
                self.report_fits(wr, result, os.path.basename(fname), 
                                 curve_data, An)
                
                self.make_plot(curve_data, curve_num, An, Anc, Anj)
                self.nfiles += 1       
        self.tidy_up(fp)    
    

class FitEaDels(FitMe):
    """ Fit the model parameters Eaj, Eav, Dels to the measured A-Ci data"""
    def __init__(self, model=None, infname=None, ofname=None, results_dir=None, 
                 data_dir=None, random_sample_grid=None):
        self.infname = infname
        FitMe.__init__(self, model=model, ofname=ofname, 
                       results_dir=results_dir, data_dir=data_dir, 
                       random_sample_grid=random_sample_grid)
        self.header = ["Param", "Hd", "SE", "Ea", "SE", "delS", "delSSE", \
                       "R2", "n", "Topt"]
        self.call_model = model.peaked_arrh
        
    def main(self, print_to_screen):   
        """ Loop over all our A-Ci measured curves and fit the Farquhar model
        parameters to this data """
        all_data = self.get_data(self.infname)
        fp = self.open_output_files(self.ofname)
        wr = self.write_file_hdr(fp, self.header)
        
        # Loop over all the measured data and fit the model params.
        
        for id in np.unique(all_data["fitgroup"]):
            data = all_data[np.where(all_data["fitgroup"] == id)]
            
            # Fit Jmax vs T first
            (ea_guess, 
                dels_guess) = self.pick_starting_point(data, data["Jnorm"])  
            params = self.setup_model_params(hd_guess=200000.0, 
                                             ea_guess=ea_guess, 
                                             dels_guess=dels_guess)
            result = minimize(self.residual, params, engine="leastsq", 
                              args=(data, data["Jnorm"]))
            if print_to_screen:
                self.print_fit_to_screen(result)
            
            # Did we resolve the error bars during the fit? If yes then
            # move onto the next A-Ci curve
            if result.errorbars:
                self.succes_count += 1
        
            (peak_fit) = self.forward_run(result, data)
            Topt = (self.calc_Topt(result.params["Hd"].value, 
                               result.params["Ea"].value, 
                               result.params["delS"].value) - 
                               self.deg2kelvin)
            self.report_fits(wr, result, data, data["Jnorm"], peak_fit, 
                             "Jmax", Topt)
            
            # Fit Vcmax vs T next 
            (ea_guess, 
                dels_guess) = self.pick_starting_point(data, data["Vnorm"])
            params = self.setup_model_params(hd_guess=200000.0, 
                                             ea_guess=ea_guess, 
                                             dels_guess=dels_guess)
            result = minimize(self.residual, params, engine="leastsq", 
                              args=(data, data["Vnorm"]))
            if print_to_screen:
                self.print_fit_to_screen(result)
            
            # Did we resolve the error bars during the fit? If yes then
            # move onto the next A-Ci curve
            if result.errorbars:
                self.succes_count += 1
        
            (peak_fit) = self.forward_run(result, data)
            Topt = (self.calc_Topt(result.params["Hd"].value, 
                               result.params["Ea"].value, 
                               result.params["delS"].value) - 
                               self.deg2kelvin)
            self.report_fits(wr, result, data, data["Vnorm"], peak_fit, 
                             "Vcmax", Topt)
       
        fp.close()  
    
    def get_data(self, infname):
        all_data = self.read_data(infname)
        all_data["Tav"] = all_data["Tav"] + self.deg2kelvin
        all_data["Jnorm"] = np.exp(all_data["Jnorm"])
        all_data["Vnorm"] = np.exp(all_data["Vnorm"])
        
        return all_data
        
    def forward_run(self, result, data):
        Hd = result.params['Hd'].value
        Ea = result.params['Ea'].value
        delS = result.params['delS'].value
        
        # First arg is 1.0, as this is calculated at the normalised temp
        model_fit = self.call_model(1.0, Ea, data["Tav"], delS, Hd)                                         
        
        return model_fit
    
    def report_fits(self, f, result, data, obs, fit, pname, Topt):
        """ Save fitting results to a file... """
        pearsons_r = stats.pearsonr(obs, fit)[0]
        row = [pname]
        for name, par in result.params.items():
            row.append("%s" % (par.value))
            row.append("%s" % (par.stderr))
        row.append("%s" % (pearsons_r**2))
        row.append("%s" % (len(fit)))
        row.append("%s" % (Topt))
        f.writerow(row)
    
    def residual(self, parameters, data, obs):
        Hd = parameters["Hd"].value
        Ea = parameters["Ea"].value
        delS = parameters["delS"].value
        
        # First arg is 1.0, as this is calculated at the normalised temp
        model = self.call_model(1.0, Ea, data["Tav"], delS, Hd)
        
        return (obs - model)
    
    def calc_Topt(self, Hd, Ha, delS, RGAS=8.314):
        """ calculate the temperature optimum """
        
        #print Ha, Hd, delS
        return Hd / (delS - RGAS * np.log(Ha / (Hd - Ha)))
        
    def pick_starting_point(self, data, obs, grid_size=100):
        """ High-density grid search to overcome issues with ending up in a 
        local minima. Values that yield the lowest SSE (without minimisation)
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
        fits = np.zeros(0)
        
        if self.random_sample_grid:
            np.random.shuffle(Ea)
            np.random.shuffle(delS)
        
            for i in xrange(len(Ea)):
                model = self.call_model(1.0, Ea[i], data["Tav"], delS[i], Hd)
                
                # Save SSE
                fits = np.append(fits, np.sum((obs - model)**2))
        else:
            # dense sampling, not selecting at random, this should be much 
            # more memory intensive, but perhaps preferable to the above...
            for i in xrange(len(Ea)):
                for j in xrange(len(delS)):
                    model = self.call_model(1.0, Ea[i], data["Tav"], delS[j],Hd)    
                    
                    # Save SSE
                    fits = np.append(fits, np.sum((obs - model)**2))      
        index = np.argmin(fits, 0) # smalles SSE
        
        return Ea[index], delS[index]
    