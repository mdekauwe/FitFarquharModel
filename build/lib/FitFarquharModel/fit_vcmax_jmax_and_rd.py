#!/usr/bin/env python

"""
Using the Levenberg-Marquardt algorithm  to fit Jmax, Vcmax and Rd
The steps here are:
    1. try and fit the parameters but if this fails...
    2. assess whether it is because the Levenberg-Marquardt scheme is sensitive 
       to the initial starting point, so try a few more fits from different 
       points.

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
    
    def __init__(self, model=None, ofname=None, ofname25=None, results_dir=None, 
                 data_dir=None, plot_dir=None, nstartpos=30):
        
        self.results_dir = results_dir
        self.ofname = os.path.join(self.results_dir, ofname)
        self.ofname25 = os.path.join(self.results_dir, ofname25)
        self.data_dir = data_dir 
        self.plot_dir = plot_dir    
        self.call_model = model.calc_photosynthesis
        self.succes_count = 0
        self.nfiles = 0
        self.nstartpos = nstartpos
        self.deg2kelvin = 273.15
        
    def main(self, print_to_screen=False):
        """ Loop over all our A-Ci measured curves and fit the Farquhar model
        parameters to this data """
        
        # open files and write header information
        fp = self.open_output_files(self.ofname)
        wr = self.write_file_hdr(fp)
        fp25 = self.open_output_files(self.ofname25)
        wr25 = self.write_file_hdr(fp25)
        
        # Loop over all the measured data and fit the model params.
        for fname in glob.glob(os.path.join(self.data_dir, "*.csv")):
            data = self.read_data(fname)
            for curve_num in np.unique(data["Curve"]):
                curve_data = data[np.where(data["Curve"]==curve_num)]
                params = self.setup_model_params(jmax_guess=180.0, 
                                                 vcmax_guess=50.0, rd_guess=2.0)
                result = minimize(self.residual, params, 
                                          engine="leastsq", 
                                          args=(curve_data, curve_data["A"]))
                
                if print_to_screen:
                    self.print_fit_to_screen(result)
                
                # Did we resolve the error bars during the fit? If yes then
                # move onto the next A-Ci curve
                if result.errorbars:
                    self.succes_count += 1
                
                # Otherwise we did not fit the data. Is it because of our 
                # starting position?? Lets vary this a little and redo the fits
                else:
                    for i in xrange(self.nstartpos):
                        params = self.try_new_params()
                        result = minimize(self.residual, params, 
                                          engine="leastsq", 
                                          args=(curve_data, curve_data["A"]))
                        
                        
                        
                        if print_to_screen:
                            self.print_fit_to_screen(result)
                        if result.errorbars:
                            succes_count += 1
                            break
                # Need to run the Farquhar model with the fitted params for
                # plotting...
                (An, Anc, Anj) = self.forward_run(result, curve_data)
                self.report_fits(wr, result, os.path.basename(fname), 
                                 curve_data, An)
                # Figure out which of our curves (if any) were measured at
                # 25 degrees C
                Tavg = np.mean(curve_data["Tleaf"]) - self.deg2kelvin
                if Tavg > 26.0 and Tavg < 29.0:
                    self.report_fits(wr25, result, os.path.basename(fname), 
                                     curve_data, An)  
                
                self.make_plot(curve_data, curve_num, An, Anc, Anj)
                self.nfiles += 1       
        self.tidy_up(fp, fp25)    
    
    def open_output_files(self, ofname):
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
        -> Curve, Tleaf, Ci, A, Species, Season, Plant
        """
        data = np.recfromcsv(fname, delimiter=delimiter, names=True, 
                             case_sensitive=True)
        data["Tleaf"] += self.deg2kelvin
        
        return data
                      
    def residual(self, params, data, obs):
        """ simple function to quantify how good the fit was for the fitting
        routine. Could use something better? RMSE?
        """
        Jmax = params['Jmax'].value
        Vcmax = params['Vcmax'].value
        Rd = params['Rd'].value  
        (An, Anc, Anj) = self.call_model(data["Ci"], data["Tleaf"], Jmax=Jmax, 
                                         Vcmax=Vcmax, Rd=Rd)
        return (obs - An)
    
    def report_fits(self, f, result, fname, curve_data, An_fit):
        """ Save fitting results to a file... """
        pearsons_r = stats.pearsonr(curve_data["A"], An_fit)[0]
        row = []
        for name, par in result.params.items():
            row.append("%s" % (par.value))
            row.append("%s" % (par.stderr))
        row.append("%s" % (np.mean(curve_data["Tleaf"] - self.deg2kelvin)))
        row.append("%s" % (pearsons_r**2))
        row.append("%s" % (len(An_fit)))
        row.append("%s" % (curve_data["Species"][0]))
        row.append("%s" % (curve_data["Season"][0]))
        row.append("%s" % (curve_data["Plant"][0]))
        row.append("%s" % (curve_data["Curve"][0]))
        row.append("%s" % (fname))
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
                          rd_guess=None):
        """ Setup parameters """
        params = Parameters()
        params.add('Jmax', value=jmax_guess, min=0.0)
        params.add('Vcmax', value=vcmax_guess, min=0.0)
        params.add('Rd', value=rd_guess)
        
        return params
    
    def try_new_params(self):
        """ Fitting routine can be sensitive to the starting guess, so randomly
        alter the initial guess """
        params = Parameters()
        params.add('Jmax', value=np.random.uniform(1.0, 250.0), min=0.0)
        params.add("Vcmax", value= np.random.uniform(1.0, 250.0), min=0.0)
        params.add('Rd', value=np.random.uniform(0.1, 6.0))
        
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
        plant = curve_data["Plant"][0]
        ofname = "%s/%s_%s_%s_%s_fit_and_residual.png" % \
                 (self.plot_dir, species, season, plant, curve_num)
        residuals = curve_data["A"] - An_fit  
        
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
        ax.plot(curve_data["Ci"], curve_data["A"], 
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

    def write_file_hdr(self, fname):  
        header = ["Jmax", "JSE", "Vcmax", "VSE", "Rd", "RSE", "Tav", "R2", \
                  "n", "Species", "Season", "Plant", "Curve", "Filename"]
        wr = csv.writer(fname, delimiter=',', quoting=csv.QUOTE_NONE, 
                        escapechar=' ')
        wr.writerow(header)
        
        return wr
        
    def tidy_up(self, fp, fp25):
        total_fits = float(self.succes_count) / self.nfiles * 100.0
        print "\nOverall fitted %.1f%% of the data\n" % (total_fits)
        fp.close()
        fp25.close() 
    
 
if __name__ == "__main__":
    
    F = FitMe(ofname="fitting_results.txt", ofname25="params_at_25.txt", 
              results_dir="results", data_dir="data", plot_dir="plots")
    F.main()     