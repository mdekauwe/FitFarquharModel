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

from FitFarquharModel import farquhar_model as FarquharC3

class FitMe(object):
    
    def __init__(self, ofname=None, ofname25=None, results_dir=None, 
                 data_dir=None, plot_dir=None):
        
        self.results_dir = results_dir
        self.ofname = os.path.join(self.results_dir, ofname)
        self.ofname25 = os.path.join(self.results_dir, ofname25)
        self.data_dir = data_dir 
        self.plot_dir = plot_dir    
        self.mod = FarquharC3()
        self.succes_count = 0
        self.nfiles = 0
        self.nstartpos = 30
        self.deg2kelvin = 273.15

    def main():
        self.write_file_hdr(self.ofname)
        
        for fname in glob.glob(os.path.join(self.data_dir, "*.csv")):
            print fname
            #data = self.read_data(fname)
            
            
            """
            for curve_num in np.unique(data["Curve"]):
                curve_data = data[np.where(data["Curve"]==curve_num)]
                params = FF.setup_model_params()
                result = FF.minimise_params(params, curve_data, curve_data["Photo"], 
                                            engine="leastsq")
                #FF.print_fit_to_screen(result)
                
                # Did we resolve the error bars during the fit?
                if result.errorbars:
                    succes_count += 1
                else:
                    # did the fit fail because of our starting position?? Lets vary 
                    # this a little and redo the fits
                    for i in xrange(nstartpos):
                        params = FF.try_new_params()
                        result = FF.minimise_params(params, curve_data, 
                                                    curve_data["Photo"], 
                                                    engine="leastsq")
                        FF.print_fit_to_screen(result)
                        if result.errorbars:
                            succes_count += 1
                            break
                FF.forward_run(result, curve_data)
                FF.report_fits(f, os.path.basename(fname), result, curve_data)
                Tavg = np.mean(curve_data["Tleaf"]) - deg2kelvin
     #           print Tavg
                if (Tavg>26.0) & (Tavg<29.0) :
                    FF.report_fits(f25, os.path.basename(fname), result, curve_data)  
                FF.make_plot(curve_data, curve_num)
                nfiles += 1        
                       
    total_fits = float(succes_count) / nfiles * 100.0
    print "\nOverall fitted %.1f%% of the data\n" % (total_fits)
    f.close()
    f25.close()
    """
    
    def read_data(self, fname, comment='#'):
        """ Read the A-Ci data. 
        Format = Curve,Tleaf,Ci,A,Species,Season,Plant
        """
        data = np.recfromcsv(fname, delimiter=",", names=True, 
                             case_sensitive=True)
        data["Tleaf"] += self.deg2kelvin
        
        return data
                      
    def residual(self, params, curve_data, obs):
        Jmax = params['Jmax'].value
        Vcmax = params['Vcmax'].value
        Rd = params['Rd'].value  
        (An, Ac, Aj) = self.call_model(curve_data["Ci"], curve_data["Tleaf"], 
                                Jmax, Vcmax, Rd)
        return (obs - An)
    
    def report_fits(self, f, fname, result, curve_data, topt=None):
        """ Save fitting results """
        pearsons_r = stats.pearsonr(curve_data["Photo"], self.An_fit)[0]
        for name, par in result.params.items():
            f.write('%.4f %.4f ' % (par.value, par.stderr)),
            
        f.write('%.4f ' % (np.mean(curve_data["Tleaf"]-self.deg2kelvin)))
        f.write('%.4f %d ' % (pearsons_r**2, len(self.An_fit)))
        f.write('%s %s %s %s' % (curve_data["Species"][0], 
 #                                curve_data["Season"][0],
                                 curve_data["Plant"][0], 
                                 curve_data["Curve"][0],fname))
        f.write("\n")
        
                 
    def forward_run(self, result, curve_data):
        Jmax = result.params['Jmax'].value
        Vcmax = result.params['Vcmax'].value
        Rd = result.params['Rd'].value  
        (self.An_fit, self.Acn_fit, 
            self.Ajn_fit) = self.call_model(curve_data["Ci"], curve_data["Tleaf"], 
                                           Jmax, Vcmax, Rd)              
        return self.An_fit, self.Acn_fit, self.Ajn_fit
                 
    def setup_model_params(self):
        """ Setup parameters """
        params = Parameters()
        params.add('Jmax', value=180.0, min=0.0)
        params.add('Vcmax', value=50.0, min=0.0)
        params.add('Rd', value=2.0)
        
        return params
    
    def try_new_params(self):
        """ Fitting routine can be sensitive to the starting guess, so randomly
        alter the initial guess """
        params = Parameters()
        params.add('Jmax', value= np.random.uniform(1.0, 250.0), min=0.0)
        params.add("Vcmax", value= np.random.uniform(1.0, 250.0), min=0.0)
        #params.add('Vcmax', value=np.random.uniform(1.0, 250.0), min=0.0)
        params.add('Rd', value=np.random.uniform(0.1, 6.0))
        #params.add('Rd', value=1.5,vary=False)
        
        return params

   
    def print_fit_to_screen(self, result):
        for name, par in result.params.items():
            print '%s = %.8f +/- %.8f ' % (name, par.value, par.stderr)
        print 
        
    def minimise_params(self, params, data, obs, engine="leastsq"):
        result = minimize(self.residual, params, engine=engine, 
                                  args=(data, obs))
        return result
    
    def make_plot(self, curve_data, curve_num):
        
        species = curve_data["Species"][0]
 #       season = curve_data["Season"][0]
        season = "all"
        plant = curve_data["Plant"][0]
        ofname = "plots/%s_%s_%s_%s_fit_and_residual.png" % \
                 (species, season, plant, curve_num)
        residuals = curve_data["Photo"] - self.An_fit  
        
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
        
        
        ax.plot(curve_data["Ci"], self.An_fit, '-', c="black", linewidth=1, label="An-Rd")
        ax.plot(curve_data["Ci"], self.Acn_fit, '--', c="red", linewidth=3, label="Ac-Rd")
        ax.plot(curve_data["Ci"], self.Ajn_fit, '--', c="blue", linewidth=3, label="Aj-Rd")
        ax.set_ylabel("A$_n$", weight="bold")
        ax.axes.get_xaxis().set_visible(False)
        ax.set_xlim(0, 1600)
        #ax.set_ylim(-5, 60)
        ax.legend(numpoints=1, loc="best")
        
        
        ax2.plot(curve_data["Ci"], residuals, "ko")
        ax2.axhline(y=0.0, ls="--", color='black')
        ax2.set_xlabel('Ci', weight="bold")
        ax2.set_ylabel("Residuals (Obs$-$Fit)", weight="bold")
        ax2.set_xlim(0, 1500)
        ax2.set_ylim(10,-10)
        fig.savefig(ofname)
        #plt.clf()    

    def write_file_hdr(fname):  
        if os.path.isfile(fname):
            os.remove(fname)
        try:
            self.odaily = open(fname, 'wb')
        except IOError:
            raise IOError("Can't open %s file for write" % fname)      
    
        header = ["Jmax", "JSE", "Vcmax", "VSE", "Rd", "RSE", "Tav", "R2", \
                  "n", "Species", "Season", "Plant", "Curve", "Filename"]
        wr = csv.writer(ffname, delimiter=',', quoting=csv.QUOTE_NONE, 
                        escapechar=' ')
        wr.writerow(header)
        
 
if __name__ == "__main__":
    
    F = FitMe(ofname="fitting_results.txt", ofname25="params_at_25.txt", 
              results_dir="results", data_dir="data", plot_dir="plots")
    F.main()     