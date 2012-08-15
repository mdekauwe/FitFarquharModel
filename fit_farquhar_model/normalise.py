#!/usr/bin/env python

"""
Normalise the data...write something sensible

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (14.08.2012)"
__email__ = "mdekauwe@gmail.com"

import csv
import sys
import numpy as np
import os
import glob
import matplotlib.mlab as mlab  # library to write structured array to file.
import matplotlib.pyplot as plt

class Normalise(object):
    def __init__(self, fname=None, ofname1=None, ofname2=None, results_dir=None, 
                 plot_dir=None, tnorm=None):
        self.results_dir = results_dir
        self.fname = os.path.join(self.results_dir, fname)
        self.ofname1 = os.path.join(self.results_dir, ofname1)
        self.ofname2 = os.path.join(self.results_dir, ofname2)
        self.plot_dir = plot_dir    
        self.tnorm = tnorm # Temperature we are normalising to
        self.deg2kelvin = 273.15
        self.header1 = ["Jmax", "Vcmax", "Species", "id", "Filename"]
        self.header2 = ["Jmax", "Vcmax", "Jnorm", "Vnorm", "Rd", "Tav", \
                        "Tarrh", "R2", "n", "Species", "Leaf", "Curve",\
                        "Filename"]
    def main(self):
        data_all = self.read_data(self.fname)
       
        # open files and write header information
        fp1 = self.open_output_files(self.ofname1)
        wr1 = self.write_file_hdr(fp1, self.header1)
        fp2 = self.open_output_files(self.ofname2)
        wr2 = self.write_file_hdr(fp2, self.header2)
        
        # Interpolate to obtain values of Jmax and Vcmax at normalising temp    
        # Find the points above and below normalising temperature. 
        # Failing that, find the two points closest to normalising T, and 
        # flag a warning
        for i, id in enumerate(np.unique(data_all["id"])): 
            # For each unique leaf, find the values above and below the 
            # normalising Temperature, note sorting the data!!
            subset = data_all[np.where(data_all["id"] == id)]
            subset.sort(order="Tav") # not all data is in order!!!
            (index, flag25) = self.find_nearest_highest_index(subset["Tav"])
            
            # In the unlikely event we have measured data at exactly 25 degC
            # use it!! Otherwise interpolate...
            if flag25:
                (jnorm, vnorm) = subset["Jmax"][index], subset["Vcmax"][index]
            elif index == 0:
                print "Missing value below Tnorm, so interpolating with"
                print "data points above - ID: %s" % (id)
                print
                vnorm = self.interpolate_temp(subset, index+1, index, "Vcmax")
                jnorm = self.interpolate_temp(subset, index+1, index, "Jmax")
            else:
                vnorm = self.interpolate_temp(subset, index, index-1, "Vcmax")
                jnorm = self.interpolate_temp(subset, index, index-1, "Jmax")
                
            self.write_outputs(jnorm, vnorm, subset, id, wr1, wr2)
        fp1.close()
        fp2.close()
        
        # Read normalised values back in to make some plot 
        self.make_plots()
            
         
    def read_data(self, fname, delimiter=","):
        """ Read the fitted data into an array 
        Expects a format of:
        -> Jmax,JSE,Vcmax,VSE,Rd,RSE,Tav,R2,n,Species,Season,Leaf,Curve,
           Filename
        """
        data = np.recfromcsv(fname, delimiter=delimiter, names=True, 
                             case_sensitive=True)
        return data
    
    def open_output_files(self, ofname):
        if os.path.isfile(ofname):
            os.remove(ofname)
            
        try:
            fp = open(ofname, 'wb')
        except IOError:
            raise IOError("Can't open %s file for write" % ofname)     
        
        return fp
    
    def write_file_hdr(self, fname, header):  
        wr = csv.writer(fname, delimiter=',', quoting=csv.QUOTE_NONE, 
                        escapechar=' ')
        wr.writerow(header)
        
        return wr
    
    def interpolate_temp(self, data, index1, index2, var):
        """ 
        Interpolate to obtain values of Jmax and Vcmax at normalising temp
        """ 
        # get relevant data, build dict for consistancy so we can use Tarrh func
        d = {}
        d["Tav"] = data["Tav"][index1] 
        x1 = self.calc_Tarrh(d)
        d["Tav"] = data["Tav"][index2]
        x2 = self.calc_Tarrh(d)  
        y1 = np.log(data[var][index1])
        y2 = np.log(data[var][index2])
        
        return np.exp(y1 - x1 * (y2 - y1) / (x2 - x1))  
        
    def find_nearest_highest_index(self, array, flag25=False):
        """ find nearest index in an array that is > a given value """
        index = (np.abs(array - self.tnorm)).argmin()
        # Check to see if in the unlikely event we measured data at exactly
        # 25 degC
        if np.abs(array[index] - self.tnorm) < 1E-08:
            flag25 = True
        # We always want the value > tnorm.
        elif array[index] < self.tnorm:
            index += 1
        
        return index, flag25
    
    def write_outputs(self, jnorm, vnorm, subset, leaf, fp1, fp2):
        """ Print out values at normalising temperature """
        fp1.writerow([jnorm, vnorm, subset["Species"][0], leaf,
                   subset["Filename"][0]])
        
        # Normalise each point by value at the normalising temperature
        vcmax_norm = np.log(subset["Vcmax"] / vnorm)
        jmax_norm = np.log(subset["Jmax"] / jnorm)
        Tarrh = self.calc_Tarrh(subset)
        for j in xrange(len(subset)):          
            row = [subset["Jmax"][j], subset["Vcmax"][j],\
                   jmax_norm[j], vcmax_norm[j], \
                   subset["Rd"][j], subset["Tav"][j], Tarrh[j], \
                   subset["R2"][j], subset["n"][j], \
                   subset["Species"][j], subset["Leaf"][j], \
                   subset["Curve"][j], \
                   subset["Filename"][j]]
            fp2.writerow(row)
    
    def make_plots(self):
        colour_list=['red', 'blue', 'green', 'yellow', 'orange', 'blueviolet',\
                     'darkmagenta', 'cyan', 'indigo', 'palegreen', 'salmon',\
                     'pink', 'darkgreen', 'darkblue',\
                     'red', 'blue', 'green', 'yellow', 'orange', 'blueviolet',\
                     'darkmagenta', 'cyan', 'indigo', 'palegreen', 'salmon',\
                     'pink', 'darkgreen', 'darkblue', 'red', 'blue', 'green']
        
        # Read normalised values back in to make some plot 
        data_all = self.read_data(self.ofname2)
        
        # Do plots by Species
        # Plot Jmax vs T   
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for i, spp in enumerate(np.unique(data_all["Species"])):
            prov = data_all[np.where(data_all["Species"]==spp)]
            ax1.plot(prov["Tav"],prov["Jmax"], ls="", lw=1.5, marker="o", 
                     c=colour_list[i], label = spp)
            ax1.set_xlabel("Temperature")
            ax1.set_ylabel("Jmax")
            ax1.legend(numpoints=1, loc='best', shadow=False).draw_frame(True)
        fig.savefig(os.path.join(self.plot_dir, "JmaxvsT.png"), dpi=100)
        
        #Plot normalised Jmax in Arrhenius plot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for i, spp in enumerate(np.unique(data_all["Species"])):
            prov = data_all[np.where(data_all["Species"]==spp)]     
            ax1.plot(self.calc_Tarrh(prov), prov["Jnorm"], ls="", lw=1.5, 
                     marker="o", c=colour_list[i], label=spp)
            ax1.set_xlabel("1/Tk - 1/298")
            ax1.set_ylabel("Normalised Jmax")
            ax1.legend(numpoints=1, loc='best', shadow=False).draw_frame(True)
        fig.savefig(os.path.join(self.plot_dir, "JArrh.png"), dpi=100)
        
        # Plot Vcmax vs T             
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for i, spp in enumerate(np.unique(data_all["Species"])):
            prov = data_all[np.where(data_all["Species"]==spp)]         
            
            ax1.plot(prov["Tav"], prov["Vcmax"],
                     ls="", lw=1.5, marker="o", c=colour_list[i], label = spp)
            ax1.set_xlabel("Temperature")
            ax1.set_ylabel("Vcmax")
            ax1.legend(numpoints=1, loc='best', shadow=False).draw_frame(True)
        fig.savefig(os.path.join(self.plot_dir, "vcmaxvsT.png"), dpi=100)
        
        #Plot normalised Vmax in Arrhenius plot
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        for i, spp in enumerate(np.unique(data_all["Species"])):
            prov = data_all[np.where(data_all["Species"]==spp)]    
            
            ax1.plot(self.calc_Tarrh(prov), prov["Vnorm"], ls="", lw=1.5, 
                     marker="o", c=colour_list[i], label = spp)
            ax1.set_xlabel("1/Tk - 1/298")
            ax1.set_ylabel("Normalised Vcmax")
            ax1.legend(numpoints=1, loc='best', shadow=False).draw_frame(True)
        fig.savefig(os.path.join(self.plot_dir, "VArrh.png"), dpi=100)

    def calc_Tarrh(self, data):
        arg1 = 1.0 / (self.tnorm + self.deg2kelvin)
        arg2 = 1.0 / (data["Tav"] + self.deg2kelvin)
         
        return arg1 - arg2
    