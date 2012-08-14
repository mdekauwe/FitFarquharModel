#!/usr/bin/env python

"""
Normalise the data...write something sensible

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (14.08.2012)"
__email__ = "mdekauwe@gmail.com"

import sys
import numpy as np
import os
import glob
import matplotlib.mlab as mlab  # library to write structured array to file.
import matplotlib.pyplot as plt


class Normalise(object):
    def __init__(self, fname=None, results_dir=None, plot_dir=None):
        self.results_dir = results_dir
        self.fname = os.path.join(self.results_dir, ofname)
        self.plot_dir = plot_dir    
        self.tnorm = 25.0 # Temperature we are normalising to
        self.deg2kelvin = 273.15
        
    def main(self):
    
    

fname = "results/fitting_results.csv"
data_all = np.recfromcsv(fname, delimiter=",", names=True, case_sensitive=True)
# Files to write to
f1 = open("results/values_at_Tnorm.txt", "w")
f1.write("Jmax Vcmax Species Leaf Filename \n")
f2 = open("results/normalised_results.txt", "w")
f2.write("Jmax Vcmax Jnorm Vnorm Rd Tav Tarrh R2 n Species Leaf Curve Filename \n")

vcmax_norm = np.ones(len(data_all))*-9999.
jmax_norm = np.ones(len(data_all))*-9999.
Tarrh = np.ones(len(data_all))*-9999.

# Find the points above and below normalising temperature. 
# Failing that, find the two points closest to normalising T, and flag a warning

for i,leaf in enumerate(np.unique(data_all["Leaf"])): # Find unique leaves
    # For each unique leaf, find the values above and below the normalising Temperature
    subset = data_all[np.where(data_all["Leaf"]==leaf)]
    diff1 = 100.0
    diff2 = -100.0
    index1 = 0
    index2 = 0
    for j in xrange(len(subset)) :
        diffj = subset["Tav"][j] - Tnorm
        if ((diffj>0.0) & (diffj < diff1)) :
            diff1 = diffj
            index1 = j
        elif ((diffj < 0.0)&(diff2 < diffj)) :
            diff2 = diffj
            index2 = j
        print leaf, j, subset["Tav"][j], diff1, diff2, index1, index2        
    
    if ((diff1 > 10.) | (diff2 > 10.)) :
        print "Missing value", i
        diff1 = 99.0
        diff2 = 100.0
        index1 = 0
        index2 = 0        
        for k in xrange(len(subset)) :
            diffj = np.fabs(subset["Tav"][k] - Tnorm)
            if (diffj < diff1) :
                diff2 = diff1
                diff1 = diffj
                index2 = index1
                index1 = k
            elif (diffj < diff2) :
                diff2 = diffj
                index2 = k            
            print leaf, k, subset["Tav"][k], diff1, diff2, index1, index2        
        if ((diff1 > 10.) | (diff2> 10.)) :
            print "Cannot normalise", i
    
    # Interpolate to obtain values of Jmax and Vcmax at normalising temp    
    x1 = 1./(Tnorm+deg2kelvin) - 1./(subset["Tav"][index1]+deg2kelvin)  
    x2 = 1./(Tnorm+deg2kelvin) - 1./(subset["Tav"][index2]+deg2kelvin)  
    v1 = np.log(subset["Vcmax"][index1])
    v2 = np.log(subset["Vcmax"][index2])
    vnorm = np.exp(v1 - x1 * (v2 - v1)/(x2 - x1))  
    #print x1,x2,v1,v2,vnorm
    j1 = np.log(subset["Jmax"][index1])
    j2 = np.log(subset["Jmax"][index2])
    jnorm = np.exp(j1 - x1 * (j2 - j1)/(x2 - x1))  
    
    # Print out values at normalising temperature
    f1.write("%f %f %s %d %s \n" % \
            (jnorm,vnorm,subset["Species"][0],leaf,subset["Filename"][0]))
    
    # Normalise each point by value at the normalising temperature
    for j in xrange(len(subset)) :          
        vcmax_norm[j] = np.log(subset["Vcmax"][j]/ vnorm)
        jmax_norm[j] = np.log(subset["Jmax"][j]/ jnorm)
        Tarrh[j] = 1./(Tnorm+deg2kelvin) - 1./(subset["Tav"][j]+deg2kelvin)
        #   print i,j subset["Vcmax"][j], vcmax25, vcmax_norm[j]
              
        f2.write("%f %f %f %f %f %f %f %f %d %s %d %d %s \n" % \
            (subset["Jmax"][j],subset["Vcmax"][j], \
            jmax_norm[j],vcmax_norm[j], \
            subset["Rd"][j],subset["Tav"][j],Tarrh[j], \
            subset["R2"][j],subset["n"][j], \
            subset["Species"][j],subset["Leaf"][j], \
            subset["Curve"][j], \
            subset["Filename"][j]))

f1.close()
f2.close()

# Read normalised values back in to plot
fname = "results/normalised_results.txt"
data_all = np.recfromcsv(fname, delimiter=" ", names=True, case_sensitive=True)

colour_list=['red', 'blue', 'green', 'yellow', 'orange', 'blueviolet',\
             'darkmagenta', 'cyan', 'indigo', 'palegreen', 'salmon',\
             'pink', 'darkgreen', 'darkblue',\
             'red', 'blue', 'green', 'yellow', 'orange', 'blueviolet',\
             'darkmagenta', 'cyan', 'indigo', 'palegreen', 'salmon',\
             'pink', 'darkgreen', 'darkblue', 'red', 'blue', 'green']


#Do plots by Species
# Plot Jmax vs T             
fig = plt.figure()
ax1 = fig.add_subplot(111)
for i,spp in enumerate(np.unique(data_all["Species"])):
    prov = data_all[np.where(data_all["Species"]==spp)]
    ax1.plot(prov["Tav"],prov["Jmax"],
        ls="", lw=1.5, marker="o", c=colour_list[i], label = spp)
ax1.set_xlabel("Temperature")
ax1.set_ylabel("Jmax")
ax1.legend(numpoints=1, loc='best', shadow=False).draw_frame(True)
fig.savefig("plots/JmaxvsT.png", dpi=100)

#Plot normalised Jmax in Arrhenius plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
for i,spp in enumerate(np.unique(data_all["Species"])):
    prov = data_all[np.where(data_all["Species"]==spp)]
    ax1.plot(-1./(prov["Tav"]+deg2kelvin) + 1/(Tnorm+deg2kelvin),prov["Jnorm"],
        ls="", lw=1.5, marker="o", c=colour_list[i], label = spp)
ax1.set_xlabel("1/Tk - 1/298")
ax1.set_ylabel("Normalised Jmax")
ax1.legend(numpoints=1, loc='best', shadow=False).draw_frame(True)
fig.savefig("plots/JArrh.png", dpi=100)
  
# Plot Vcmax vs T             
fig = plt.figure()
ax1 = fig.add_subplot(111)
for i,spp in enumerate(np.unique(data_all["Species"])):
    prov = data_all[np.where(data_all["Species"]==spp)]
    ax1.plot(prov["Tav"],prov["Vcmax"],
        ls="", lw=1.5, marker="o", c=colour_list[i], label = spp)
ax1.set_xlabel("Temperature")
ax1.set_ylabel("Vcmax")
ax1.legend(numpoints=1, loc='best', shadow=False).draw_frame(True)
fig.savefig("plots/vcmaxvsT.png", dpi=100)

#Plot normalised Vmax in Arrhenius plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
for i,spp in enumerate(np.unique(data_all["Species"])):
    prov = data_all[np.where(data_all["Species"]==spp)]
    ax1.plot(-1./(prov["Tav"]+deg2kelvin) + 1/(Tnorm+deg2kelvin),prov["Vnorm"],
        ls="", lw=1.5, marker="o", c=colour_list[i], label = spp)
ax1.set_xlabel("1/Tk - 1/298")
ax1.set_ylabel("Normalised Vcmax")
ax1.legend(numpoints=1, loc='best', shadow=False).draw_frame(True)
fig.savefig("plots/VArrh.png", dpi=100)  
               
 

