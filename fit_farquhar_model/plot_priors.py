import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm as tn

import pymc
#mu = 25.0
#sigma = 11.25
#a = 1.0
#b = 650.0
#vals = tn(a=a, b=b, loc=mu, scale=sigma)
#plt.hist(vals.rvs(100000), bins=50)
#plt.xlim(0, 100)
#plt.show()


N = 10000
Vcmax = [pymc.TruncatedNormal('Vcmax25', \
          mu=25.0, tau=1.0/11.25**2, a=0.0, b=650.0).value \
          for i in xrange(N)]

Jfac = [pymc.TruncatedNormal('Jfac', mu=1.8, tau=1.0/0.5**2, \
        a=0.0, b=5.0).value for i in xrange(N)]
        
        
Rdfac = [pymc.Uniform('Rdfac', lower=0.005, upper=0.05).value \
         for i in xrange(N)]
        
Eaj = [pymc.TruncatedNormal('Eaj', mu=40000.0, tau=1.0/10000.0**2, a=0.0, 
       b=199999.9).value for i in xrange(N)]
        
Eav = [pymc.TruncatedNormal('Eav', mu=60000.0, tau=1.0/10000.0**2, a=0.0, 
       b=199999.9).value for i in xrange(N)]
        

Ear = [pymc.TruncatedNormal('Ear', mu=34000.0, tau=1.0/10000.0**2, a=0.0, 
       b=199999.9).value for i in xrange(N)]
       
           
delSj = [pymc.TruncatedNormal('delSj', mu=640.0, tau=1.0/10.0**2, a=300.0,
         b=800.0).value for i in xrange(N)]
        
delSv = [pymc.TruncatedNormal('delSv', mu=640.0, tau=1.0/10.0**2, 
         a=300.0, b=800.0).value for i in xrange(N)]          

plt.rcParams['figure.subplot.hspace'] = 0.3
plt.rcParams['figure.subplot.wspace'] = 0.3
plt.rcParams['font.size'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10.0
plt.rcParams['ytick.labelsize'] = 10.0
plt.rcParams['axes.labelsize'] = 10.0


var_names = ["Vcmax", "Jfac", "Rdfac", "Eaj", "Eav", "Ear", "delSj", "delSv"]
vars = [Vcmax, Jfac, Rdfac, Eaj, Eav, Ear, delSj, delSv]


fig = plt.figure(figsize=(10,10))
bins = 50

for index, var in enumerate(var_names):
    
    ax = fig.add_subplot(4,2,(index+1))
    ax.set_title(var_names[index])
    ax.hist(vars[index], bins=bins)

fig.savefig("/Users/mdekauwe/Desktop/priors.png", dpi=150)
plt.show()