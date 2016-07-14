#!/usr/bin/env python

"""
Fit Jmax, Vcmax, Rd for each dataset using MCMC. This is pared down version,
with none of the bells of whistles of the main fitting package. It will only
work off a single A-Ci curve file for now.

That's all folks.
"""

__author__ = "Martin De Kauwe"
__version__ = "1.0 (14.07.2016)"
__email__ = "mdekauwe@gmail.com"

import os
import sys
import glob
import pymc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fit_farquhar_model.farquhar_model import FarquharC3
from fit_farquhar_model.fit_model import FitJmaxVcmaxRd
from fit_farquhar_model.fit_model import FitJmaxVcmaxKnownRdark

deg2kelvin = 273.15
ofname = "fitting_results_%s.csv" % ("Mercado")
results_dir = "results"
data_dir = "data"
plot_dir = "plots"

df = pd.read_csv("data/A_CI_Rdark_4Martin.csv")
df["Tleaf"] += deg2kelvin
obs = df["Photo"]
obs_sigma = obs * 0.05 # assume near perfect obs

# V.broad priors for now
Vcmax = pymc.Uniform("Vcmax", 5.0, 200, value=20.0)
Jmax = pymc.Uniform("Jmax", 5.0, 400, value=40.0)
Rd = pymc.Uniform("Rd", 0.0, 3.0, value=0.5)

F = FarquharC3(peaked_Jmax=True, peaked_Vcmax=True)

@pymc.deterministic
def farquhar_wrapper(df=df, Vcmax=Vcmax, Jmax=Jmax, Rd=Rd):
    (An, Anc, Anj) = F.calc_photosynthesis(Ci=df["Ci"], Tleaf=df["Tleaf"],
                                           Jmax=Jmax, Vcmax=Vcmax, Rd=Rd)
    return An


y = pymc.Normal('y', mu=farquhar_wrapper, tau=1.0/obs_sigma**2,
                value=obs, observed=True)

N = 100000
model = pymc.Model([y, df, Vcmax, Jmax, Rd])
M = pymc.MCMC(model)
M.sample(iter=N, burn=N*0.1, thin=10)

Vcmax = M.stats()["Vcmax"]['mean']
Jmax = M.stats()["Jmax"]['mean']
Rd = M.stats()["Rd"]['mean']
(An, Anc, Anj) = F.calc_photosynthesis(Ci=df["Ci"], Tleaf=df["Tleaf"],
                                       Jmax=Jmax, Vcmax=Vcmax, Rd=Rd)
rmse = np.sqrt(((obs - An)**2).mean(0))
print "RMSE: %.4f" % (rmse)

# Get the fits
Vcmax = M.trace('Vcmax').gettrace()
Jmax = M.trace('Jmax').gettrace()
Rd = M.trace('Rd').gettrace()
for v in ["Vcmax", "Jmax", "Rd"]:
    print "%s: %.4f +/- %.4f" % \
        (v, M.stats()[v]['mean'], M.stats()[v]['standard deviation'])

fig = plt.figure(figsize=(10,10))
fig.subplots_adjust(hspace=0.5)

ax1 = fig.add_subplot(231)
ax1.set_title(r"Trace of Vcmax")
ax1.plot(Vcmax, color="#467821", label="Vcmax")
#ax1.legend(loc="best", numpoints=1)

ax2 = fig.add_subplot(232)
ax2.set_title(r"Trace of Jmax")
ax2.plot(Jmax, color="#467821", label="Jmax")
#ax2.legend(loc="best", numpoints=1)

ax3 = fig.add_subplot(233)
ax3.set_title(r"Trace of Rd")
ax3.plot(Rd, color="#467821", label="Rd")
#ax3.legend(loc="best", numpoints=1)

ax4 = fig.add_subplot(234)
ax4.set_title(r"Posterior dist. of Vcmax")
ax4.hist(Vcmax, bins=50, histtype='stepfilled', alpha=0.85, color="#467821",
         normed=False)
conf_lower, conf_upper = M.stats()['Vcmax']['95% HPD interval']
ax4.axvline(conf_lower, linewidth=2, color='grey', linestyle='dotted')
ax4.axvline(conf_upper, linewidth=2, color='grey', linestyle='dotted')
ax4.set_ylabel("Frequency")

ax5 = fig.add_subplot(235)
ax5.set_title(r"Posterior dist. of Jmax")
ax5.hist(Jmax, bins=50, histtype='stepfilled', alpha=0.85, color="#467821",
         normed=False)
conf_lower, conf_upper = M.stats()['Jmax']['95% HPD interval']
ax5.axvline(conf_lower, linewidth=2, color='grey', linestyle='dotted')
ax5.axvline(conf_upper, linewidth=2, color='grey', linestyle='dotted')

ax6 = fig.add_subplot(236)
ax6.set_title(r"Posterior dist. of Rd")
ax6.hist(Rd, bins=50, histtype='stepfilled', alpha=0.85, color="#467821",
         normed=False)
conf_lower, conf_upper = M.stats()['Rd']['95% HPD interval']
ax6.axvline(conf_lower, linewidth=2, color='grey', linestyle='dotted')
ax6.axvline(conf_upper, linewidth=2, color='grey', linestyle='dotted')

fig.savefig("model_fits.png", dpi=150)
