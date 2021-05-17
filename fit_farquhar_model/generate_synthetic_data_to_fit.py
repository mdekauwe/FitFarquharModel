import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from fit_farquhar_model.farquhar_model import FarquharC3
fname = "/Users/mdekauwe/Google_Drive/ACI_datasets/data/Way/Picea_mariana.csv"
#fname = "/Users/mdekauwe/Desktop/Picea_mariana.csv"
df = pd.read_csv(fname, sep=",", header=0)
df = df[df["fitgroup"]=="Cool"]
df["Tleaf"] += 273.15
df.index = range(len(df)) # need to reindex slice
#df.loc[:,'Tleaf'] += 273.15


Eaj = 27535.63574182
Ear = 21053.28192588
Eav = 44013.15555911
Jfac =  1.47680402
Rdfac = 0.04542771
Vcmax25_1 = 19.57163149
Vcmax25_2 = 24.76596270
Vcmax25_3 = 23.28029167
Vcmax25_4 = 33.51815661
Vcmax25_5 = 28.37893914
Vcmax25_6 = 17.31574023
delSj = 643.39267144
delSv = 640.09326969

Hdv = 200000.0
Hdj = 200000.0
Vvals = [Vcmax25_1,Vcmax25_2,Vcmax25_3,Vcmax25_4,Vcmax25_5,Vcmax25_6]

add_noise = False

Anx = np.zeros(0)
Cix = np.zeros(0)
#f = open("../examples/data_pymc/synthetic_data.csv", "w")
f = open("/Users/mdekauwe/Desktop/synthetic_data.csv", "w")
print("Species,Leaf,Curve,Photo,Ci,Tleaf,Season,fitgroup", file=f)
for curve_num in np.unique(df["Curve"]):

    curve_df = df[df["Curve"]==curve_num]
    curve_df = curve_df.sort_values(['Ci'], ascending=True)
    curve_df.index = range(len(curve_df)) # need to reindex slice

    #print  curve_df["Leaf"][0]
    leaf_index = curve_df["Leaf"][0] - 1

    Vcmax25 = Vvals[leaf_index]
    Jmax25 = Vvals[leaf_index] * Jfac
    Rd25 = Vvals[leaf_index] * Rdfac

    model = FarquharC3(peaked_Jmax=True, peaked_Vcmax=True, model_Q10=False)
    (An, Anc, Anj) = model.calc_photosynthesis(Ci=curve_df["Ci"], Tleaf=curve_df["Tleaf"],
                           Par=None, Jmax=None, Vcmax=None,
                           Jmax25=Jmax25, Vcmax25=Vcmax25, Rd=None,
                           Q10=None, Eaj=Eaj, Eav=Eav,
                           deltaSj=delSj, deltaSv=delSv, Rd25=Rd25,
                          Ear=Ear, Hdv=Hdv, Hdj=Hdj)

    from scipy.stats import truncnorm
    for i, A in enumerate(An):

        if add_noise:
            #A = truncnorm.rvs(a=0., b=20.0, loc=A, scale=3.0)
            #print A,
            A += np.random.normal(0.0, 1.0)
            print("adding noise")
        else:
            noise = 0.0
        print("%s,%d,%d,%f,%f,%f,%s,%s"  % (curve_df["Species"][i], curve_df["Leaf"][i],\
                                            curve_df["Curve"][i], A, \
                                            curve_df["Ci"][i], curve_df["Tleaf"][i]-273.15,\
                                            curve_df["Season"][i], curve_df["fitgroup"][i]), file=f)
    #Anx = np.append(Anx, An)
    #for c in curve_df["Ci"].values:
    #    Cix = np.append(Cix, c)


#df["Photo"] = Anx
#df["Ci"] = Cix
#df.to_csv('../examples/data_pymc/synthetic_data.csv', index_label="Index")
