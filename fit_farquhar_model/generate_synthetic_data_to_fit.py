import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from fit_farquhar_model.farquhar_model import FarquharC3
fname = "/Users/mdekauwe/Google_Drive/ACI_datasets/data/Way/Picea_mariana.csv"
df = pd.read_csv(fname, sep=",", header=0)
df = df[df["fitgroup"]=="Cool"]
df["Tleaf"] += 273.15


Vcmax25_1 = 14.98038380
Vcmax25_2 = 21.53558136
Vcmax25_3 = 18.38515647
Vcmax25_4 = 27.13037479
Vcmax25_5 = 19.84849665 
Vcmax25_6 = 12.87992473
Vvals = [Vcmax25_1,Vcmax25_2,Vcmax25_3,Vcmax25_4,Vcmax25_5,Vcmax25_6]

Rdfac = 0.015
Jfac = 1.8
Eaj = 28222.03727894
delSj = 638.82241123
Eav = 42900.87707605
delSv = 623.15655819
Ear = 34000.0
Hdv = 200000.0
Hdj = 200000.0

Anx = np.zeros(0)
for curve_num in np.unique(df["Curve"]):
    curve_df = df[df["Curve"]==curve_num]
    curve_df = curve_df.sort(['Ci'], ascending=True)
    curve_df.index = range(len(curve_df)) # need to reindex slice
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
    Anx = np.append(Anx, An)
   
df["Photo"] = Anx
df.to_csv('/Users/mdekauwe/Desktop/WAY_TEST/data/synthetic_data.csv', index_label="Index")
     
                       





