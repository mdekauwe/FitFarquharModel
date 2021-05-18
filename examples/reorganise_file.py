#!/usr/bin/env python

import csv
import numpy as np
import os
import glob
import pandas as pd

import xlrd
import sys

def assign_repeat_number(header_row, in_file, tmp_file):

    df = pd.read_csv(in_file)
    f = open(tmp_file, "w")
    write = csv.writer(f, delimiter=',', quotechar='|',
                       quoting=csv.QUOTE_MINIMAL)
    write.writerow(header_row)

    temp_row = {'Curve': None, 'CO2S': None} # used to compare with first row
    repeat_num = 0
    curve_num = 1
    for index, row in df.iterrows():
        Species = row['Species']
        Photo = row['Photo']
        Ci = row['Ci']
        Tleaf = row['Tleaf']
        VPD = row["VPD"]
        Site = row['Location']
        Trmmol = row['Trmmol']
        Datacontrib = row['Datacontrib']
        Cond = row['Cond']
        CO2S = row['CO2S']
        PARin = row['PARin']
        Press = row['Patm']
        Season = "whatever" # could be anything
        Leaf = 1
        fitgroup = row['Location']

        # Need to identify unique curves, they've used a bogus low PAR to
        # seperate
        if PARin > 100:
            row_list = [curve_num, Tleaf, Ci, Photo, Species, \
                        Site, Cond, Trmmol, VPD, CO2S, \
                        PARin, Press, Season, Datacontrib, Leaf, fitgroup]

        else:
            curve_num += 1



        write.writerow(row_list)
    f.close()


def get_avg_reading(header_row, tmp_file, out_file):

    #########################################################
    ## PART II: take mean value based on the repeat number ##
    #########################################################

    # read in the csv file with unique repeat number
    df = pd.read_csv(tmp_file)

    f2 = open(out_file, "w") # the final file that we want to use
    write2 = csv.writer(f2, delimiter=',', quotechar='|',
                        quoting=csv.QUOTE_MINIMAL)
    write2.writerow(header_row)

    skip_next = False
    for i in range(len(df)-1):

        # Get current data
        Curve = df['Curve'][i]
        Leaf = df['Leaf'][i]
        fitgroup = df['fitgroup'][i]
        Species = df['Species'][i]
        Photo = df['Photo'][i]
        Ci = df['Ci'][i]
        Tleaf = df['Tleaf'][i]
        VPD = df["VPD"][i]
        Site = df['Site'][i]
        Trmmol = df['Trmmol'][i]
        Datacontrib = df['Datacontrib'][i]
        Cond = df['Cond'][i]
        CO2S = df['CO2S'][i]
        PARin = df['PARin'][i]
        Press = df['Press'][i]
        Season = df['Season'][i]

        # Get next data to check against for repeats
        CO2S_next = df['CO2S'][i+1]
        Curve_next = df['Curve'][i+1]
        Species_next = df['Species'][i]

        if skip_next == False:
            # Condition for assigning repeat number: if Ci at current row is +/- 8%
            # of the previous row
            if (Curve == Curve_next and CO2S_next / CO2S > 0.92 and
                CO2S_next / CO2S < 1.08 and Species == Species_next):

                Photo = (df['Photo'][i] + df['Photo'][i+1]) / 2.
                Ci = (df['Ci'][i] + df['Ci'][i+1]) / 2.
                Tleaf = (df['Tleaf'][i] + df['Tleaf'][i+1]) / 2.
                VPD = (df["VPD"][i] + df['VPD'][i+1]) / 2.
                Trmmol = (df['Trmmol'][i] + df['Trmmol'][i+1]) / 2.
                Cond = (df['Cond'][i] + df['Cond'][i+1]) / 2.
                CO2S = (df['CO2S'][i] + df['CO2S'][i+1]) / 2.
                PARin = (df['PARin'][i] + df['PARin'][i+1]) / 2.
                Press = (df['Press'][i] + df['Press'][i+1]) / 2.

                skip_next = True

                #print(Curve, Curve_next, CO2S, CO2S_next)

            row_list = [Curve, Tleaf, Ci, Photo, Species, \
                        Site, Cond, Trmmol, VPD, CO2S, \
                        PARin, Press, Season, Datacontrib, Leaf, fitgroup]

            write2.writerow(row_list)
        else:
            skip_next = False
    f2.close()
    if os.path.exists(tmp_file):
        os.remove(tmp_file)


header_row = ['Curve', 'Tleaf', 'Ci', 'Photo', 'Species',
              'Site', 'Cond', 'Trmmol', 'VPD', 'CO2S',
              'PARin', 'Press', 'Season', 'Datacontrib', 'Leaf', 'fitgroup']
in_file = "raw_data/WUEdatabase_Martin-StPaul_FPB_2012_OK2.csv"
tmp_file = "data/Martin-StPaul_temp.csv"
out_file = "data/Martin-StPaul_cleaned.csv"
assign_repeat_number(header_row, in_file, tmp_file)
get_avg_reading(header_row, tmp_file, out_file)
