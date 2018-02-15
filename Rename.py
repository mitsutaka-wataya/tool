# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 12:20:13 2017

@author: test
"""
import pandas as pd
import os
import glob

def rename(v):
    fname=glob.glob("*.tif")
    v = v.values.flatten()
    for i in range(len(fname)):
        new=fname[i].split(".")[0]
        new=new+"_%02dV.tif" %v[i]
        os.rename(fname[i],new)
    os.chdir("..")
    os.chdir("..")
    
def get_vol():
    d=glob.glob("*vol*.csv")[0]
    vol = pd.read_csv(d,index_col=0)
    return(vol)

def main(directory):
    os.chdir(directory)
    vol = get_vol()
    os.chdir("Stream")
    rename(vol)
    