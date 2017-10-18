# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 12:20:13 2017

@author: test
"""
import pandas as pd
import os
import glob

def newname(v):
    fname=glob.glob("*.tif")
    v = v.values.flatten()
    for i in range(200):
        new=fname[i].split(".")[0]
        new=new+"_%02dV.tif" %v[i]
        os.rename(fname[i],new)
    os.chdir("..")
    os.chdir("..")
    