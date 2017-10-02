# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:29:34 2017

@author: test
make random list csv

"""
import random 
import pandas as pd
import glob
import os

def make_allrandom_list(Voltage,repeat=20):
    Vol_rep = Voltage*repeat
    rep_n = len(Vol_rep)
    result1 = random.sample(Vol_rep,rep_n)
    print("repeat_number:"+str(rep_n))
    result2 = [result1[i*len(Voltage) : i*len(Voltage)+len(Voltage)] for i in range(repeat)]
    print(result2)
    return(pd.DataFrame(result1),pd.DataFrame(result2))
    
def make_random_list(Voltage,repeat=20):
    l = len(Voltage)
    Voltage_rep = [random.sample(Voltage,l) for i in range(20)]
    result1 = []
    for i in Voltage_rep:
        result1 += i
    print(result1)
    result2 = [result1[i*len(Voltage) : i*len(Voltage)+len(Voltage)] for i in range(repeat)]
    result2 = pd.DataFrame(result2)
    
    return(pd.DataFrame(result1),result2.T)
    

def save_list(v_list,cheatsheet,name="rand_vol",d=None):
    if d == None:
        v_list.to_csv(name+".csv")
        cheatsheet.to_excel("sheet_"+name+".xlsx")
    else :
        try:os.mkdir(d)
        except:print("dir has existed")
        v_list.to_csv(d+"/"+name+".csv")
        cheatsheet.to_excel(d+"/sheet_"+name+".xlsx")

def make_vol_lists(Voltage,repeat=20,fnum=100,d="Vol_Lists"):
    for i in range(fnum):
        num="{0:04d}".format(i+1)
        a,b = make_random_list(Voltage,repeat)
        b.index=[str(i+1)+"-"+str(r) for r in list(b.index)]
        save_list(a,b,name=num+"rand_vol",d=d)
    
if __name__ == "__main__":
    x=[0,5,10,15,20,25,30,35,40,50]
    rep_n = 20
    
    df_sub,sub4exp = make_random_list(x,rep_n)
    df_all,all4exp = make_allrandom_list(x,rep_n)
    
    df_all.to_excel("input_rondamize_100_does_list.xlsx")
    all4exp.T.to_excel("cheatsheet_rondamize_100_does.xlsx")
    #df_sub.to_excel("input_sub_rand_list.xlsx")
    #sub4exp.T.to_excel("cheatsheet_sub.xlsx")