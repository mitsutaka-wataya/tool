# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:29:34 2017

@author: test
make random list csv

"""
import random 
import pandas as pd

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
    return(pd.DataFrame(result1),pd.DataFrame(result2))
    

if __name__ == "__main__":
    x=[0,5,10,15,20,25,30,35,40,50]
    rep_n = 10
    
    df_sub,sub4exp = make_random_list(x,rep_n)
    df_all,all4exp = make_allrandom_list(x,rep_n)
    
    df_all.to_excel("input_rondamize_100_does_list.xlsx")
    all4exp.T.to_excel("cheatsheet_rondamize_100_does.xlsx")
    #df_sub.to_excel("input_sub_rand_list.xlsx")
    #sub4exp.T.to_excel("cheatsheet_sub.xlsx")