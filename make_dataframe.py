# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 00:53:21 2017

@author: waizoo
"""
import pandas as pd

def make_vol_index(df_num,vol_list):
    vol_index_num = int(df_num/len(vol_list))
    vol_index = vol_list*vol_index_num
    mod = df_num%len(vol_list)
    vol_index += vol_list[:mod]
    return(vol_index)

def make_stim_index(df_index_num,stim_point):
    stim_index = []
    for i in range(df_index_num):
        if i <stim_point:
            stim_index += [False]
        elif i >= stim_point:
            stim_index += [True]
    return(stim_index)
    
def reset_index(df_list):
    for i in df_list:
        i.reset_index(drop=True,inplace=True)
    return(df_list)

def add_voltage(df_list,vol_index):
    for i in range(len(df_list)):
        df_list[i]["voltage"] = vol_index[i]
    return(df_list)        

def add_stim(df_list,stim_index):
    for i in range(len(df_list)):
        df_list[i]["stim"] = stim_index
    return(df_list)    

def add_repeat(df_list):
    for i in range(len(df_list)):
        df_list[i]["repeat"] = i
    return(df_list)

def add_time(df_list,duration = 70):
    time_list = [i*duration for i in range(len(df_list[0].index))]
    for i in df_list:
        i["time"] = time_list
    return(df_list)
    
    
def split_intensity_df(df):
    label = df.colmuns
    ID_list = [df[i] for i in label]
    return(ID_list)
    
def make_label_index(df_index_num,label_list):
    label = []    
    for i in label_list:
        label += [[i for j in range(df_index_num)]]
    return(label)
    

def make_dataframe(dfs):
    
    vol_list = [30]
    stim = 15
    duration = 70
    df_variable = ["voltage","stim","repeat","time"]
    df_list = [i.copy() for i in dfs]
    df_num = len(df_list)
    df_index_num = len(df_list[0])
    label_list = df_list[0].columns
    #label_list = label_list[:19]    
    
    vol_index = make_vol_index(df_num,vol_list)
    stim_index = make_stim_index(df_index_num,stim)    
    
    df_list = reset_index(df_list=df_list)
    df_list = add_time(df_list,duration)
    df_list = add_voltage(df_list,vol_index)
    df_list = add_stim(df_list,stim_index)
    df_list = add_repeat(df_list)
    result = []
    for i in df_list:
        labeled_df = [i[[j]+df_variable] for j in label_list]
        k=0
        for i,j in zip(labeled_df,label_list):
            i["ID"] = k
            i.rename(columns={j:"intensity"},inplace=True)
            result += [i]            
            k+=1
    all_result = pd.concat(result)
    
    return(all_result,result)    
