# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:08:40 2016

@author: waizoo
"""
import pandas as pd
import numpy as np



def make_rate_df(raw_df_list , basal=None):
    dataframe = [i.copy() for i in raw_df_list]
    """    
    if basal == None:    
        time0 = [df.iloc[1,:].values.flatten() for df in dataframe]
        t0=time0[0]
    else:
    """
    
    t0 = basal.iloc[0,:].values.flatten() 
    dataframe = [df / t0 for df in dataframe]
    return(dataframe)

def make_diff_df(raw_df_list,basal=None):
    dataframe = [i.copy() for i in raw_df_list]
    """    
    if basal == None:
        time0 = [df.iloc[0,:].values.flatten() for df in dataframe]
        t0=time0[0]
    else:
    """        
    t0 = basal.iloc[0,:].values.flatten()
    dataframe = [df - t0 for df in dataframe]
    return (dataframe)

def make_ind_rate_df(raw_df_list):
    dataframe = [i.copy() for i in raw_df_list]
    t0_list = [df.iloc[0:5,:].mean().values for df in dataframe]
    dataframe = [df / np.abs(t0) for df,t0 in zip(dataframe,t0_list)]
    return(dataframe)

def make_ind_diff_df(raw_df_list):
    dataframe = [i.copy() for i in raw_df_list]
    t0_list = [df.iloc[0:5,:].mean().values for df in dataframe]
    dataframe = [df - t0 for df,t0 in zip(dataframe,t0_list)]
    return(dataframe)

def make_ind_diffSeries_df(raw_df_list):
    dataframe = [i.copy() for i in raw_df_list]
    shift = [s.shift() for s in dataframe]
    diffSeries = [d-s for d,s in zip(dataframe,shift)]
    return(diffSeries)
def search_peak(df_list):
    peak = [df.max() for df in df_list]
    peak_df = pd.DataFrame(peak)
    return(peak_df)
    #print(peak)b

def get_fbasal(df_list,stim_time=5):
    basal = [df.iloc[:stim_time,:].mean() for df in df_list ]
    basal_df = pd.DataFrame(basal)
    return(basal_df)

def get_fbasal_std(df_list,cal_frame=5):
    basal = [df.iloc[:,:].std() for df in df_list ]
    basal_df = pd.DataFrame(basal)
    return(basal_df)

def get_bbasal(df_list,cal_frame=5):
    basal = [df.iloc[:-1*cal_frame,:].mean() for df in df_list ]
    basal_df = pd.DataFrame(basal)
    return(basal_df)

def get_auc(df_list,base,time_per_frame=70):
    integral = [df.sum()*time_per_frame for df in df_list]
    integral = pd.concat(integral,axis=1).T
    auc = integral
    return(auc)

def get_modify_auc(df_list,base,time_per_frame=70):
    frame_num = len(df_list[0].iloc[:,0])
    integral = [df.sum()*time_per_frame for df in df_list]
    integral = pd.concat(integral,axis=1).T
    base_integral = base*frame_num*time_per_frame
    auc = integral - base_integral
    return(auc)
    
    