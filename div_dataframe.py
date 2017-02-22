# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 13:59:26 2016
dataの順序を
0V->100V,100V->0Vの順に返します。
@author: waizoo
"""
import pandas as pd
import numpy as np

def get_x_label(exp_type):
    if exp_type == 0:
        print("there are no matching exp_type")
    elif exp_type == 1:    
        x_label = np.array([0,3,10,20,30,50,100])
        legend_label=("1_0V->100V","2_100V->0V")
    elif exp_type == 2:
        x_label = np.array([0,3,10,20,30,50,100])
        legend_label=("1_0V->100V","2_0V->100V","3_0V->100V","4_100V->0V","5_100V->0V","6_100V->0V")
    elif exp_type == 3:
        x_label = np.array([0,3,10,20,30,50,100])
        legend_label=("1_0V->100V","2_0V->100V","3_0V->100V")
    elif exp_type == 4:
        x_label = np.array([0,3,5,10,15,20,25,30,35,40,50,75,100])
        legend_label=("1_0V->100V","2_100V->0V","3_0V->100V","4_100V->0V","5_0V->100V","6_100V->0V","7_0V->100V","8_100V->0V","9_0V->100V","10_100V->0V")
    elif exp_type == 5:
        x_label = np.array([0,3,5,10,15,20,25,30,35,40,50,75,100])
        legend_label=("1_0V->100V","2_100V->0V","3_0V->100V","4_100V->0V")
    elif exp_type == 6:
        x_label = np.array([0,5,10,15,20,25,30,35,40,50,75,100])
        legend_label=("1_0V->100V","2_100V->0V","3_0V->100V","4_100V->0V")   
    elif exp_type == 7:
        x_label = np.array([0,5,10,15,20,25,30,35,40,50,75,100])
        legend_label=("1_0V->100V","2_100V->0V")
    elif exp_type == 8:
        x_label = np.array([0,5,10,15,20,25,30])
        legend_label=("1_0V->30V","2_30V->0V","3_0V->30V","4_30V->0V","5_0V->30V","6_30V->0V","7_0V->30V","8_30V->0V")
        
    return(x_label,legend_label)

def div_label_stream(df,exp_type = 0):
    if exp_type == 0:
        print("there are no matching dataframe   ")
    elif exp_type == 1:
        return (div_label_stream1(df))
    elif exp_type == 2:
        return (div_label_stream2(df))
    elif exp_type == 3:
        return (div_label_stream3(df))
    elif exp_type == 4:
        return (div_label_stream4(df))
    elif exp_type == 5:
        return (div_label_stream5(df))
    elif exp_type == 6:
        return (div_label_stream6(df))
    elif exp_type == 7:
        return (div_label_stream7(df))
    elif exp_type == 8:
        return (div_label_stream8(df))
        
def div_label_propaty(df,exp_type = 0):
    if exp_type == 0:
        print("there are no matching dataframe   ")
    elif exp_type == 1:
        return (div_label_propaty1(df))
    elif exp_type == 2:
        return (div_label_propaty2(df))
    elif exp_type == 3:
        return (div_label_propaty3(df))
    elif exp_type == 4:
        return (div_label_propaty4(df))
    elif exp_type == 5:
        return (div_label_propaty5(df))
    elif exp_type == 6:
        return (div_label_propaty6(df))
    elif exp_type == 7:
        return (div_label_propaty7(df))
    elif exp_type == 8:
        return (div_label_propaty8(df))
        
#0,3,10,20,30,50,100,50,30,20,10,3,0V
def div_label_stream1(df):
    label_div_1 = df[:7]
    label_div_2 = df[::-1]
    label_div_2 = label_div_2[:7]
    return([label_div_1,label_div_2])
#0x3,3x3,10x3,20x3,30x3,50x3,100x1,50x3,30x3,20x3,10x3,3x3,0x3Vを１往復    
def div_label_stream2(df):
    df_inv = df[::-1]        
    label_div_1 = [df[0],df[3],df[6],df[9],df[12],df[15],df[18]]
    label_div_2 = [df[1],df[4],df[7],df[10],df[13],df[16],df[19]]
    label_div_3 = [df[2],df[5],df[8],df[11],df[14],df[17],df[20]]
    label_div_4 = [df_inv[0],df_inv[3],df_inv[6],df_inv[9],df_inv[12],df_inv[15],df_inv[18]]
    label_div_5 = [df_inv[1],df_inv[4],df_inv[7],df_inv[10],df_inv[13],df_inv[16],df_inv[19]]
    label_div_6 = [df_inv[2],df_inv[5],df_inv[8],df_inv[11],df_inv[14],df_inv[17],df_inv[20]]
    return([label_div_1,label_div_2,label_div_3,label_div_4,label_div_5,label_div_6])
#0x3,3x3,10x3,20x3,30x3,50x3,100x1   
def div_label_stream3(df):
    label_div_1 = [df[0],df[3],df[6],df[9],df[12],df[15],df[18]]
    label_div_2 = [df[1],df[4],df[7],df[10],df[13],df[16],df[19]]
    label_div_3 = [df[2],df[5],df[8],df[11],df[14],df[17],df[20]]
    return([label_div_1,label_div_2,label_div_3])
#0,3,5,10,15,20,25,30,35,40,50,100,50,40,35,30,25,20,15,10,5,3,0Vを５往復
def div_label_stream4(df):
    label_div_1 = df[:13]   
    label_div_2 = df[12:25]
    label_div_2 = label_div_2[::-1]
    label_div_3 = df[24:37]
    label_div_4 = df[36:49]
    label_div_4 = label_div_4[::-1]
    label_div_5 = df[48:61]
    label_div_6 = df[60:72]
    label_div_6 = label_div_6[::-1]
    label_div_7 = df[72:85]
    label_div_8 = df[84:97]
    label_div_8 = label_div_8[::-1]
    
    label_div_9 = df[96:109]
    label_div_10 = df[108:121]
    label_div_10 = label_div_10[::-1]
    return([label_div_1,label_div_2,label_div_3,label_div_4,label_div_5,label_div_6,label_div_7,label_div_8,label_div_9,label_div_10])
#0,3,5,10,15,20,25,30,35,40,50,100,50,40,35,30,25,20,15,10,5,3,0Vを2往復
def div_label_stream5(df):
    label_div_1 = df[:13]   
    label_div_2 = df[12:25]
    label_div_2 = label_div_2[::-1]
    label_div_3 = df[24:37]
    label_div_4 = df[36:49]
    label_div_4 = label_div_4[::-1]
    
    return([label_div_1,label_div_2,label_div_3,label_div_4])

def div_label_stream6(df):
    label_div_1 = df[:12]   
    label_div_2 = df[11:23]
    label_div_2 = label_div_2[::-1]
    label_div_3 = df[23:35]
    label_div_4 = df[34:46]
    label_div_4 = label_div_4[::-1]
    
    return([label_div_1,label_div_2,label_div_3,label_div_4])
    
def div_label_stream7(df):
    label_div_1 = df[:12]   
    label_div_2 = df[11:23]
    label_div_2 = label_div_2[::-1]
     
    return([label_div_1,label_div_2])

def div_label_stream8(df):
    label_div_1 = df[:7]   
    label_div_2 = df[7:14]
    label_div_2 = label_div_2[::-1]
    label_div_3 = df[14:21]   
    label_div_4 = df[21:28]
    label_div_4 = label_div_4[::-1]
    label_div_5 = df[28:35]   
    label_div_6 = df[35:42]
    label_div_6 = label_div_6[::-1]
    label_div_7 = df[42:49]   
    label_div_8 = df[49:56]
    label_div_8 = label_div_8[::-1]     
    return([label_div_1,label_div_2,label_div_3,label_div_4,label_div_5,label_div_6,label_div_7,label_div_8])
    
def div_label_propaty1(df):
    label_div_nd_1 = [df[i][:7].values for i in df.columns]
    label_div_nd_2 = [df[i][6:].values for i in df.columns]
    label_div_nd_2 = [i[::-1] for i in label_div_nd_2]
    return([label_div_nd_1,label_div_nd_2])
#0x3,3x3,10x3,20x3,30x3,50x3,100x1,50x3,30x3,20x3,10x3,3x3,0x3Vを1往復 
def div_label_propaty2(df):
    label_div_1 = pd.concat([df.iloc[0,:],df.iloc[3,:],df.iloc[6,:],df.iloc[9,:],df.iloc[12,:],df.iloc[15,:],df.iloc[18,:]],axis=1)
    label_div_2 = pd.concat([df.iloc[1,:],df.iloc[4,:],df.iloc[7,:],df.iloc[10,:],df.iloc[13,:],df.iloc[16,:],df.iloc[19,:]],axis=1)        
    label_div_3 = pd.concat([df.iloc[2,:],df.iloc[5,:],df.iloc[8,:],df.iloc[11,:],df.iloc[14,:],df.iloc[17,:],df.iloc[20,:]],axis=1)
    label_div_4 = pd.concat([df.iloc[-3,:],df.iloc[-6,:],df.iloc[-9,:],df.iloc[-12,:],df.iloc[-15,:],df.iloc[-18,:],df.iloc[-21,:]],axis=1)
    label_div_5 = pd.concat([df.iloc[-2,:],df.iloc[-5,:],df.iloc[-8,:],df.iloc[-11,:],df.iloc[-14,:],df.iloc[-17,:],df.iloc[-20,:]],axis=1)
    label_div_6 = pd.concat([df.iloc[-1,:],df.iloc[-4,:],df.iloc[-7,:],df.iloc[-10,:],df.iloc[-13,:],df.iloc[-16,:],df.iloc[-19,:]],axis=1 )       

    label_div_1 = label_div_1.values        
    label_div_2 = label_div_2.values
    label_div_3 = label_div_3.values
    label_div_4 = label_div_4.values
    label_div_5 = label_div_5.values
    label_div_6 = label_div_6.values
    return([label_div_1,label_div_2,label_div_3,label_div_4,label_div_5,label_div_6])
 #0x3,3x3,10x3,20x3,30x3,50x3,100x1 
def div_label_propaty3(df):
    label_div_1 = pd.concat([df.iloc[0,:],df.iloc[3,:],df.iloc[6,:],df.iloc[9,:],df.iloc[12,:],df.iloc[15,:],df.iloc[18,:]],axis=1)
    label_div_2 = pd.concat([df.iloc[1,:],df.iloc[4,:],df.iloc[7,:],df.iloc[10,:],df.iloc[13,:],df.iloc[16,:],df.iloc[19,:]],axis=1)        
    label_div_3 = pd.concat([df.iloc[2,:],df.iloc[5,:],df.iloc[8,:],df.iloc[11,:],df.iloc[14,:],df.iloc[17,:],df.iloc[20,:]],axis=1)
    
    label_div_1 = label_div_1.values        
    label_div_2 = label_div_2.values
    label_div_3 = label_div_3.values
    return([label_div_1,label_div_2,label_div_3])
#0,3,5,10,15,20,25,30,35,40,50,100,50,40,35,30,25,20,15,10,5,3,0Vを５往復
def div_label_propaty4(df):
    label_div_nd_1 = [df[i][0:13].values for i in df.columns]
    label_div_nd_2 = [df[i][12:25].values for i in df.columns]
    label_div_nd_2 = [i[::-1] for i in label_div_nd_2]
    label_div_nd_3 = [df[i][24:37].values for i in df.columns]
    label_div_nd_4 = [df[i][36:49].values for i in df.columns]
    label_div_nd_4 = [i[::-1] for i in label_div_nd_4]
    label_div_nd_5 = [df[i][48:61].values for i in df.columns]
    label_div_nd_6 = [df[i][60:73].values for i in df.columns]
    label_div_nd_6 = [i[::-1] for i in label_div_nd_6]
    label_div_nd_7 = [df[i][72:85].values for i in df.columns]
    label_div_nd_8 = [df[i][84:97].values for i in df.columns]
    label_div_nd_8 = [i[::-1] for i in label_div_nd_8]
    label_div_nd_9 = [df[i][96:109].values for i in df.columns]
    label_div_nd_10 = [df[i][108:121].values for i in df.columns]
    label_div_nd_10 = [i[::-1] for i in label_div_nd_10]
    
    return([label_div_nd_1,label_div_nd_2,label_div_nd_3,label_div_nd_4,label_div_nd_5,label_div_nd_6,label_div_nd_7,label_div_nd_8,label_div_nd_9,label_div_nd_10])
#0,3,5,10,15,20,25,30,35,40,50,100,50,40,35,30,25,20,15,10,5,3,0Vを2往復 
def div_label_propaty5(df):
    label_div_nd_1 = [df[i][0:13].values for i in df.columns]
    label_div_nd_2 = [df[i][12:25].values for i in df.columns]
    label_div_nd_2 = [i[::-1] for i in label_div_nd_2]
    label_div_nd_3 = [df[i][24:37].values for i in df.columns]
    label_div_nd_4 = [df[i][36:49].values for i in df.columns]
    label_div_nd_4 = [i[::-1] for i in label_div_nd_4]
    
    return([label_div_nd_1,label_div_nd_2,label_div_nd_3,label_div_nd_4])
def div_label_propaty6(df):
    label_div_nd_1 = [df[i][0:12].values for i in df.columns]
    label_div_nd_2 = [df[i][11:23].values for i in df.columns]
    label_div_nd_2 = [i[::-1] for i in label_div_nd_2]
    label_div_nd_3 = [df[i][23:35].values for i in df.columns]
    label_div_nd_4 = [df[i][34:46].values for i in df.columns]
    label_div_nd_4 = [i[::-1] for i in label_div_nd_4]
    
    return([label_div_nd_1,label_div_nd_2,label_div_nd_3,label_div_nd_4])
    
def div_label_propaty7(df):
    label_div_nd_1 = [df[i][0:12].values for i in df.columns]
    label_div_nd_2 = [df[i][11:23].values for i in df.columns]
    label_div_nd_2 = [i[::-1] for i in label_div_nd_2]
    
    return([label_div_nd_1,label_div_nd_2])

def div_label_propaty8(df):
    label_div_nd_1 = [df[i][0:7].values for i in df.columns]
    label_div_nd_2 = [df[i][7:14].values for i in df.columns]
    label_div_nd_2 = [i[::-1] for i in label_div_nd_2]
    label_div_nd_3 = [df[i][14:21].values for i in df.columns]
    label_div_nd_4 = [df[i][21:28].values for i in df.columns]
    label_div_nd_4 = [i[::-1] for i in label_div_nd_4]    
    label_div_nd_5 = [df[i][28:35].values for i in df.columns]
    label_div_nd_6 = [df[i][35:42].values for i in df.columns]
    label_div_nd_6 = [i[::-1] for i in label_div_nd_6]    
    label_div_nd_7 = [df[i][42:49].values for i in df.columns]
    label_div_nd_8 = [df[i][49:56].values for i in df.columns]
    label_div_nd_8 = [i[::-1] for i in label_div_nd_8]    
    
    return([label_div_nd_1,label_div_nd_2,label_div_nd_3,label_div_nd_4,label_div_nd_5,label_div_nd_6,label_div_nd_7,label_div_nd_8])