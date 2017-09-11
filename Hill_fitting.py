# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 15:56:12 2017

@author: test
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os
import glob
import scipy
from scipy.optimize import curve_fit


#model
def Hill_fun(x,C,Kd,n):
    #C=initial_param[0]
    #Kd=initial_param[1]
    #n=initial_param[2]
    return(C*(x**n/(Kd**n + x**n)))

#出力形式
class Cell_Dose(object):
    def __init__(self,amp,auc,vol,repeat_num):
        self.amplitude   = amp
        self.AUC        = auc
        self.Voltage    = vol
        self.repeat_num = repeat_num


def read_feature_csv(fname):
    diff = pd.read_csv(fname,index_col=0)
    diff = diff.sort_values("Voltage")
    
    #get all cells ID
    cell_num = diff.ID.values.flatten()
    cell_num.sort()
    cell_num=np.unique(cell_num)
    
    #get feature array divited by ID
    cell = {}
    for i in cell_num:
        vol = diff[diff.ID==i].Voltage.values.flatten()
        vol.sort()
        vol = np.unique(vol)
        repeat_num = int((diff[diff.ID==i].repeat.max()+1)/10)
        amp = [diff[(diff.ID==i)&(diff.repeat>=10*r)&(diff.repeat<10*(r+1))].amplitude.values.flatten() for r in range(repeat_num)]
        auc = [diff[(diff.ID==i)&(diff.repeat>=10*r)&(diff.repeat<10*(r+1))].AUC.values.flatten() for r in range(repeat_num)]
        #cell = {"amplitude":amp , "AUC":auc,"Voltage":vol,"repeat":repeat_num}
        cell[i] = Cell_Dose(amp,auc,vol,repeat_num)
    return(cell_num,cell)
    
def Hill_fit(cell_id,cell,Kd0=30.,n0=3.,nmax=10.,Kdmax_rate=2.):
    cell_num =cell_id
    for i in cell_num:
        p0_amp=np.array([np.max(cell[i].amplitude),Kd0,n0])
        p0_auc=np.array([np.max(cell[i].AUC),Kd0,n0])
        b_amp = ([0,[np.max(cell[i].amplitude),np.max(cell[i].Voltage)*Kdmax_rate,nmax]])
        b_auc = ([0,[np.max(cell[i].AUC),np.max(cell[i].Voltage)*Kdmax_rate,nmax]])
        repeat = cell[i].repeat_num
        popt_amp = []
        popt_auc = []
        error_num = 0
        sim_xt = np.linspace(0,np.max(cell[i].Voltage),500)
        sim_amp = []
        sim_auc = []
        for r in range(repeat):
            try:
                p_amp,pcov = curve_fit(Hill_fun,xdata=cell[i].Voltage,ydata=cell[i].amplitude[r],p0=p0_amp,bounds=b_amp,method="trf")
                sim_am = Hill_fun(sim_xt,p_amp[0],p_amp[1],p_amp[2])
            except:
                p_amp = np.array([-1,-1,-1])
                sim_am = Hill_fun(sim_xt,1,1,1)
                error_num +=1
            try:
                p_auc,pcov = curve_fit(Hill_fun,xdata=cell[i].Voltage,ydata=cell[i].AUC[r],p0=p0_auc,bounds=b_auc,method="trf")
                sim_au = Hill_fun(sim_xt,p_auc[0],p_auc[1],p_auc[2])
            except:
                p_auc = np.array([-1,-1,-1])
                sim_au = Hill_fun(sim_xt,1,1,1)
                error_num +=1
            popt_amp+=[p_amp]
            popt_auc+=[p_auc]
            sim_amp+=[np.array(sim_am)]
            sim_auc+=[np.array(sim_au)]
        popt_amp = np.array(popt_amp)
        popt_auc = np.array(popt_auc)
        cell[i].C_amp = popt_amp[:,0]
        cell[i].Kd_amp = popt_amp[:,1]
        cell[i].n_amp = popt_amp[:,2]
        cell[i].C_auc = popt_auc[:,0]
        cell[i].Kd_auc = popt_auc[:,1]
        cell[i].n_auc = popt_auc[:,2]
        cell[i].sim_amp = sim_amp
        cell[i].sim_auc = sim_auc
        cell[i].sim_x = sim_xt
        print("finished id:%d , error num was %d" %(i,error_num))
    return(cell)

def save_param(ID,cell,fname,out=None):
    cell_num = ID
    Kd_amp = [list(cell[i].Kd_amp) for i in cell_num]
    n_amp = [list(cell[i].n_amp) for i in cell_num]
    C_amp = [list(cell[i].C_amp) for i in cell_num]
    Kd_auc = [list(cell[i].Kd_auc) for i in cell_num]
    n_auc = [list(cell[i].n_auc) for i in cell_num]
    C_auc = [list(cell[i].C_auc) for i in cell_num]
    cell[0].repeat_num
    param=[]
    for i in cell_num:
        x=pd.DataFrame({"Kd_amp":Kd_amp[i],"m_amp":n_amp[i],"C_amp":C_amp[i],
                        "Kd_auc":Kd_auc[i],"n_auc":n_auc[i],"C_auc":C_auc[i]})
        x["ID"] = i
        x["repeat"] = list(x.index)
        param +=[x]
    param = pd.concat(param)
    param.reset_index()
    
    name = fname.split(".")[0]
    if out == None:
        param.to_csv("fit_param"+name+".csv")
    else:
        param.to_csv(out+"/fit_param"+name+".csv")
def save_dose(ID,cell,fname,out=None):
    
    cell_num = ID
    dose = []
    for i in cell_num:
        x=pd.DataFrame(cell[i].amplitude)
        x.columns=cell[i].Voltage
        x["ID"]=i
        dose += [x]
    dose = pd.concat(dose)
    dose = dose.reset_index()
    dose = dose.rename(columns={"index":"repeat"})
    
    name = fname.split(".")[0]
    if out == None:
        dose.to_csv("dose_amplitude"+name+".csv")
    else:
        dose.to_csv(out+"/dose_amplitude"+name+".csv")
    dose_amp = dose
    dose = []
    for i in cell_num:
        x=pd.DataFrame(cell[i].AUC)
        x.columns=cell[i].Voltage
        x["ID"]=i
        dose += [x]
    dose = pd.concat(dose)
    dose = dose.reset_index()
    dose = dose.rename(columns={"index":"repeat"})
    if out == None:
        dose.to_csv("dose_AUC"+name+".csv")
    else:
        dose.to_csv(out+"/dose_AUC"+name+".csv")
    dose_auc = dose
    
    return[dose_amp,dose_auc]
    
def plot_dose(cell_id,cell):
    for i in cell_id:
        for j in range(cell[i].repeat_num):
            plt.plot(cell[i].sim_x,cell[i].sim_amp[j])
            plt.scatter(cell[i].Voltage,cell[i].amplitude[j],s=5)
        plt.title("id %damplitude dose_response" %i)
        plt.show()
def plot_param_scatter(cell_id,cell):
    for i in cell_id:
        plt.scatter(cell[i].Kd_amp.flatten(),cell[i].n_amp.flatten(),s=3)
    plt.xlabel("Kd")
    plt.ylabel("Hill param(n)")
    plt.title("amplitude:opt param ")
    plt.show()
        
def fit_dose2hill(fname):
    ID,cell = read_feature_csv(fname)
    Hill_fit(ID,cell)
    plot_dose(ID,cell)
    plot_param_scatter(ID,cell)
    return(ID,cell)

    