# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 16:15:31 2017

@author: waizoo
"""
import pandas as pd
from scipy import stats 
from scipy.optimize import curve_fit
from scipy import integrate
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import lowpass_filtter as lpf
import os
import gc

class AllCells(object):
    def __init__(self,fname):
        df = pd.read_csv(fname,index_col = 0)
        self.stream = df.copy()
        self.repeat_num = df.repeat.max() + 1
        self.cell_num = df.ID.max() + 1
        self.cell = [Cell(df[df.ID == i],self.repeat_num) for i in range(self.cell_num) ]        
        self.repeat_prop = [i.propaty for i in self.cell]
        self.repeat_prop = pd.concat(self.repeat_prop)
        self.repeat_prop.reset_index(inplace=True)
        self.stream = pd.concat([i.tmseries for i in self.cell])
        self.repeat_prop.to_csv("feature_"+fname,index=False)
        
        self.feature = ["amplitude","AUC","preAUC","postAUC","waveform","peak_time","pre_timeconst","post_timeconst","SNrate","pre_waveform","post_waveform","fold_change"]
        
    def select_cell(self,limit = 0):        
        x=self.repeat_prop[self.repeat_prop.repeat == limit]
        ecxite_cell_ID = x[x.count_sd == 0].ID
        ecxite_cell = [self.repeat_prop[self.repeat_prop.ID == i] for i in ecxite_cell_ID]
        ecxite_cell = pd.concat(ecxite_cell)
        ecxite_cell.reset_index(inplace=True)
        print("select_cell:"+str(len(ecxite_cell_ID))+"/"+str(self.cell_num))
        new_cell = []        
        for i in ecxite_cell_ID:
            for j in self.cell:
                if j.ID == i:
                    new_cell += [j]
        self.cell = new_cell
        self.repeat_prop = [i.propaty for i in self.cell]
        self.repeat_prop = pd.concat(self.repeat_prop)
        #return (ecxite_cell)
        
    def get_timeconst(self):
        self.max_timeconst = [i.fix_breach(i.propaty.max_intensity.values) for i in self.cell]
        #self.real_timeconst = [i.get_timeconst(i.propaty.max_intensity.values) for i in self.cell]        
        #self.base_timeconst = [i.fix_breach(i.propaty.pre_basal.values)[0] for i in self.cell]
        #self.amp_timeconst = [i.fix_breach(i.propaty.amplitude.values)[0] for i in self.cell]
    
    def show_cells_feature(self,feature="max_intensity"):
        #x = df["repeat"]        
        for i in self.cell:
            sns.plt.plot(i.propaty[feature],label=str(i.ID))
            sns.plt.legend()
            sns.plt.show()
            
    def show_stream(self,repeat_num=0):
        for i in self.cell:
            plt.plot(i.stream[repeat_num].time,i.stream[repeat_num].intensity,label=str(i.ID))
            plt.legend()
            #plt.ylim((-10,180))
            plt.show()
            print("repeat:"+str(i.stream[repeat_num].repeat.iloc[0]))
            print("count_sd:"+str(i.propaty[i.propaty.repeat == repeat_num].count_sd.values))
    
    def plot_stream(self):
        try:os.mkdir("timeSeries")
        except:print("directly has already existed")
        for id in range(self.cell_num):
            pal=sns.color_palette("seismic", self.repeat_num)
            grid = sns.FacetGrid(data=self.stream[self.stream.ID==id],row="Voltage",hue="repeat",aspect=4,palette=pal)
            plt.rcParams["figure.dpi"] = 200
            grid.map(plt.plot,"time","intensity",ms=.7,alpha=0.4)
            grid.fig.suptitle("ID_"+str(id)+"_timeSeries")
            grid.fig.subplots_adjust(top=.95)
            plt.savefig("timeSeries/ID_"+str(id)+"_timeSeries")
    def plot_diff_stream(self):
        try:os.mkdir("diff_timeSeries")
        except:print("directly has already existed")
        for id in range(self.cell_num):
            pal=sns.color_palette("seismic", self.repeat_num)
            grid = sns.FacetGrid(data=self.stream[self.stream.ID==id],row="Voltage",hue="repeat",aspect=4,palette=pal)
            plt.rcParams["figure.dpi"] = 200
            grid.map(plt.plot,"time","diff_intensity",ms=.7,alpha=0.4)
            grid.fig.suptitle("ID_"+str(id)+"_diff_timeSeries")
            grid.fig.subplots_adjust(top=.95)
            plt.savefig("diff_timeSeries/ID_"+str(id)+"_diff_timeSeries")
    def plot_rate_stream(self):
        try:os.mkdir("rate_timeSeries")
        except:print("directly has already existed")
        for id in range(self.cell_num):
            pal=sns.color_palette("seismic", self.repeat_num)
            grid = sns.FacetGrid(data=self.stream[self.stream.ID==id],row="Voltage",hue="repeat",aspect=4,palette=pal)
            plt.rcParams["figure.dpi"] = 200
            grid.map(plt.plot,"time","rate_intensity",ms=.7,alpha=0.4)
            grid.fig.suptitle("ID_"+str(id)+"_rate_timeSeries")
            grid.fig.subplots_adjust(top=.95)
            plt.savefig("rate_timeSeries/ID_"+str(id)+"_rate_timeSeries")
    def plot_norm_stream(self):
        try:os.mkdir("norm_timeSeries")
        except:print("directly has already existed")
        for id in range(self.cell_num):
            pal=sns.color_palette("seismic", self.repeat_num)
            grid = sns.FacetGrid(data=self.stream[self.stream.ID==id],row="Voltage",hue="repeat",aspect=4,palette=pal)
            plt.rcParams["figure.dpi"] = 200
            grid.map(plt.plot,"time","norm_intensity",ms=.7,alpha=0.4)
            grid.fig.suptitle("ID_"+str(id)+"_norm_timeSeries")
            grid.fig.subplots_adjust(top=.95)
            plt.savefig("norm_timeSeries/ID_"+str(id)+"_norm_timeSeries")
    def plot_does_scatter(self,feature="amplitude",out=""):
        pal=sns.color_palette("seismic", int(self.repeat_num))
        grid =sns.FacetGrid(data=self.repeat_prop,col="ID",hue="repeat",col_wrap=5,palette=pal)
        plt.rcParams["figure.dpi"] = 200
        grid.map(plt.plot,"Voltage",feature,marker="o",ms=5,alpha=0.7)
        grid.fig.suptitle('Voltage_vs_'+feature)
        grid.fig.subplots_adjust(top=.9)
        plt.savefig(out+"Does_"+feature)
        grid.fig.clf()
        sns.plt.close()
        gc.collect()
    def plot_all_does_scatter(self):
        try:os.mkdir("DoesResponse_scatter")
        except:print("directly has already existed")
        for f in self.feature:
            self.plot_does_scatter(feature=f,out="DoesResponse_scatter/")
    def plot_does(self,feature="amplitude",out=""):
        pal=sns.color_palette("seismic", int(self.repeat_num))
        grid=sns.lmplot(x="Voltage",y=feature,data=self.repeat_prop,col="ID",x_ci=95,fit_reg=False,col_wrap=5,palette=pal,x_estimator=np.mean)
        plt.rcParams["figure.dpi"] = 100
        #grid.map(sns.regplot,"Voltage","amplitude",x_estimator=np.mean)
        grid.fig.suptitle('Voltage_vs_'+feature)
        grid.fig.subplots_adjust(top=.9)
        #grid.fig.tight_layout(w_pad=1)
        plt.savefig(out+"Does_"+feature)
        grid.fig.clf()
        sns.plt.close()
        gc.collect()
    def plot_all_does(self):
        try:os.mkdir("DoesResponse")
        except:print("directly has already existed")
        for f in self.feature:
            self.plot_does(f,out="Doesresponse/")
    def plot_hist(self,feature="amplitude"):
        pal=sns.color_palette("seismic", int(10))
        try:os.mkdir("hist_"+feature)
        except:print("dirctly is already has existed")
        
        grid =sns.FacetGrid(data=self.repeat_prop,row="Voltage",hue="Voltage",aspect=4,palette=pal)
        plt.rcParams["figure.dpi"] = 100
        grid.map(sns.distplot,feature)
        grid.fig.suptitle("Allcell_"+feature)
        grid.fig.subplots_adjust(top=.95)
        plt.savefig("hist_"+feature+"/Allcell_"+feature)
        grid.fig.clf()
        sns.plt.close()
        gc.collect()
        for id in range(self.repeat_prop.ID.max()+1):
            grid =sns.FacetGrid(data=self.repeat_prop[self.repeat_prop.ID==id],row="Voltage",hue="Voltage",aspect=4,palette=pal)
            plt.rcParams["figure.dpi"] = 100
            grid.map(sns.distplot,feature)
            grid.fig.suptitle("ID_"+str(id)+"_"+feature)
            grid.fig.subplots_adjust(top=.95)
            plt.savefig("hist_"+feature+"/ID_"+str(id)+"_"+feature)
            grid.fig.clf()
            sns.plt.close()
            gc.collect()
    def plot_all_hist(self):
        for f in self.feature:
            self.plot_hist(f)
            
    def plot_pairplot(self):
        try:os.mkdir("pairplot")
        except:print("dirctly is already has existed")
        pal=sns.color_palette("seismic", 10)
        for id in range(int(self.repeat_prop.ID.max()+1)):
            x=self.repeat_prop[self.repeat_prop.ID==id].loc[:,self.feature+["Voltage"]]
            plt.rcParams["figure.dpi"] = 100
            sns.pairplot(data=x,hue="Voltage",palette=pal)
            plt.suptitle('pairplot_ID_'+str(id))
            plt.subplots_adjust(top=.95)
            plt.savefig("pairplot/ID_"+str(id))
            plt.clf()
            sns.plt.close()
            gc.collect()
            
class Cell(object):
    def __init__(self,cell_df,repeat_num):
        self.stream = [cell_df[cell_df.repeat == i] for i in range(repeat_num)]
        self.tmseries = cell_df.copy()
        self.ID = int(self.stream[0].ID.iloc[0])     
        self.make_df()
        if self.propaty.count_sd.min() == 1:
            self.max_repeat = 0
        else:
            self.max_repeat = self.propaty[self.propaty.count_sd == 0].repeat.max()
        
        
        diff = [i.intensity.values for i in self.stream]
        rate = [i.intensity.values for i in self.stream]
        norm = [i.intensity.values for i in self.stream]        
        for i in range(repeat_num):
            diff[i] = diff[i] - np.double(self.propaty[self.propaty.repeat==i].pre_basal)
            rate[i] = rate[i] / np.double(self.propaty[self.propaty.repeat==i].pre_basal)
            norm[i] = (norm[i] - np.double(self.propaty[self.propaty.repeat==i].pre_basal))/np.double(self.propaty[self.propaty.repeat==i].pre_basal)
        self.tmseries["diff_intensity"] = list(np.concatenate(diff)) 
        self.tmseries["rate_intensity"] = list(np.concatenate(rate))
        self.tmseries["norm_intensity"] = list(np.concatenate(norm))
    def remove_no_excite(self):
        self.propaty = self.propaty[self.propaty.count_sd <3]
        print(len(self.propaty))
    def make_df(self):
        alpha = 0.01
        repeat_num = []
        pre_basal = []
        pre_std = []
        post_std = []
        max_int = []
        peak_time = []
        amplitude = []
        snrate = []
        p_value = []
        sig_diff = []
        count_sd = []
        cell_ID = []
        Voltage = []
        pre_timeconst = []
        post_timeconst = []
        preAUC = []
        postAUC =[]
        AUC = []
        pre_waveform = []
        post_waveform = []
        waveform = []
        fold_change = [] 
        
        count=0
        for df in self.stream:
            df = df.sort("time")
            repeat_num += [df["repeat"].iloc[0]]
            Voltage += [int(df["Voltage"].iloc[0])]            
            prebasal = df[df.stim==False].intensity.mean()
            pre_basal += [prebasal]            
            prestd = df[df.stim==False].intensity.std()
            pre_std += [prestd]
            post_std += [df[df.stim==True].intensity.std()]            
            maxint = df[df.stim==True].intensity.max()
            max_int += [maxint]
            peaktime = df[df.intensity == maxint]
            peaktime = peaktime[peaktime.stim == True].time.min()
            peaktime = np.int(peaktime)            
            peak_time += [peaktime]
            amp = (np.double(maxint-prebasal))
            amplitude+=[amp]
            snr = np.double((maxint-prebasal)/prestd)
            snrate += [snr]
            frc = maxint/prebasal
            fold_change += [frc]
            
            t = df.time.iloc[3] - df.time.iloc[2]
            pretime = np.array([ peaktime -3*t ,peaktime -2*t,peaktime - 1*t,peaktime])
            preint = np.array([df[df.time==pretime[0]].intensity.max(),df[df.time==pretime[1]].intensity.max(),df[df.time==pretime[2]].intensity.max(),maxint])
            preauc = integrate.simps(preint,pretime)
            preAUC += [np.double(preauc)]
            x1 = pretime[(preint - ((preint[-1]-preint[0])/2 + preint[0]))  <= 0].max()
            x2 = x1+t
            y1 = df[df.time == x1].intensity.max()
            y2 = df[df.time == x2].intensity.max()
            preharf = (x1 + (x2-x1)*((0.5*maxint - y1)/(y2-y1)))
            pre_timeconst += [preharf]

            posttime = df[df.time >= peaktime].time.values
            postint = df[df.time >= peaktime].intensity.values
            postauc = integrate.simps(postint,posttime)
            postAUC += [np.double(postauc)]            
            AUC += [np.double(preauc+postauc)]
            
            pre_waveform += [np.double(preauc/((pretime.max()-pretime.min())*maxint))]
            post_waveform += [np.double(postauc/((posttime.max()-posttime.min())*maxint))]
            waveform += [np.double((preauc+postauc)/(maxint*(posttime.max()-pretime.min())))]
            x1 = posttime[ (postint - ((postint[0]-postint[-1])/2 + postint[-1])) >= 0].max()
            x2 = x1+t
            y1 = df[df.time == x1].intensity.max()
            y2 = df[df.time == x2].intensity.max()
            postharf = (x1 + (x2-x1)*((y1 - 0.5*maxint)/(y1-y2)))
            post_timeconst += [postharf]
            
            
            
            p = stats.mannwhitneyu(df[df.stim ==True].intensity,df[df.stim ==False].intensity,alternative="greater")      
            p = p.pvalue
            p_value += [p]
            
            #x1 = df[df.time == (peaktime-70*2)].intensity
            x2 = df[df.time == (peaktime-70*1)].intensity
            x3 = df[df.time == (peaktime+70*1)].intensity
            x4 = df[df.time == (peaktime+70*2)].intensity         
            #x1 = np.double(x1)
            x2 = np.double(x2)
            x3 = np.double(x3)
            x4 = np.double(x4)
            #sig_diff += [0 if alpha > p else 1 ]
            
            if  (maxint-x2)>0 and (maxint-x3)>0 and (x3-x4)>0:
                sig_diff += [0]
            else:
                sig_diff += [1]
                count += 1
            """
            sig_diff += [0 if alpha > p else 1 ] 
            if alpha < p:            
                count += 1
            """
            count_sd +=[count]
            cell_ID += [self.ID]

        
        self.propaty = pd.DataFrame({ "repeat":repeat_num ,
                                      "Voltage":Voltage,
                                      "pre_basal":pre_basal ,
                                      "pre_std":pre_std ,
                                      "post_std":post_std ,
                                      "max_intensity":max_int ,
                                      "peak_time":peak_time,
                                      "amplitude":amplitude,
                                      "SNrate":snrate ,
                                      "fold_change":fold_change,
                                      "pre_timeconst":pre_timeconst,
                                      "post_timeconst":post_timeconst,
                                      "preAUC":preAUC,
                                      "postAUC":postAUC,
                                      "AUC":AUC,
                                      "waveform":waveform,
                                      "pre_waveform":pre_waveform,
                                      "post_waveform":post_waveform,
                                      "p_value":p_value ,
                                      "sig_diff":sig_diff, 
                                      "count_sd":count_sd,
                                      "ID":cell_ID })
        print("ID:"+ str(self.ID))   
    
    def get_timeconst(self,y):
        x = self.propaty.repeat.values
        y_time = y[0]* 0.36787944117144233
        time_const = x[y>y_time].max()        
        print("real time const:"+str(time_const))
        return (time_const)
        
    def fix_breach(self,y,low_pass=True):
        x= self.propaty.repeat.values   
        if low_pass :
            y = lpf.butter(x,y)
        """if intensity == "max":
            y = self.propaty.max_intensity.values
        elif intensity == "base":
            y = self.propaty.pre_basal.values
        """
        
        ymin = np.min(y)
        y = y-ymin
        y0 = y[0]        
        y = y/y0
        parame, cov = curve_fit(breaching_functiion,x,y,p0=[2,0.01,0.5],bounds=(-1,10))
        time_const = parame[0]
        fit_y = np.array([breaching_functiion(i,parame[0],parame[1],parame[2]) for i in x])
        RSS = np.sum(np.square(y-fit_y))
        plt.plot(x,y,x,fit_y)
        plt.show()
        print("cell ID:"+str(self.ID))
        print("RSS:"+str(RSS))
        #print("x0:"+str(parame[2]))
        #print("C:"+str(parame[2]))
        print("c1,c2:"+str(parame[2])+" , "+str(1-parame[2]))
        print("b:"+str(parame[0]))
        print("t:"+str(parame[1]))        
        #fix_y = y/fix_y
        return(parame[0],parame[1])
    

            
def breaching_functiion(x,b,t,c1):
    """
    if t1 < 0.01:
        t1 = 0.01
    if int(t2) <0.01:
        t2 = 0.01
    if C > 1:
        C = 1
    if C < 0:
        C = 0 
    return(C*np.exp(-x/t1) +(1-C)*np.exp(-x/t2) )
    """ 
    if 0 > b:
        b = 0.00000001
    if 0 > t:
        t = 0.00000001    
    if c1 < 0:
        c1 = 0
    n = np.sqrt(np.square(b)+4*np.square(t))
    c2 = 2*t/(b+n) - c1
    return(c1*((b+n)/(2*t))*np.exp((-0.5*n-b-2*t)*x) + c2*((b-n)/(2*t))*np.exp((0.5*n-b-2*t)*x)) 
   
    
class All_exp(object):
    
    def __init__(self,csv_name_list):
        self.exp_type = [AllCells(i) for i in csv_name_list]
        ND = [3,6,12,25,50]
        for i,j in zip(self.exp_type,ND):
            i.repeat_prop["ND_filtter"] = j 
        dfset=[i.repeat_prop for i in self.exp_type]
        self.dfset = pd.concat(dfset)

    def select_all_cell(self):
        for i in self.exp_type:
            i.select_cell()
        ND = [3,6,12,25,50]
        for i,j in zip(self.exp_type,ND):
            i.repeat_prop["ND_filtter"] = j 
        dfset=[i.repeat_prop for i in self.exp_type]
        self.dfset = pd.concat(dfset)
        
    def choice_repeat(self):
        self.choised_df = [self.dfset[self.dfset.repeat == i*10] for i in range(40)]
        self.choised_df = pd.concat(self.choised_df)
 
    def plot_feature_sub(self,col="ND_filtter",hue="ND_filtter",x="repeat",y="pre_basal"):
        grid=sns.FacetGrid(self.choised_df,col=col,hue=hue,size=5,col_wrap=5)
        grid = (grid.map(plt.plot,x,y).add_legend())
        sns.plt.savefig("ALL_ND_"+x+"_vs_"+y+"_by_"+hue+".png",dpi=300)        
        sns.plt.show()
       
            

            
            
            