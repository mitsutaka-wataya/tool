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
        self.diff_repeat_prop = [i.diff_propaty for i in self.cell]
        self.diff_repeat_prop = pd.concat(self.diff_repeat_prop)
        self.diff_repeat_prop.reset_index(inplace=True)
        self.rate_repeat_prop = [i.rate_propaty for i in self.cell]
        self.rate_repeat_prop = pd.concat(self.rate_repeat_prop)
        self.rate_repeat_prop.reset_index(inplace=True)
        self.norm_repeat_prop = [i.norm_propaty for i in self.cell]
        self.norm_repeat_prop = pd.concat(self.norm_repeat_prop)
        self.norm_repeat_prop.reset_index(inplace=True)        
        
        
        self.outname = fname.split(".")[0]
        self.stream = pd.concat([i.tmseries for i in self.cell])
        self.stream.to_csv("time_series_"+fname,index=False)
        
        self.repeat_prop.to_csv("feature_raw_"+fname,index=False)
        self.diff_repeat_prop.to_csv("feature_diff_"+fname,index=False)
        self.rate_repeat_prop.to_csv("feature_rate_"+fname,index=False)
        self.norm_repeat_prop.to_csv("feature_norm_"+fname,index=False)
        
        self.feature = ["amplitude","AUC","preAUC","postAUC","waveform","peak_time","pre_timeconst","post_timeconst","SNrate","pre_waveform","post_waveform"]
        
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
        out = self.outname+"_timeSeries"
        try:os.mkdir(out)
        except:print("directly has already existed")
        for id in range(self.cell_num):
            pal=sns.color_palette("seismic", self.repeat_num)
            grid = sns.FacetGrid(data=self.stream[self.stream.ID==id],row="Voltage",hue="repeat",aspect=4,palette=pal)
            plt.rcParams["figure.dpi"] = 200
            grid.map(plt.plot,"time","intensity",ms=.7,alpha=0.4)
            grid.fig.suptitle("ID_"+str(id)+"_timeSeries")
            grid.fig.subplots_adjust(top=.95)
            plt.savefig(out+"/ID_"+str(id)+"_timeSeries",dpi=300)
    def plot_diff_stream(self):
        out = self.outname + "_diff_timeSeries"
        try:os.mkdir(out)
        except:print("directly has already existed")
        for id in range(self.cell_num):
            pal=sns.color_palette("seismic", self.repeat_num)
            grid = sns.FacetGrid(data=self.stream[self.stream.ID==id],row="Voltage",hue="repeat",aspect=4,palette=pal)
            plt.rcParams["figure.dpi"] = 200
            grid.map(plt.plot,"time","diff_intensity",ms=.7,alpha=0.4)
            grid.fig.suptitle("ID_"+str(id)+"_diff_timeSeries")
            grid.fig.subplots_adjust(top=.95)
            plt.savefig(out + "/ID_"+str(id)+"_diff_timeSeries",dpi=300)
    def plot_rate_stream(self):
        out =self.outname + "_rate_timeSeries"
        try:os.mkdir(out)
        except:print("directly has already existed")
        for id in range(self.cell_num):
            pal=sns.color_palette("seismic", self.repeat_num)
            grid = sns.FacetGrid(data=self.stream[self.stream.ID==id],row="Voltage",hue="repeat",aspect=4,palette=pal)
            plt.rcParams["figure.dpi"] = 200
            grid.map(plt.plot,"time","rate_intensity",ms=.7,alpha=0.4)
            grid.fig.suptitle("ID_"+str(id)+"_rate_timeSeries")
            grid.fig.subplots_adjust(top=.95)
            plt.savefig(out+"/ID_"+str(id)+"_rate_timeSeries",dpi=300)
    def plot_norm_stream(self):
        out = self.outname + "_norm_timeSeries"
        try:os.mkdir(out)
        except:print("directly has already existed")
        for id in range(self.cell_num):
            pal=sns.color_palette("seismic", self.repeat_num)
            grid = sns.FacetGrid(data=self.stream[self.stream.ID==id],row="Voltage",hue="repeat",aspect=4,palette=pal)
            plt.rcParams["figure.dpi"] = 200
            grid.map(plt.plot,"time","norm_intensity",ms=.7,alpha=0.4)
            grid.fig.suptitle("ID_"+str(id)+"_norm_timeSeries")
            grid.fig.subplots_adjust(top=.95)
            plt.savefig(out+"/ID_"+str(id)+"_norm_timeSeries",dpi=300)
            
    def plot_does_scatter(self,propaty,feature="amplitude",out=""):
        pal=sns.color_palette("seismic", int(self.repeat_num))
        #grid =sns.FacetGrid(data=propaty,col="ID",hue="repeat",col_wrap=5,palette=pal)
        grid = sns.factorplot(x='Voltage', y=feature, data=propaty,col="ID", hue='repeat', kind='swarm', col_wrap=5,palette=pal)
        plt.rcParams["figure.dpi"] = 200
        #grid.map(sns.factorplot,"Voltage",feature,data=propaty,kind="swarm",alpha=0.7)
        grid.fig.suptitle('Voltage_vs_'+feature)
        grid.fig.subplots_adjust(top=.9)
        plt.savefig(out+"Does_"+feature,dpi=300)
        grid.fig.clf()
        sns.plt.close()
        gc.collect()
    def plot_all_does_scatter(self,raw=False,diff=True,rate=False,norm=False):
        out = self.outname
        
        if raw:     
            try:os.mkdir(out+"_DoesResponse_scatter")
            except:print("directly has already existed")
            for f in self.feature:
                self.plot_does_scatter(feature=f,out=out+"_DoesResponse_scatter/",propaty=self.repeat_prop)
        if diff:
            try:os.mkdir(out+"_DoesResponse_scatter_diff")
            except:print("directly has already existed")
            for f in self.feature:
                self.plot_does_scatter(feature=f,out=out+"_DoesResponse_scatter_diff/",propaty=self.diff_repeat_prop)
        if rate:    
            try:os.mkdir(out+"_DoesResponse_scatter_rate")
            except:print("directly has already existed")
            for f in self.feature:
                self.plot_does_scatter(feature=f,out=out+"_DoesResponse_scatter_rate/",propaty=self.rate_repeat_prop)
        if norm:
            try:os.mkdir(out+"_DoesResponse_scatter_norm")
            except:print("directly has already existed")
            for f in self.feature:
                self.plot_does_scatter(feature=f,out=out+"_DoesResponse_scatter_norm/",propaty=self.norm_repeat_prop)

    def plot_does(self,propaty,feature="amplitude",out=""):
        pal=sns.color_palette("seismic", int(self.repeat_num))
        grid=sns.lmplot(x="Voltage",y=feature,data=propaty,col="ID",x_ci=95,fit_reg=False,col_wrap=5,palette=pal,x_estimator=np.mean)
        plt.rcParams["figure.dpi"] = 100
        #grid.map(sns.regplot,"Voltage","amplitude",x_estimator=np.mean)
        grid.fig.suptitle('Voltage_vs_'+feature)
        grid.fig.subplots_adjust(top=.9)
        #grid.fig.tight_layout(w_pad=1)
        plt.savefig(out+"Does_"+feature,dpi=300)
        grid.fig.clf()
        sns.plt.close()
        gc.collect()
    def plot_all_does(self,raw=False,diff=True,rate=False,norm=False):
        out = self.outname
        if raw:
            try:os.mkdir(out+"_DoesResponse")
            except:print("directly has already existed")          
            for f in self.feature:
                self.plot_does(feature=f,out=out+"_Doesresponse/",propaty=self.repeat_prop)
        if diff:
            try:os.mkdir(out+"_DoesResponse_diff")
            except:print("directly has already existed")
            for f in self.feature:
                self.plot_does(feature=f,out=out+"_Doesresponse_diff/",propaty=self.diff_repeat_prop)
        if rate:
            try:os.mkdir(out+"_DoesResponse_rate")
            except:print("directly has already existed")
            for f in self.feature:
                self.plot_does(feature=f,out=out+"_Doesresponse_rate/",propaty=self.rate_repeat_prop)
        if norm:
            try:os.mkdir(out+"_DoesResponse_norm")
            except:print("directly has already existed")
            for f in self.feature:
                self.plot_does(feature=f,out=out+"_Doesresponse_norm/",propaty=self.norm_repeat_prop)
                
    def plot_hist(self,propaty,out,feature="amplitude"):
        #pal=sns.color_palette("seismic", int(10))
        try:os.mkdir(out+"hist_"+feature)
        except:print("dirctly is already has existed")
        grid =sns.FacetGrid(data=propaty,row="Voltage",hue="Voltage",aspect=4)
        plt.rcParams["figure.dpi"] = 100
        grid.map(sns.distplot,feature,rug=True,color="b")
        grid.fig.suptitle("Allcell_"+feature)
        grid.fig.subplots_adjust(top=.95)
        plt.savefig(out+"hist_"+feature+"/Allcell_"+feature)
        grid.fig.clf()
        sns.plt.close()
        gc.collect()
        for id in range(self.repeat_prop.ID.max()+1):
            grid =sns.FacetGrid(data=propaty[propaty.ID==id],row="Voltage",hue="Voltage",aspect=4)
            plt.rcParams["figure.dpi"] = 100
            grid.map(sns.distplot,feature,rug=True,color="b")
            grid.fig.suptitle("ID_"+str(id)+"_"+feature)
            grid.fig.subplots_adjust(top=.95)
            plt.savefig(out+"hist_"+feature+"/ID_"+str(id)+"_"+feature,dpi=300)
            grid.fig.clf()
            sns.plt.close()
            gc.collect()
    def plot_all_hist(self,raw=False,diff=True,rate=False,norm=False):
        out = self.outname
        
        if raw:
            try:os.mkdir(out+"_hist")
            except:print("dirctly is already has existed")
            for f in self.feature:
                self.plot_hist(feature=f,out=out+"_hist/",propaty=self.repeat_prop)
        if diff:   
            try:os.mkdir(out+"_hist_diff")
            except:print("dirctly is already has existed")
            for f in self.feature:
                self.plot_hist(feature=f,out=out+"_hist_diff/",propaty=self.diff_repeat_prop)
        if rate:    
            try:os.mkdir(out+"_hist_rate")
            except:print("dirctly is already has existed")
            for f in self.feature:
                self.plot_hist(feature=f,out=out+"_hist_rate/",propaty=self.rate_repeat_prop)    
        if norm:
            try:os.mkdir(out+"_hist_norm")
            except:print("dirctly is already has existed")
            for f in self.feature:
                self.plot_hist(feature=f,out=out+"_hist_norm/",propaty=self.norm_repeat_prop)
            
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
            plt.savefig("pairplot/ID_"+str(id),dpi=300)
            plt.clf()
            sns.plt.close()
            gc.collect()
            
class Cell(object):
    def __init__(self,cell_df,repeat_num):
        self.stream = [cell_df[cell_df.repeat == i].copy() for i in range(repeat_num)]
        self.tmseries = cell_df.copy()
        self.ID = int(self.stream[0].ID.iloc[0])     
        self.propaty = self.make_df(self.stream,fc=True)
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
        
        diff = [self.tmseries[self.tmseries.repeat == i].copy() for i in range(repeat_num)]
        diff = [i.loc[:,["diff_intensity","time","Voltage","ID","repeat","stim"]] for i in diff]
        diff = [i.rename(columns={"diff_intensity":"intensity"})for i in diff]
        rate = [self.tmseries[self.tmseries.repeat == i].copy() for i in range(repeat_num)]
        rate = [i.loc[:,["rate_intensity","time","Voltage","ID","repeat","stim"]] for i in rate]
        rate = [i.rename(columns={"rate_intensity":"intensity"})for i in rate]
        norm = [self.tmseries[self.tmseries.repeat == i].copy() for i in range(repeat_num)]
        norm = [i.loc[:,["norm_intensity","time","Voltage","ID","repeat","stim"]] for i in norm]
        norm = [i.rename(columns={"norm_intensity":"intensity"})for i in norm]
        
        self.diff_propaty = self.make_df(diff,fc=False)
        self.rate_propaty = self.make_df(rate,fc=False)
        self.norm_propaty = self.make_df(norm,fc=False)
        
    def remove_no_excite(self):
        self.propaty = self.propaty[self.propaty.count_sd <3]
        print(len(self.propaty))
    def make_df(self,ts_list,fc=True):
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
        error_num =0
        count=0
        for df in ts_list:
            df = df.sort_index()
            print(df)
            repeat_num += [df["repeat"].iloc[0]]
            Voltage += [int(df["Voltage"].iloc[0])]            
            prebasal = df[df.stim==False].intensity.mean()
            pre_basal += [prebasal]            
            prestd = df[df.stim==False].intensity.std()
            pre_std += [prestd]
            post_std += [df[df.stim==True].intensity.std()]            
            maxint = df[(df.stim==True)&(df.time!=df.time.max())&(df.time!=df.time.min())].intensity.max()
            max_int += [maxint]
            peaktime = df[df.intensity == maxint]
            peaktime = peaktime[peaktime.stim == True].time.min()
            peaktime = np.int(peaktime)            
            peak_time += [peaktime]
            amp = (np.double(maxint-prebasal))
            amplitude+=[amp]
            snr = np.double((maxint-prebasal)/prestd)
            snrate += [snr]
            
            if fc:
                frc = maxint/prebasal
                fold_change += [frc]
            
            t = df.time.iloc[3] - df.time.iloc[2]
            preint = df[(df.time<= peaktime)].intensity.values.flatten()
            pretime = df[(df.time<= peaktime)].time.values.flatten()
            preauc = preauc = integrate.simps(preint,pretime)
            preAUC += [np.double(preauc)]
            peakhaｌf = amp/2 + prebasal 
            if [] == list(np.where(preint[preint==peakhalf])):
                prehalf = pretime[np.where(preint[preint==peakhalf])]
            else:
                try:t1 = pretime[np.where((preint<peakhalf)&(preint!=preint[-1]))].max()
                except:t1 = np.int(peaktime-t)
                t2 = np.int(t1 + t)
                i1 = preint[np.where(pretime==t1)].max()
                i2 = preint[np.where(pretime==t2)].max()
                prehalf = t1 + t*np.abs((peakhalf-i1)/(i2-i1))
            pre_timeconst += [prehalf]

            postint = df[(df.time>= peaktime)].intensity.values.flatten()
            posttime = df[(df.time>= peaktime)].time.values.flatten()
            postauc = postauc = integrate.simps(postint,posttime)
            postAUC += [np.double(preauc)]

            if [] == list(np.where(postint[postint==peakhalf])):
                posthalf = posttime[np.where(postint[postint==peakhalf])]
            else:
                try:t2 = posttime[np.where((postint<peakhalf)&(postint!=postint[0]))].min()
                except:t2 = int(peaktime + t)
                t1 = np.int(t2-t)
                i1 = postint[np.where(posttime==t1)].max()
                i2 = postint[np.where(posttime==t2)].max()
                posthalf = t1 + t*np.abs((peakhalf-i1)/(i2-i1))
            post_timeconst += [posthalf]
            
            AUC += [preauc+postauc]
            pre_waveform += [np.double(np.absolute(preauc/((pretime.max()-pretime.min())*maxint)))]
            post_waveform += [np.double(np.absolute(postauc/((posttime.max()-posttime.min())*maxint)))]
            waveform += [np.double(np.absolute((preauc+postauc)/(maxint*(posttime.max()-pretime.min()))))]
            x1 = posttime[ (postint - ((postint[0]-postint[-1])/2 + postint[-1])) >= 0].max()
            x2 = x1+t
            
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

        if fc:
            propaty = pd.DataFrame({ "repeat":repeat_num ,
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
        else:
            propaty = pd.DataFrame({ "repeat":repeat_num ,
                                          "Voltage":Voltage,
                                          "pre_basal":pre_basal ,
                                          "pre_std":pre_std ,
                                          "post_std":post_std ,
                                          "max_intensity":max_int ,
                                          "peak_time":peak_time,
                                          "amplitude":amplitude,
                                          "SNrate":snrate ,
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
        return(propaty)
           
    
    def get_timeconst(self,y):
        x = self.propaty.repeat.values
        y_time = y[0]* np.e
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
       
            

            
            
            