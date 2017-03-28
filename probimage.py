# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 09:14:45 2016

@author: waizoo
"""

import numpy as np
from skimage.measure import regionprops
from skimage import measure
from skimage import io
from skimage.io import ImageCollection
import glob
import os
import pandas as pd
import re
import process_stream_csv as psc
import matplotlib.pyplot as plt
import div_dataframe as dd
import gc
import mahotas.labeled as mal
from scipy import stats
import time
import multiprocess as mp

class Label_Image(object):
    #dir means experiment data directory
    def __init__(self,dir=None,roi=None,label_image=None,mask=True,exp_type=None,mask_type=1,split=False,back_ground_subtract=True,plot_back_hist=False,SBtype=1):
        """
        exp_type 0: error
        exp_type 1: 0,3,10,20,30,50,100,50,30,20,10,3,0V
        exp_type 2: 0x3,3x3,10x3,20x3,30x3,50x3,100x1,50x3,30x3,20x3,10x3,3x3,0x3Vを１往復  
        exp_type 3: 0x3,3x3,10x3,20x3,30x3,50x3,100x1   
        exp_type 4: 0,3,5,10,15,20,25,30,35,40,50,100,50,40,35,30,25,20,15,10,5,3,0Vを５往復
        exp_type 5: 0,3,5,10,15,20,25,30,35,40,50,100,50,40,35,30,25,20,15,10,5,3Vを2往復
        exp_type 6: 0,5,10,15,20,25,30,35,40,50,100,50,40,35,30,25,20,15,10,5,0Vを2往復
        exp_type 7: 0,5,10,15,20,25,30,35,40,50,100,50,40,35,30,25,20,15,10,5,0Vを1往復      
        exp_type 8: 0,5,10,15,20,25,30,30,25,20,15,10,5,0Vを4往復  
        """
        """
        SBtype 0:no subtract
        SBtype 1:every photos subtract by every photos's background
        SBtype 2:every photos subtract by 1 Stream's background
        """
        print("program has started")
        
        self.back_ground_subtract = back_ground_subtract
        self.exp_type = exp_type
        self.dir = dir.replace("\\","/")
        self.expname = self.dir.split("/")[-1]
        
        self.frame_time = 70
        self.area_thresh = 500
        if back_ground_subtract:
            if SBtype == 1:self.out_dir=self.dir + "/output_ind_subback_by_mode"
            elif SBtype == 2:self.out_dir=self.dir + "/output_stream_subback_by_mode"
        else:self.out_dir=self.dir + "/output"
        
        try:os.mkdir(self.out_dir)
        except:print("output has existed already")    
        print(self.out_dir)

        
        self.interval = self.get_interval()
        self.feature = ("f_basal","b_basal","peak","auc","modified_auc","auc_divided_by_peak")
                
        if split:
            stream_dir = glob.glob(dir+"/split/*V*")
            stream_dir = [i.replace("\\","/") for i in stream_dir]
        else:
            stream_fname = glob.glob(dir+"/Stream/*")
            stream_fname = [i.replace("\\","/") for i in stream_fname]
            
        streamname = [i.split("/")[-1] for i in stream_fname]
        self.Vol_list = [self.get_voltage(i) for i in streamname]                         
        self.Vol_str_list = [str(self.get_voltage(i))+"V" for i in streamname]
        all_vol = list(set(self.Vol_list))
        repeat_num_each_vol = [self.Vol_list.index(i) for i in all_vol]
        self.repeat_num = min(repeat_num_each_vol)
        
        if mask:
            start_time = self.get_time()
            
            #read mask image
            mask_dir = glob.glob(dir+"/"+str(mask_type)+"*mask*.tif")
            mask_dir = mask_dir[0].replace("\\","/")
            mask_fig = io.imread(mask_dir)
            self.labelimage = measure.label(mask_fig)           
            back_ground_label=(mask_fig+1)*(mask_fig==0)
            
            # remove small object
            sizes = mal.labeled_size(self.labelimage)            
            self.labelimage = mal.remove_regions_where(self.labelimage,sizes<self.area_thresh)
            self.labelimage,counts = mal.relabel(self.labelimage)
            self.label_num = self.labelimage.max()
            
            # get each average of intensity and remove back ground  
            label_props = regionprops(self.labelimage)
            self.label_area = [i.area for i in label_props]
            self.back_mod = []
            self.intensity_df_list = [self.get_stream_label_intensity(i,back_ground_label,SBtype=SBtype,plot_hist=plot_back_hist) for i in stream_fname]
            self.back_mod= pd.DataFrame(self.back_mod)
            self.back_mod.to_csv(self.out_dir+"/background_mode.csv")
            
            end_time=self.get_time()            
            print("intensity was mesured successfuly!!")
            print("times:"+str(int(end_time - start_time))+"s")
            
        elif label_image:
            self.labelimage = self.label_ndarray(dir)
            self.label_num = self.labelimage.max()
            self.intensity_df_list = [self.get_stream_label_intensity(i) for i in stream_dir]
            print("intensity was mesured successfuly!!")
            label_props = regionprops(self.labelimage)
            self.label_area = [i.area for i in label_props]        
        elif roi :
            csv_dir = glob.glob(dir+"/csv/*.csv")
            csv_dir = [i.replace("\\","/") for i in csv_dir]
            self.intensity_df_list = [pd.read_csv(i) for i in csv_dir ]
            time=list(range(len(self.intensity_df_list[0])))
            time = [i*self.frame_time for i in time ]
            for i in self.intensity_df_list:
                i.index=time
            self.intensity_df_list = [i.iloc[:,1:] for i in self.intensity_df_list]
            print("intensity was mesured successfuly!!")
            
        self.label_num = len(self.intensity_df_list[0].columns)        
        #self.x_label,self.legend_label = dd.get_x_label(self.exp_type)
        #self.get_propaty()
        self.save_csv()
        #self.save_figure()
        
    def get_time(self):
            return(time.time())

    def get_voltage(self,fname):
        
        pattern1 = r"[0-9]+V"
        result1 = re.search(pattern1,fname)
        pattern2 = r"[0-9]+"
        result2 = re.search(pattern2,result1.group())
        return(int(result2.group()))
        
    def get_interval(self):
        pattern1 = r"int[0-9]+"
        result1 = re.search(pattern1,self.expname)
        pattern2 = r"[0-9]+"
        result2 = re.search(pattern2,result1.group())
        return(int(result2.group()))

    #井上さんプログラムのマスクデータ（csv）からマスクされたnd配列を生成
    def label_ndarray(self,dir):
        label_data = glob.glob(dir+"/*labelImage.csv") 
        label_data[0]=label_data[0].replace("\\","/")
        #labelImage = np.loadtxt(label_data[0],dtype = "i")
        labellmage = pd.read_csv(label_data[0],header=None,)
        labellmage = labellmage.iloc[:,:].as_matrix()    
        labellmage = labellmage.astype(np.int64)
        return(labellmage)
    
    def get_label_intensity(self,intensity_image,int_df,labeled_back,SBtype=0):
        intensity = regionprops(self.labelimage,intensity_image)
        int_mean = [i.mean_intensity for i in intensity]
        if self.back_ground_subtract:
            if SBtype==0:
                print("error.Need define SBtype ")
            if SBtype==1:
                #every photos subtract by every photos's background
                back_region = measure.regionprops(labeled_back,intensity_image)[0]
                back_val = back_region.intensity_image[back_region.coords[:,0],back_region.coords[:,1]]
                back_intensity,counts = stats.mstats.mode(back_val)
                self.back_mod += [back_intensity]
                df = pd.DataFrame(int_mean).T - back_intensity
                return(pd.concat([int_df,df]))
                
            elif SBtype==2:
                #every photos subtract by 1 Stream's background
                back_region = measure.regionprops(labeled_back,intensity_image)[0]
                back_val = back_region.intensity_image[back_region.coords[:,0],back_region.coords[:,1]]
                self.back_val = np.r_[self.back_val,back_val]                
                df = pd.DataFrame(int_mean).T
                return(pd.concat([int_df,df]))                
        else:
            df = pd.DataFrame(int_mean).T
            return(pd.concat([int_df,df]))
        
    def get_stream_label_intensity(self,stream_fname,labeled_back,SBtype=0,plot_hist=True):
        # read stream tiff
        stream_img = ImageCollection(stream_fname)
        
        # make hist dir
        if SBtype==2:
            try:os.mkdir(self.out_dir+"/hist")
            except:print("")
        
        time = [i*self.frame_time for i in range(len(stream_img))]
        int_df = pd.DataFrame([])
        self.back_val = np.array([])        
        
        for i in stream_img:
            int_df = self.get_label_intensity(i,int_df,labeled_back,SBtype=SBtype)
            
        # remove back ground
        if SBtype==2:
            back_intensity,counts = stats.mstats.mode(self.back_val)
            int_df = int_df - back_intensity
            self.back_mod += [back_intensity]  
            if plot_hist:
                #plot back ground hist
                plt.figure()
                plt.hist(self.back_val,bins=300,range=(0,1200))
                figname = stream_fname.split("/")[-1]
                figname = figname.split(".")[0]
                plt.savefig(self.out_dir+"/hist/back_hist_"+figname+".png")
                plt.clf()                
                plt.close()
                
            
        label_name = ["label"+str(i+1) for i in range(self.label_num)]
        int_df.columns = label_name
        int_df.index = time        
        return(int_df)

    def get_propaty(self,ind=True):
        vol=self.Vol_str_list        
        col = self.x_label
        
        self.f_basal_df = psc.get_fbasal(self.intensity_df_list,stim_time= 5)    
        self.b_basal_df = psc.get_bbasal(self.intensity_df_list,cal_frame= 5)    
        if ind:        
            self.rate_df_list = psc.make_ind_rate_df(self.intensity_df_list)
            self.diff_df_list = psc.make_ind_diff_df(self.intensity_df_list)
            self.diffSeries_df_list = psc.make_ind_diffSeries_df(self.intensity_df_list)
        else:    
            self.rate_df_list = psc.make_rate_df(self.intensity_df_list,self.f_basal_df)
            self.rate_df_list = psc.make_diff_df(self.intensity_df_list,self.f_basal_df)
        #raw data        
        self.f_basal_df = psc.get_fbasal(self.intensity_df_list,stim_time= 5)    
        self.b_basal_df = psc.get_bbasal(self.intensity_df_list,cal_frame= 5) 
        self.peak_df = psc.search_peak(self.intensity_df_list)
        self.auc = psc.get_auc(self.intensity_df_list,self.frame_time)
        self.modified_auc=psc.get_modify_auc(self.intensity_df_list,self.f_basal_df,self.frame_time)
        self.f_basal_std = psc.get_fbasal_std(self.intensity_df_list,cal_frame=5)
        
        f_basal_list =dd.div_label_propaty(self.f_basal_df,exp_type=self.exp_type)
        b_basal_list =dd.div_label_propaty(self.b_basal_df,exp_type=self.exp_type)
        div_peak_list =dd.div_label_propaty(self.peak_df,exp_type=self.exp_type)
        div_auc_list = dd.div_label_propaty(self.auc,exp_type=self.exp_type)
        div_modified_auc_list =dd.div_label_propaty(self.modified_auc,exp_type=self.exp_type)
        
        self.div_f_basal_df = [pd.DataFrame(i,columns=col) for i in f_basal_list ]
        self.div_b_basal_df = [pd.DataFrame(i,columns=col) for i in b_basal_list ]
        self.div_peak = [pd.DataFrame(i,columns=col) for i in div_peak_list]
        self.div_auc = [pd.DataFrame(i,columns=col) for i in div_auc_list]
        self.div_modified_auc = [pd.DataFrame(i,columns=col) for i in div_modified_auc_list ]
        
        self.b_basal_df.index = vol
        self.f_basal_df.index = vol        
        self.peak_df.index = vol                
        self.auc.index = vol
        self.modified_auc.index = vol
        self.f_basal_std.index = vol
        
 
        #diff data
        self.diff_f_basal_df = psc.get_fbasal(self.diff_df_list,stim_time= 5)    
        self.diff_b_basal_df = psc.get_bbasal(self.diff_df_list,cal_frame= 5) 
        self.diff_peak_df = psc.search_peak(self.diff_df_list)
        self.diff_auc = psc.get_auc(self.diff_df_list,self.frame_time)
        self.diff_modified_auc=psc.get_modify_auc(self.diff_df_list,self.diff_f_basal_df,self.frame_time)
        self.diff_f_basal_std = psc.get_fbasal_std(self.diff_df_list,cal_frame=5)

        f_basal_list =dd.div_label_propaty(self.diff_f_basal_df,exp_type=self.exp_type)
        b_basal_list =dd.div_label_propaty(self.diff_b_basal_df,exp_type=self.exp_type)
        div_peak_list =dd.div_label_propaty(self.diff_peak_df,exp_type=self.exp_type)
        div_auc_list =dd.div_label_propaty(self.diff_auc,exp_type=self.exp_type)
        div_modified_auc_list =dd.div_label_propaty(self.diff_modified_auc,exp_type=self.exp_type)
        
        self.div_diff_f_basal_df = [pd.DataFrame(i,columns=col) for i in f_basal_list]
        self.div_diff_b_basal_df = [pd.DataFrame(i,columns=col) for i in b_basal_list]
        self.div_diff_peak = [pd.DataFrame(i,columns=col) for i in div_peak_list]
        self.div_diff_auc = [pd.DataFrame(i,columns=col) for i in div_auc_list]
        self.div_diff_modified_auc = [pd.DataFrame(i,columns=col) for i in div_modified_auc_list]
        
        self.diff_b_basal_df.index = vol
        self.diff_f_basal_df.index = vol        
        self.diff_peak_df.index = vol                
        self.diff_auc.index = vol
        self.diff_modified_auc.index = vol
        self.diff_f_basal_std.index = vol
        
                
        #rate data
        self.rate_f_basal_df = psc.get_fbasal(self.rate_df_list,stim_time= 5)    
        self.rate_b_basal_df = psc.get_bbasal(self.rate_df_list,cal_frame= 5) 
        self.rate_peak_df = psc.search_peak(self.rate_df_list)
        self.rate_auc = psc.get_auc(self.rate_df_list,self.frame_time)
        self.rate_modified_auc=psc.get_modify_auc(self.rate_df_list,self.rate_f_basal_df,self.frame_time)
        self.rate_f_basal_std = psc.get_fbasal_std(self.rate_df_list,cal_frame=5)

        f_basal_list =dd.div_label_propaty(self.rate_f_basal_df,exp_type=self.exp_type)
        b_basal_list =dd.div_label_propaty(self.rate_b_basal_df,exp_type=self.exp_type)
        div_peak_list =dd.div_label_propaty(self.rate_peak_df,exp_type=self.exp_type)
        div_auc_list = dd.div_label_propaty(self.rate_auc,exp_type=self.exp_type)
        div_modified_auc_list = dd.div_label_propaty(self.rate_modified_auc,exp_type=self.exp_type)

        self.div_rate_f_basal_df = [pd.DataFrame(i,columns=col) for i in f_basal_list]
        self.div_rate_b_basal_df = [pd.DataFrame(i,columns=col) for i in b_basal_list]
        self.div_rate_peak = [pd.DataFrame(i,columns=col) for i in div_peak_list]
        self.div_rate_auc = [pd.DataFrame(i,columns=col) for i in div_auc_list]
        self.div_rate_modified_auc = [pd.DataFrame(i,columns=col) for i in div_modified_auc_list]
        
        self.rate_b_basal_df.index = vol
        self.rate_f_basal_df.index = vol        
        self.rate_peak_df.index = vol                
        self.rate_auc.index = vol
        self.rate_modified_auc.index = vol
        self.rate_f_basal_std.index = vol

        #diffSeries data
        self.diffsr_f_basal_df = psc.get_fbasal(self.diffSeries_df_list,stim_time= 5)    
        self.diffsr_b_basal_df = psc.get_bbasal(self.diffSeries_df_list,cal_frame= 5) 
        self.diffsr_peak_df = psc.search_peak(self.diffSeries_df_list)
        self.diffsr_auc = psc.get_auc(self.diffSeries_df_list,self.frame_time)
        self.diffsr_modified_auc=psc.get_modify_auc(self.diffSeries_df_list,self.diffsr_f_basal_df,self.frame_time)
        self.diffsr_f_basal_std = psc.get_fbasal_std(self.diffSeries_df_list,cal_frame=5)

        f_basal_list =dd.div_label_propaty(self.diffsr_f_basal_df,exp_type=self.exp_type)
        b_basal_list =dd.div_label_propaty(self.diffsr_b_basal_df,exp_type=self.exp_type)
        div_peak_list =dd.div_label_propaty(self.diffsr_peak_df,exp_type=self.exp_type)
        div_auc_list = dd.div_label_propaty(self.diffsr_auc,exp_type=self.exp_type)
        div_modified_auc_list = dd.div_label_propaty(self.diffsr_modified_auc,exp_type=self.exp_type)

        self.div_diffsr_f_basal_df = [pd.DataFrame(i,columns=col) for i in f_basal_list]
        self.div_diffsr_b_basal_df = [pd.DataFrame(i,columns=col) for i in b_basal_list]
        self.div_diffsr_peak = [pd.DataFrame(i,columns=col) for i in div_peak_list]
        self.div_diffsr_auc = [pd.DataFrame(i,columns=col) for i in div_auc_list]
        self.div_diffsr_modified_auc = [pd.DataFrame(i,columns=col) for i in div_modified_auc_list]
        
        self.diffsr_b_basal_df.index = vol
        self.diffsr_f_basal_df.index = vol        
        self.diffsr_peak_df.index = vol                
        self.diffsr_auc.index = vol
        self.diffsr_modified_auc.index = vol
        self.diffsr_f_basal_std.index = vol
        
        self.pop_raw_mean_list,self.pop_raw_std_list=self.get_cell_population_represent_value(self.div_f_basal_df,self.div_b_basal_df,self.div_peak,self.div_auc,self.div_modified_auc)
        self.pop_rate_mean_list,self.pop_rate_std_list=self.get_cell_population_represent_value(self.div_rate_f_basal_df,self.div_rate_b_basal_df,self.div_rate_peak,self.div_rate_auc,self.div_rate_modified_auc)        
        self.pop_diff_mean_list,self.diff_std_list=self.get_cell_population_represent_value(self.div_diff_f_basal_df,self.div_diff_b_basal_df,self.div_diff_peak,self.div_diff_auc,self.div_diff_modified_auc)


    def get_cell_population_represent_value(self,f_basal,b_basal,peak,auc,modified_auc,champ=False,thresh=None):
        pop_mean = []
        pop_std = []
        mean_col=[i+"_mean" for i in self.feature]
        std_col=[i+"_std" for i in self.feature]

        if champ:
            f_basal =[ f.ix[self.div_rate_peak[0].apply(max, axis=1)>thresh,:] for f in f_basal]
            b_basal =[ f.ix[self.div_rate_peak[0].apply(max, axis=1)>thresh,:] for f in b_basal]
            peak =[ f.ix[self.div_rate_peak[0].apply(max, axis=1)>thresh,:] for f in peak]
            auc =[ f.ix[self.div_rate_peak[0].apply(max, axis=1)>thresh,:] for f in auc]
            modified_auc =[ f.ix[self.div_rate_peak[0].apply(max, axis=1)>thresh,:] for f in modified_auc]
        auc_divided_by_peak = [a/p for a,p in zip(auc,peak) ]
        for f,b,p,a,m,a_p in zip(f_basal,b_basal,peak,auc,modified_auc,auc_divided_by_peak):
            df1 = pd.concat([f.mean(),b.mean(),p.mean(),a.mean(),m.mean(),a_p.mean()],axis=1)
            df1.columns=mean_col
            df2 = pd.concat([f.std(),b.std(),p.std(),a.std(),m.std(),a_p.std()],axis=1)
            df2.columns = std_col
            pop_mean += [df1]
            pop_std += [df2]
        return(pop_mean,pop_std)
    
    
    def plot_cell_population_dose_response(self,raw=True,rate=True,diff=True,champ=True):
        
        self.threshs=self.div_rate_peak[0].apply(max, axis=1).values.flatten()
        thresh = self.threshs        
        thresh.sort()
        thresh=thresh[-6]
        cell_id=str(list(self.div_peak[0].ix[self.div_rate_peak[0].apply(max, axis=1)>thresh,:].index))
        print("thresh:"+str(thresh))
        out_dir= self.out_dir
        if champ:
            raw_mean_list,raw_std_list=self.get_cell_population_represent_value(self.div_f_basal_df,self.div_b_basal_df,self.div_peak,self.div_auc,self.div_modified_auc,champ=champ,thresh=thresh)
            rate_mean_list,rate_std_list=self.get_cell_population_represent_value(self.div_rate_f_basal_df,self.div_rate_b_basal_df,self.div_rate_peak,self.div_rate_auc,self.div_rate_modified_auc,champ=champ,thresh=thresh)        
            diff_mean_list,diff_std_list=self.get_cell_population_represent_value(self.div_diff_f_basal_df,self.div_diff_b_basal_df,self.div_diff_peak,self.div_diff_auc,self.div_diff_modified_auc,champ=champ,thresh=thresh)
        else:
            raw_mean_list,raw_std_list=self.get_cell_population_represent_value(self.div_f_basal_df,self.div_b_basal_df,self.div_peak,self.div_auc,self.div_modified_auc,champ=champ)
            rate_mean_list,rate_std_list=self.get_cell_population_represent_value(self.div_rate_f_basal_df,self.div_rate_b_basal_df,self.div_rate_peak,self.div_rate_auc,self.div_rate_modified_auc,champ=champ)        
            diff_mean_list,diff_std_list=self.get_cell_population_represent_value(self.div_diff_f_basal_df,self.div_diff_b_basal_df,self.div_diff_peak,self.div_diff_auc,self.div_diff_modified_auc,champ=champ)
        if champ:        
            num=len(self.div_peak[0].ix[self.div_rate_peak[0].apply(max, axis=1)>thresh,:])
        fig = plt.figure(figsize=(10,10))
        axes = [fig.add_subplot(len(self.feature)/2+1,2,i+1) for i in range(len(self.feature))]
        for i in range(len(self.feature)):
            legend_mean = []            
            for mean,std in zip(raw_mean_list,raw_std_list):
                p = axes[i].plot(self.x_label,mean.iloc[:,i].values,"-o",linewidth=0.6,markersize=1,)
                axes[i].errorbar(self.x_label,mean.iloc[:,i].values,yerr=std.iloc[:,i].values,linewidth=0.3)  
                legend_mean += [p]
            axes[i].set_xscale('log')
            axes[i].tick_params(labelsize=5)
            axes[i].legend(legend_mean,labels=self.legend_label,loc="upper left",fontsize=3) 
            axes[i].set_title(self.feature[i],fontsize=5)
            

        if champ:
            fig.suptitle("Dose Response(raw)_"+str(num)+"_cells_int"+str(self.interval)+"\n cell id:"+cell_id,fontsize=10)    
        else:
            fig.suptitle("Dose Response(raw)_all_cell_int"+str(self.interval),fontsize=40)
        fig.subplots_adjust()
        if champ:
            #fig.savefig(out_dir+"/"+self.expname+"Dose Response(raw)_"+str(num)+"cells_int"+str(self.interval)+".pdf")
            fig.savefig(out_dir+"/"+self.expname+"Dose Response(raw)_"+str(num)+"cells_int"+str(self.interval)+".png",dpi=300)
        else:
            #fig.savefig(out_dir+"/"+self.expname+"Dose Response(raw)_all_cell_int"+str(self.interval)+".pdf")
            fig.savefig(out_dir+"/"+self.expname+"Dose Response(raw)_all_cell_int"+str(self.interval)+".png",dpi=300)

        plt.clf()
        plt.close()                
        
        fig = plt.figure(figsize=(10,10))
        axes = [fig.add_subplot(len(self.feature)/2+1,2,i+1) for i in range(len(self.feature))]
        legend_mean=[]
        for i in range(len(self.feature)):
            for mean,std in zip(rate_mean_list,rate_std_list):
                p = axes[i].plot(self.x_label,mean.iloc[:,i].values,"-o",linewidth=0.6,markersize=1,)
                axes[i].errorbar(self.x_label,mean.iloc[:,i].values,yerr=std.iloc[:,i].values,linewidth=0.3,)            
                legend_mean += [p]
            axes[i].tick_params(labelsize=5)                
            axes[i].set_xscale('log')
            axes[i].legend(legend_mean,labels=self.legend_label,loc="upper left",fontsize=3) 
            axes[i].set_title(self.feature[i],fontsize=5)
        if champ:
            fig.suptitle("Dose Response(rate)_"+str(num)+"cells_int"+str(self.interval)+"\n cell id:"+cell_id,fontsize=10)    
        else:
            fig.suptitle("Dose Response(rate)_all_cell_int"+str(self.interval),fontsize=40)
        fig.subplots_adjust()
        
        if champ:
            #fig.savefig(out_dir+"/"+self.expname+"Dose Response(rate)_"+str(num)+"cells_int"+str(self.interval)+".pdf")
            fig.savefig(out_dir+"/"+self.expname+"Dose Response(rate)_"+str(num)+"cells_int"+str(self.interval)+".png",dpi=300)
        else:
            #fig.savefig(out_dir+"/"+self.expname+"Dose Response(rate)_all_cell_int"+str(self.interval)+".pdf")
            fig.savefig(out_dir+"/"+self.expname+"Dose Response(rate)_all_cell_int"+str(self.interval)+".png",dpi=300)
        plt.clf()
        plt.close()        
        

        
        fig = plt.figure(figsize=(10,10))
        axes = [fig.add_subplot(len(self.feature)/2+1,2,i+1) for i in range(len(self.feature))]
        legend_mean=[]
        for i in range(len(self.feature)):
            for mean,std in zip(diff_mean_list,diff_std_list):
                p = axes[i].plot(self.x_label,mean.iloc[:,i].values,"-o",linewidth=0.6,markersize=1,)
                axes[i].errorbar(self.x_label,mean.iloc[:,i].values,yerr=std.iloc[:,i].values,linewidth=0.3,)            
                legend_mean += [p]
            axes[i].tick_params(labelsize=5)                
            axes[i].set_xscale('log')
            axes[i].legend(legend_mean,labels=self.legend_label,loc="upper left",fontsize=3) 
            axes[i].set_title(self.feature[i],fontsize=5)
        if champ:
            fig.suptitle("Dose Response(diff)_"+str(num)+"cells_int"+str(self.interval)+"\n cell id:"+cell_id,fontsize=10)    
        else:
            fig.suptitle("Dose Response(diff)_all_cell_int"+str(self.interval),fontsize=40)
        fig.subplots_adjust()
        
        if champ:
            #fig.savefig(out_dir+"/"+self.expname+"Dose Response(diff)_"+str(num)+"cells_int"+str(self.interval)+".pdf")
            fig.savefig(out_dir+"/"+self.expname+"Dose Response(diff)_"+str(num)+"cells_int"+str(self.interval)+".png",dpi=300)
        else:
            #fig.savefig(out_dir+"/"+self.expname+"Dose Response(diff)_all_cell_int"+str(self.interval)+".pdf")
            fig.savefig(out_dir+"/"+self.expname+"Dose Response(diff)_all_cell_int"+str(self.interval)+".png",dpi=300)
        plt.clf()
        plt.close()        
        

    
    def plot_feature_dose_response(self,df,feature_name="str",fold_change=True):
        out_dir= self.out_dir 
        div_nd = dd.div_label_propaty(df,exp_type=self.exp_type)
        IQR_max = stats.scoreatpercentile(df.max().values.flatten(),75) - stats.scoreatpercentile(df.max().values.flatten(),25)
        IQR_min = stats.scoreatpercentile(df.min().values.flatten(),75) - stats.scoreatpercentile(df.min().values.flatten(),25)
        ymax=stats.scoreatpercentile(df.max().values.flatten(),75) + IQR_max*1.5
        ymin=stats.scoreatpercentile(df.min().values.flatten(),25) - IQR_min*1.5
        fig = plt.figure(figsize=(20,20))
        axes = [fig.add_subplot(self.label_num/6+1,6,i+1) for i in range(self.label_num)]   
        for i in range(self.label_num):
            for j in range(len(div_nd)):
                axes[i].plot(self.x_label,div_nd[j][i],"-o",markersize=1,linewidth = 0.6,)
            axes[i].tick_params(labelsize=5)
            axes[i].set_xscale('log')
            axes[i].set_ylim(ymin,ymax)
            axes[i].legend(div_nd,labels=self.legend_label,loc="upper left",fontsize=3) 
            axes[i].set_title("label"+str(i+1),fontsize=5)
        fig.suptitle("Dose Response_int"+str(self.interval)+"_"+feature_name,fontsize=40)
        
        #fig.savefig(out_dir+"/"+feature_name+self.expname+".pdf")
        fig.savefig(out_dir+"/"+feature_name+self.expname+".png",dpi=300)
        plt.clf()        
        plt.close()
        gc.collect()
        
            
    def plot_stream(self,df,title="str",ymax=None,ymin=None):
        out_dir=self.out_dir
        time_seq = np.array(self.intensity_df_list[0].index)
        div_df_list = dd.div_label_stream(df,self.exp_type)
        label_name = df[0].columns
        fig = plt.figure(figsize=(28,70))
        axes = [fig.add_subplot(self.label_num,len(self.x_label),i+1) for i in range(self.label_num*len(self.x_label))]
        IQR_max = stats.scoreatpercentile(pd.concat(df).max().values.flatten(),75) - stats.scoreatpercentile(pd.concat(df).max().values.flatten(),25) 
        IQR_min = stats.scoreatpercentile(pd.concat(df).min().values.flatten(),75) - stats.scoreatpercentile(pd.concat(df).min().values.flatten(),25) 
        if ymax==None:ymax = stats.scoreatpercentile(pd.concat(df).max().values.flatten(),75) + IQR_max*1.5
        if ymin==None:ymin = stats.scoreatpercentile(pd.concat(df).min().values.flatten(),25) - IQR_min*1.5  
        i=0        
        for label in label_name:
            for vol in range(len(self.x_label)):
                line = [l[vol][label].values for l in div_df_list]
                for l in line:
                    axes[i].plot(time_seq,l,linewidth = 0.6)
                axes[i].legend(l,labels=self.legend_label,loc="upper right",fontsize=3)
                axes[i].set_title(label+"_"+str(self.x_label[vol])+"V",fontsize=3)
                axes[i].set_xlim(0,time_seq[-1])
                axes[i].set_ylim(ymin,ymax)
                axes[i].tick_params(labelsize=5)
                i = i+1
        fig.suptitle(title+"_int"+str(self.interval)+"_"+"_stream",fontsize=40)
        #fig.savefig(out_dir+"/"+title+self.expname+".pdf")
        fig.savefig(out_dir+"/"+title+self.expname+".png",dpi=300)
        plt.clf()        
        plt.close()        
        gc.collect()
            
    def save_figure(self):
        
        start=time.time()
        self.plot_cell_population_dose_response(champ=True)
        self.plot_cell_population_dose_response(champ=False)
        print("cell population Does Response graph has outputed!!")
        print(str(int(time.time()-start))+"s")
        
        start=time.time()
        self.plot_feature_dose_response(self.peak_df,"peak")
        self.plot_feature_dose_response(self.auc,"AUC")        
        self.plot_feature_dose_response(self.f_basal_df,"forword_basal")
        self.plot_feature_dose_response(self.b_basal_df,"back_basal")
        self.plot_feature_dose_response(self.b_basal_df/self.f_basal_df,"basal(back_divided_by_forword)")
        self.plot_feature_dose_response(self.auc/self.peak_df,"AUC_divided_by_peak")
        #self.plot_feature_dose_response(self.modified_auc/self.peak_df,"modifyied_AUC_divided_by_peak")
        print("individual cells Does Response graph has outputed!!")
        print("time:"+str(int(time.time()-start))+"s")
        
        start=time.time()                        
        self.plot_feature_dose_response(self.rate_peak_df,"rate_peak")
        self.plot_feature_dose_response(self.rate_auc,"rate_AUC")
        self.plot_feature_dose_response(self.rate_b_basal_df,"rate_back_basal")
        self.plot_feature_dose_response(self.rate_auc/self.rate_peak_df,"rate_AUC_divided_by_peak")
        print("individual cells rate Does Response graph has outputed!!")        
        print("time:"+str(int(time.time()-start))+"s")
        
        start=time.time()                        
        self.plot_feature_dose_response(self.diff_peak_df,"diff_peak")
        self.plot_feature_dose_response(self.diff_auc,"diff_AUC")
        self.plot_feature_dose_response(self.diff_b_basal_df,"diff_back_basal")
        self.plot_feature_dose_response(self.diff_auc/self.diff_peak_df,"diff_AUC_divided_by_peak")
        print("individual cells diff Does Response graph has outputed!!")        
        print("time:"+str(int(time.time()-start))+"s")
        
        start=time.time()                        
        self.plot_feature_dose_response(self.diffsr_peak_df,"diffSerise_peak")
        self.plot_feature_dose_response(self.diffsr_auc,"diffSerise_AUC")
        self.plot_feature_dose_response(self.diffsr_b_basal_df,"diffSerise_back_basal")
        self.plot_feature_dose_response(self.diffsr_auc/self.diffsr_peak_df,"diffSerise_AUC_divided_by_peak")
        print("individual cells diffseries Does Response graph has outputed!!")        
        print("time:"+str(int(time.time()-start))+"s")
  
        start=time.time()                
        self.plot_stream(self.intensity_df_list,"stream_raw")
        print("individual cells wave graph has outputed!!")
        print("time:"+str(int(time.time()-start))+"s") 

        start=time.time()
        self.plot_stream(self.rate_df_list,"stream_rate")
        print("individual cells rate wave graph has outputed!!")
        print("time:"+str(int(time.time()-start))+"s")        
        
        start=time.time()
        self.plot_stream(self.diff_df_list,"stream_diff")
        print("individual cells diff wave graph has outputed!!")
        print("time:"+str(int(time.time()-start))+"s")        
        
        start=time.time()
        self.plot_stream(self.diffSeries_df_list,"stream_diffSeries",ymax=100,ymin=-50)
        print("individual cells diffSeries wave graph has outputed!!")
        print("time:"+str(int(time.time()-start))+"s") 
        
        plt.imsave(fname=self.out_dir+"/relabeled_image.png",arr=self.labelimage)
        gc.collect()        
        
    def save_series_csv(self,df,dir_name):
        csv_dir = self.out_dir+"/csv"
        try:os.mkdir(csv_dir)
        except:print("dir has existed already")
        csv_dir = csv_dir + "/" + dir_name         
        try:os.mkdir(csv_dir)
        except:print("dir has existed already")
        div_intensity = dd.div_label_stream(df,exp_type=self.exp_type)        
        repeat=1
        for df_rep in div_intensity:
            csv_repeat_dir=csv_dir+"/repeat_"+str(repeat)
            try:os.mkdir(csv_repeat_dir)
            except:print("dir has existed already")  
            repeat += 1            
            for df_vol,voltage in zip(df_rep,self.x_label):
                csv_repeat_vol_dir = csv_repeat_dir+"/"+str(voltage)+"V"
                df_vol.to_csv(csv_repeat_vol_dir+".csv")          
                df_vol.to_excel(csv_repeat_vol_dir+"_"+self.expname+"_"+".xlsx")

    
    def save_feature_csv(self,df,dirname):
        csv_dir = self.out_dir+"/csv"
        try:os.mkdir(csv_dir)
        except:print("dir has existed already")
        csv_dir = csv_dir + "/feature"          
        try:os.mkdir(csv_dir)
        except:print("dir has existed already")
        df.to_csv(csv_dir + "/" + self.expname +"_" + dirname +".csv")
        df.to_excel(csv_dir + "/" + self.expname +"_" + dirname +".xlsx")
        
    def save_csv(self):
        label_list = list(self.intensity_df_list[0].columns)
        label = list(range(len(label_list)))
        time_list = list(self.intensity_df_list[0].index)
        repeat_num = len(self.intensity_df_list)
        All_cell_list = []
        for num in range(repeat_num):
            cell_list = [self.intensity_df_list[num][x] for x in label_list]
            cell_list = [pd.DataFrame(x) for x in cell_list]
            for i,j in zip(cell_list,label):
                i.columns = ["intensity"]
                i["time"] = time_list
                i["Voltage"] = self.Vol_list[num]
                i["ID"] = j
                i["repeat"] = num
            if num == 0:
                All_cell_list = cell_list
            else :
                for i in range(len(All_cell_list)):
                    All_cell_list[i] = pd.concat([All_cell_list[i],cell_list[i]])
        self.all_df = pd.concat(All_cell_list)
        self.all_df["stim"] = 210
        self.all_df.to_csv(self.out_dir+"/"+self.expname+".csv")
                
                
        
        """
        self.save_series_csv(self.intensity_df_list,"raw")
        self.save_series_csv(self.rate_df_list,"rate")
        self.save_series_csv(self.diff_df_list,"diff")
        self.save_series_csv(self.diffSeries_df_list,"diffSeries")
        
        self.save_feature_csv(self.f_basal_df,"basal")
        """
    #save_smart_CSV(self):
     #   self.intensity_df_list