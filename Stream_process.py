# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:15:07 2016

@author: test
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
from skimage import io
import os
from scipy import signal
import matplotlib
import ijroi.ijroi
import re
#matplotlib.style.use('ggplot')

#直下のディレクトリにあるdataからファイルを読み込みます。
#ファイル名は　C2C12_GCaMP_(電圧)V_(通し番号n=0~) で固定です。
#ImageJのROIファイルがdata直下にzipとしておいてある想定です。
class ReadFile(object):

    def __init__(self,data_dir,Stream=True,get_roi =None):
        self.dir = data_dir
        data_dir = data_dir.replace("\\","/")
        if Stream == True:      #Stream を分割
            self.split_stream(data_dir)

        self.splited_dir  = glob.glob("split/*")       #data/splitフォルダにある分割tifからパスを取得
        self.splited_dir = [name.replace("\\","/") for name in self.splited_dir]

        if get_roi:        
            self.roi = []
            self.get_roi()     #roiの取得(imageJのzipから)
            self.roi_header = ["roi"+str(i+1) for i in  range(len(self.roi))]
            #self.roi_header[-1] = "background"
            self.roi_header = tuple(self.roi_header)        
            for dir in self.splited_dir:        #各Streamから輝度値のCSVファイルを生成
                self.get_stream_roi_area(dir)
        #image[20][y0:yend,x0:xend] = 0
        #plt.imshow(image[20])
        #plt.show()

    def split_stream(self,data_dir):
        fname = glob.glob(data_dir+'/*V.tif')
        #print("\n fnname:" + fname[0])        
        fnum = len(fname)
        absolute_file_pass = [fname[i].replace("\\", "/") for i in range(fnum)]
        fname = [fname[i].split("\\")[-1] for i in range(fnum)]
        fname = [fname[i].split("_")[-2] + "_" + fname[i].split("_")[-1] for i in range(fnum)]
        try:os.mkdir(data_dir+"/split")
        except:print("split has existed aleady")
            
        for stream_pass in absolute_file_pass:
            save_dir_pass = stream_pass.split(".")[0]
            streame_file_name = stream_pass.split("/")[-1]
            streame_file_name = streame_file_name.split(".")[0]
            try:os.mkdir(data_dir + "/split/" + streame_file_name)
            except:print("dirctry has existed aleady")
            read_stream_image = io.imread(stream_pass)
            #print(len(read_stream_image))
            i=0
            for divided_image in read_stream_image:
                if i <10:
                    save_file_name = streame_file_name + "_0000"  + str(i) + ".tif"
                elif i > 9 and i < 100  :
                    save_file_name = streame_file_name + "_000" + str(i) + ".tif"
                elif i > 99 and i < 1000:
                    save_file_name = streame_file_name + "_00" + str(i) + ".tif"
                elif i > 999 and i < 10000:
                    save_file_name = streame_file_name + "_0" + str(i) + ".tif"
                elif i > 9999 and i < 100000:
                    save_file_name = streame_file_name + "_" + str(i) + ".tif"
                else:
                    print("error.you need files to decrease less than 99999.")
                #else :                 
                #    print("i:"+str(i)+" / some error occured.")
                i =i + 1
                io.imsave(data_dir+"/split/"+streame_file_name+ "/" + save_file_name ,divided_image)
                #print(save_dir_pass+"/"+save_file_name)

    def get_roi(self):
        read_roi = ijroi.read_roi_zip(glob.glob('*.zip')[0])
        read_roi = [read_roi[i][1] for i in range(len(read_roi))]
        for sub_roi in read_roi:
            y0 = int(sub_roi[0, 0])
            yend = int(sub_roi[2, 0])
            x0 = int(sub_roi[0, 1])
            xend = int(sub_roi[2, 1])
            self.roi += [(y0, yend, x0, xend)]
        #print(self.roi)
        #print(len(self.roi))

    def get_roi_area(self,image):
        roi_value =[np.mean(image[cut_area[0]:cut_area[1], cut_area[2]:cut_area[3]]) for cut_area in self.roi]
        return (roi_value)

    #streamを分割したファイルの有るフォルダのディレクトリを受取り、ROIの輝度の平均値をリストで返します。
    def get_stream_roi_area(self,stream_data_dir):
        image_set_pass = glob.glob(stream_data_dir+"/*.tif")
        image_set_pass = [image_pass.replace("\\","/") for image_pass in image_set_pass]
        print(image_set_pass)
        #image_set = [image.split("/")[-1] for image in image_set]
        #image_set = [image.split(".")[0] for image in image_set]
       # image_set_matrix = np.zeros((len(image_set_pass), len(self.roi)))
        try:
            os.mkdir("csv")
        except:
            print("Aleady \"csv folda\" has existed")
        image_set_matrix = [self.get_roi_area(io.imread(image)) for image in image_set_pass]
       # np.hstack([zip(self.roi_header), image_set_matrix])
        csv_name = stream_data_dir.split("/")[-1]
        pcsv = pd.DataFrame(image_set_matrix,columns= self.roi_header)
        #pcsv["Voltage"] = self.get_voltage(csv_name)
        pcsv.to_csv("csv/"+csv_name+".csv")
        #print(pcsv)
        #print(image_set_matrix)

    def get_voltage(self,fname):
        pattern1 = r"[0-9]*V"
        result1 = re.search(pattern1,fname)
        pattern2 = r"[0-9]*"
        result2 = re.search(pattern2,result1.group())
        return(int(result2.group()))

    def get_all_stream_roi_area(self):
        stream_data_dir = glob.glob("split/*")
        stream_data_dir.replace("\\", "/")
        for stream_dir in stream_data_dir:
            self.get_stream_roi_area(stream_dir)

"""
dir = glob.glob("./data/*")
dir = [i.replace("\\","/") for i in dir]
#test=ReadFile(data_dir = dir[0])
obj = [ReadFile(i) for i in dir]
print("finish!!")
"""