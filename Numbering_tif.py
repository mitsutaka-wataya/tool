# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 21:42:54 2016

@author: waizoo
"""

import re
import glob
import os

def add_vol1(num):
    if num == 1:
        return("Stream_" + numbering(num) + "_0V.tif" )
    if num == 2:
        return("Stream_" + numbering(num) + "_0V.tif" )
    if num == 3:
        return("Stream_" + numbering(num) + "_0V.tif" )
    if num == 4:
        return("Stream_" + numbering(num) + "_3V.tif" )
    if num == 5:
        return("Stream_" + numbering(num) + "_3V.tif" )
    if num == 6:
        return("Stream_" + numbering(num) + "_3V.tif" )
    if num == 7:
        return("Stream_" + numbering(num) + "_10V.tif" )
    if num == 8:
        return("Stream_" + numbering(num) + "_10V.tif" )
    if num == 9:
        return("Stream_" + numbering(num) + "_10V.tif" )
    if num == 10:
        return("Stream_" + numbering(num) + "_20V.tif" )
    if num == 11:
        return("Stream_" + numbering(num) + "_20V.tif" )
    if num == 12:
        return("Stream_" + numbering(num) + "_20V.tif" )
    if num == 13:
        return("Stream_" + numbering(num) + "_30V.tif" )
    if num == 14:
        return("Stream_" + numbering(num) + "_30V.tif" )
    if num == 15:
        return("Stream_" + numbering(num) + "_30V.tif" )
    if num == 16:
        return("Stream_" + numbering(num) + "_50V.tif" )
    if num == 17:
        return("Stream_" + numbering(num) + "_50V.tif" )
    if num == 18:
        return("Stream_" + numbering(num) + "_50V.tif" )
    if num == 19:
        return("Stream_" + numbering(num) + "_100V.tif" )
    if num == 20:
        return("Stream_" + numbering(num) + "_50V.tif" )
    if num == 21:
        return("Stream_" + numbering(num) + "_50V.tif" )
    if num == 22:
        return("Stream_" + numbering(num) + "_50V.tif" )
    if num == 23:
        return("Stream_" + numbering(num) + "_30V.tif" )
    if num == 24:
        return("Stream_" + numbering(num) + "_30V.tif" )
    if num == 25:
        return("Stream_" + numbering(num) + "_30V.tif" )
    if num == 26:
        return("Stream_" + numbering(num) + "_20V.tif" )
    if num == 27:
        return("Stream_" + numbering(num) + "_20V.tif" )
    if num == 28:
        return("Stream_" + numbering(num) + "_20V.tif" )
    if num == 29:
        return("Stream_" + numbering(num) + "_10V.tif" )
    if num == 30:
        return("Stream_" + numbering(num) + "_10V.tif" )
    if num == 31:
        return("Stream_" + numbering(num) + "_10V.tif" )
    if num == 32:
        return("Stream_" + numbering(num) + "_3V.tif" )
    if num == 33:
        return("Stream_" + numbering(num) + "_3V.tif" )
    if num == 34:
        return("Stream_" + numbering(num) + "_3V.tif" )
    if num == 35:
        return("Stream_" + numbering(num) + "_0V.tif" )
    if num == 36:
        return("Stream_" + numbering(num) + "_0V.tif" )
    if num == 37:
        return("Stream_" + numbering(num) + "_0V.tif" )

 


def add_vol2(num):
    if num%24 == 1:
        return("Stream_" + numbering(num) + "_0V.tif" )
    if num%24 == 2:
        return("Stream_" + numbering(num) + "_3V.tif" )
    if num%24 == 3:
        return("Stream_" + numbering(num) + "_5V.tif" )
    if num%24 == 4:
        return("Stream_" + numbering(num) + "_10V.tif" )
    if num%24 == 5:
        return("Stream_" + numbering(num) + "_15V.tif" )
    if num%24 == 6:
        return("Stream_" + numbering(num) + "_20V.tif" )
    if num%24 == 7:
        return("Stream_" + numbering(num) + "_25V.tif" )
    if num%24 == 8:
        return("Stream_" + numbering(num) + "_30V.tif" )
    if num%24 == 9:
        return("Stream_" + numbering(num) + "_35V.tif" )
    if num%24 == 10:
        return("Stream_" + numbering(num) + "_40V.tif" )
    if num%24 == 11:
        return("Stream_" + numbering(num) + "_50V.tif" )
    if num%24 == 12:
        return("Stream_" + numbering(num) + "_75V.tif" )
    if num%24 == 13:
        return("Stream_" + numbering(num) + "_100V.tif" )
    if num%24 == 14:
        return("Stream_" + numbering(num) + "_75V.tif" )
    if num%24 == 15:
        return("Stream_" + numbering(num) + "_50V.tif" )
    if num%24 == 16:
        return("Stream_" + numbering(num) + "_40V.tif" )
    if num%24 == 17:
        return("Stream_" + numbering(num) + "_35V.tif" )
    if num%24 == 18:
        return("Stream_" + numbering(num) + "_30V.tif" )
    if num%24 == 19:
        return("Stream_" + numbering(num) + "_25V.tif" )
    if num%24 == 20:
        return("Stream_" + numbering(num) + "_20V.tif" )
    if num%24 == 21:
        return("Stream_" + numbering(num) + "_15V.tif" )
    if num%24 == 22:
        return("Stream_" + numbering(num) + "_10V.tif" )
    if num%24 == 23:
        return("Stream_" + numbering(num) + "_5V.tif" )
    if num%24 == 0:
        return("Stream_" + numbering(num) + "_3V.tif" )
        
def add_vol3(num):
    if num%23 == 1:
        return("Stream_" + numbering(num) + "_0V.tif" )
    if num%23 == 2:
        return("Stream_" + numbering(num) + "_5V.tif" )
    if num%23 == 3:
        return("Stream_" + numbering(num) + "_10V.tif" )
    if num%23 == 4:
        return("Stream_" + numbering(num) + "_15V.tif" )
    if num%23 == 5:
        return("Stream_" + numbering(num) + "_20V.tif" )
    if num%23 == 6:
        return("Stream_" + numbering(num) + "_25V.tif" )
    if num%23 == 7:
        return("Stream_" + numbering(num) + "_30V.tif" )
    if num%23 == 8:
        return("Stream_" + numbering(num) + "_35V.tif" )
    if num%23 == 9:
        return("Stream_" + numbering(num) + "_40V.tif" )
    if num%23 == 10:
        return("Stream_" + numbering(num) + "_50V.tif" )
    if num%23 == 11:
        return("Stream_" + numbering(num) + "_75V.tif" )
    if num%23 == 12:
        return("Stream_" + numbering(num) + "_100V.tif" )
    if num%23 == 13:
        return("Stream_" + numbering(num) + "_75V.tif" )
    if num%23 == 14:
        return("Stream_" + numbering(num) + "_50V.tif" )
    if num%23 == 15:
        return("Stream_" + numbering(num) + "_40V.tif" )
    if num%23 == 16:
        return("Stream_" + numbering(num) + "_35V.tif" )
    if num%23 == 17:
        return("Stream_" + numbering(num) + "_30V.tif" )
    if num%23 == 18:
        return("Stream_" + numbering(num) + "_25V.tif" )
    if num%23 == 19:
        return("Stream_" + numbering(num) + "_20V.tif" )
    if num%23 == 20:
        return("Stream_" + numbering(num) + "_15V.tif" )
    if num%23 == 21:
        return("Stream_" + numbering(num) + "_10V.tif" )
    if num%23 == 22:
        return("Stream_" + numbering(num) + "_5V.tif" )
    if num%23 == 0:
        return("Stream_" + numbering(num) + "_0V.tif" )

def add_vol4(num):
    if num%14 == 1:
        return("Stream_" + numbering(num) + "_0V.tif" )
    if num%14 == 2:
        return("Stream_" + numbering(num) + "_5V.tif" )
    if num%14 == 3:
        return("Stream_" + numbering(num) + "_10V.tif" )
    if num%14 == 4:
        return("Stream_" + numbering(num) + "_15V.tif" )
    if num%14 == 5:
        return("Stream_" + numbering(num) + "_20V.tif" )
    if num%14 == 6:
        return("Stream_" + numbering(num) + "_25V.tif" )
    if num%14 == 7:
        return("Stream_" + numbering(num) + "_30V.tif" )
    if num%14 == 8:
        return("Stream_" + numbering(num) + "_30V.tif" )
    if num%14 == 9:
        return("Stream_" + numbering(num) + "_25V.tif" )
    if num%14 == 10:
        return("Stream_" + numbering(num) + "_20V.tif" )
    if num%14 == 11:
        return("Stream_" + numbering(num) + "_15V.tif" )
    if num%14 == 12:
        return("Stream_" + numbering(num) + "_10V.tif" )
    if num%14 == 13:
        return("Stream_" + numbering(num) + "_5V.tif" )
    if num%14 == 0:
        return("Stream_" + numbering(num) + "_0V.tif" )
    
def add_all_0V(num):
    return("Stream_" + numbering(num) + "_0V.tif" )
    
def numbering(num):
     if 10>num:
         return("00"+str(num))
     if 100>num and 9<num:
         return("0"+str(num))
     if 1000>num and 99<num:
         return(str(num))
     if 1000>num:
         print("over 1000")
             
def main(path,all0V = False):
    path.replace("\\","/")    
    os.chdir(path)
    origin_name = glob.glob("Stream*.tif")
    pattern = r"[0-9]+"
    #number = [re.search(pattern,i) for i in origin_name]
    #number = [i.group() for i in number]
    for i in origin_name:
        num = re.search(pattern,i)
        if num:        
            num = num.group()  
            num = int(num)
            num = num
            if all0V:
                os.rename(i,add_all_0V(num))
            else:
                os.rename(i,add_vol3(num))
        else:
            num = 1
            os.rename(i,add_vol4(int(num)))
    os.chdir("..")
    os.chdir("..")
#print("fir")
#main(path="C:/Users/waizoo/Desktop/ラボ/結果/161117/3_2_int30s_hoechst")
        
     
                