# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 18:15:15 2017

@author: waizoo
"""

from scipy import signal
import numpy as np
import matplotlib.pyplot as plt

def get_parame(x,y):
    
    """dt = サンプリング周期[Hz]
       fp = 通過域端周波数[Hz]
       gpasss = 通過域最大損失量[dB]
       gstop = 阻止域最小減衰量[dB]
    """
    n = len(x)
    dt = (np.max(x)-np.min(x))/n
    fn = 1/(2*(dt))    #ナイキスト周波数
     
    fp = 1/50
    fs = 1/20
    gpass = 1
    gstop = 40
     #正規化
    Wp = fp/fn
    Ws = fs/fn
    
    return(gpass,gstop,Wp,Ws)
     
def butter(x,y,fig=False):
    gpass,gstop,Wp,Ws = get_parame(x,y)
    N, Wn = signal.buttord(Wp,Ws,gpass,gstop)
    b1, a1 = signal.butter(N,Wn,"low")
    y_low = signal.filtfilt(b1,a1,y)
    if fig:
        plt.figure()
        plt.plot(x,y)
        plt.plot(x,y_low)
        plt.show()
        print("butter")
    return(y_low)
    

