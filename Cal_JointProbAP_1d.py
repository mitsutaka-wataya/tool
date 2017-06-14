# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 15:56:36 2017

@author: test
"""


import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt


"""
cal_JointProbAP_1d
make_prob
range2idx
area_sep
bin_counter
Chi_test]
"""

def cal_JointProbAP_1d(X,m,xran,plothist = False):
    d=Depth(X=X,xran=xran,m=m)
    d.plot_hist()
    for i in range(m-1):
        d=Depth(pre_depth=d)
        if plothist:
            d.plot_hist()
            print("N_defctive_rate:"+str(np.array(d.flag).sum()/len(d.flag)))
        if d.flag.count(1)==0:
            break
    ret_prob = d.ret_prob()
    ret_edge = d.x_edges
    N_total = d.X.shape[0]
    N_defctive_rate = np.array(d.flag).sum()/len(d.flag)
    return(ret_prob,ret_edge,N_total,N_defctive_rate)

def area_sep(X,x_range,n_sep):
    
    x_edge = np.linspace(x_range[0],x_range[1],n_sep+1)
    
    N_count = np.zeros(n_sep)
    for a in range(len(x_edge)-1):
        xtemp = [x_edge[a],x_edge[a+1]]
        N_count[a] = bin_counter(X,xtemp)
    return(N_count)
        
def bin_counter(X,x_range):
    temp = np.array(X)
    x_low = temp[temp >=x_range[0]]
    x_low_upper = x_low[x_low<x_range[1]]
    return(x_low_upper.size)

def range2idx(x_range,x_edge):
    #真ん中で切ったBin幅と近いサンプリング点を求める。
    low = np.abs(x_edge-x_range(0))
    idx_low = low.argmin()
    upper = np.abs(x_edge-x_range(1))
    idx_upper = upper.argmin()
    x_idx = (idx_low,idx_upper-1)
    return(x_idx)

def Chi_test(N_InBins,alpha):
    
    N_total = N_InBins.sum()
    if N_total == 0:
        retH = 0
        return(retH)
    n_bin = len(N_InBins)
    expe = N_total/n_bin #1つのBinに入っているサンプル数の期待値
    #print(expe)
    #chi2　統計量の算出
    temp = (N_InBins-expe)**2 / (expe)
    chi2 = temp.sum()
    threshold = st.chi2.ppf(1-alpha,n_bin-1)
    #print(chi2)
    #print(threshold)
    if chi2 <= threshold:
        retH = 0#H0 is accepted
    else:
        retH = 1#H0 is rejected
    return(retH)
    
class Depth(object):
    def __init__(self,X=None,xran=None,pre_depth=None,m=None):
        self.alpha = 0.1
        if pre_depth == None:
            self.part = [xran] #binの端
            self.flag = [1] #1:まだ分けられる 0:分けられない
            self.X = X
            self.m=m
            self.x_edges = np.linspace(xran[0],xran[1],2**(m)+1)
            self.N_bin = bin_counter(X=self.X,x_range=xran)
            print("initialized")
        else:
            pre_part = pre_depth.part
            pre_flag = pre_depth.flag
            self.flag = []
            self.part = []
            self.X = pre_depth.X
            self.m = pre_depth.m
            self.x_edges = pre_depth.x_edges
            for i in range(len(pre_flag)):
                xmin = pre_part[i][0]
                xmax = pre_part[i][1]
                xmid = (xmin+xmax)*0.5
                if pre_flag[i] == 1:
                    #print("sepatrete")
                    
                    H0 = self.separate(xmin,xmax)
                    #print(H0)
                    if H0 == 1:
                        self.flag += [1,1]
                        self.part += [(xmin,xmid),(xmid,xmax)]
                    elif H0 == 0:
                        self.flag += [0]
                        self.part += [(xmin,xmax)]
                    
                else:
                    self.flag += [0]
                    self.part += [(xmin,xmax)]
        
        #print(self.flag)
                    
    def separate(self,xmin,xmax):
        N_CountThere = area_sep(self.X,(xmin,xmax),2)
        res1_chi = Chi_test(N_CountThere,self.alpha)
        N_CountThere = area_sep(self.X,(xmin,xmax),4)
        res2_chi = Chi_test(N_CountThere,self.alpha)
        res0_chi = res1_chi*res2_chi
        return(res0_chi)

   
    def range2idx(self,x_range):
        #Binと近いサンプリング点のIndexをX_edegsから求める。
        low = np.abs(self.x_edges-x_range[0])
        idx_low = low.argmin()
        upper = np.abs(self.x_edges-x_range[1])
        idx_upper = upper.argmin()
        x_idx = (idx_low,idx_upper)
        return(x_idx)
    
    def ret_bins(self):
        #サンプル数n+1のビン端を返す
        x_idx = [self.range2idx(i) for i in self.part]
        ret = [self.x_edges[i[0]] for i in x_idx]
        ret += [self.x_edges[-1]+0.0001]
        self.bins = np.array(ret)
        return(self.bins)
    
    def ret_prob(self):
        bins = self.ret_bins()
        x_idx = [self.range2idx(i) for i in self.part]
        ret_prob = np.zeros(len(self.x_edges))
        for i in range(len(self.part)):
            NinBin = bin_counter(X=self.X,x_range=(bins[i],bins[i+1]))
            ret_prob[x_idx[i][0]:x_idx[i][1]] = NinBin
        bin_len = np.abs(self.x_edges[0]-self.x_edges[1])
        self.prob = ret_prob/(ret_prob.sum()*bin_len)
        return(self.prob)
    
    def plot_hist(self):
        bins=self.ret_bins()
        plt.hist(x=self.X,bins=bins,histtype="step")
        bin_num = len(bins)
        samplesize = self.X.shape[0]
        plt.title("bin_num:"+str(bin_num)+"\nsample size:"+str(samplesize))
        plt.show()


if __name__ == "__main__":
    m =20 #bin数=2^(m-1)
    z1=np.random.normal(loc=-5,scale=1,size=100000)
    z2=np.random.normal(loc=5,scale=3,size=1000)
    z=np.hstack((z1,z2))
    z=z.reshape(-1) #二峰性の混合ガウス分布
    
    z_ran = (z.min(),z.max()) #サンプルの最大、最小のタプル
    prob,x_edge,N_total,N_defective_rate = cal_JointProbAP_1d(z,m,z_ran,plothist=True)
    """
    返り値
    x_edge　ビン幅の端。2^(m-1)+1 のnumpy arrayが返り値
    prob　確率。2^(m-1) のnumpy arrayが返り値
    N_total,サンプルサイズ
    N_defective_rate　まだ分割できる部分の割合。
    """