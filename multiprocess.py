# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:33:50 2017

@author: waizoo
"""
from multiprocessing import Pool
from multiprocessing import Process

def list_split(inlist,n):
    listlen = len(inlist)
    size = int(listlen/n +1)
    f = [i*size for i in range(n)]
    result=[inlist[i:i+size] for i in f]
    result[-1] = inlist[f[-1]:]
    return(result)
    
def multi(inlist,n,func):
    if n>6:
        print("too many process. n should be under 6 ")
        return(None)
    else:
        p=Pool(n)
        splited_list = list_split(inlist,n)
        result = p.map(func,splited_list)
        return(result)