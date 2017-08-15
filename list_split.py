# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 14:33:50 2017

@author: waizoo
"""

def list_split(inlist,n):
    listlen = len(inlist)
    size = int(listlen/n +1)
    f = [i*size for i in range(n)]
    result=[inlist[i:i+size] for i in f]
    result[-1] = inlist[f[-1]:]
    return(result)