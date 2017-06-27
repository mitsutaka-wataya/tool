import numpy as np
import pandas as pd

def numIndex(indexlist):
    return len(set(indexlist))

def numCell_ineachIndex(indexlist):
    return np.array([indexlist.count(value) for value in list(set(indexlist))])

def uniqueindex(indexlist):
    return list(set(indexlist))
