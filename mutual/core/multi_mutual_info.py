import pandas as pd
import itertools as it
from . import mutual_infomation as mi
from .base import *

def allsubset(response, include_empty = False):
    fullset = response.fullset()
    sublist = []
    start = 0
    if not include_empty:
        start = 1
    for i in range(start, len(fullset)+1):
        sublist += list(it.combinations(fullset, i))
    return sublist

def mutual_allsubsets(response, k, prior):
    import multiprocessing
    p = multiprocessing.Pool()


    reslist = allsubset(response)
    resultlist = p.map(mi.p_mutual, [[response, res, k, prior] for res in reslist])
    return {v[0]: v[1] for v in resultlist}
