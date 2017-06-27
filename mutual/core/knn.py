import numpy as np
import pandas as pd
import scipy.spatial.distance as spd
import math
from .base import *

def knn_distmat(responses, k):
    celllist = responses.response
    indexlist = sorted(responses.input)
    if not hasattr(celllist[0], '__iter__'):
        celllist = np.concatenate((celllist, np.zeros((celllist.shape[0])))).reshape(2, celllist.shape[0]).T

    cellframe = pd.DataFrame(celllist, index = indexlist).sort_index()
    distmat = pd.DataFrame(spd.cdist(cellframe, cellframe, 'Euclidean'), index = indexlist)
    
    distmat[distmat == 0] = np.Inf
    rankmat = distmat.groupby(level=0).rank()
    knn_distmat = distmat[rankmat == k]
    return knn_distmat.groupby(level = 0).sum()


def knn_density(responses, k, save = False, filename = 'density.csv'):
    celllist = responses.response
    indexlist = sorted(responses.input)
    if hasattr(celllist[0], '__iter__'):
        dim = len(celllist[0])
    else:
        dim = 1
    unitball = (math.pi ** (dim/2)) / math.gamma(dim / 2 +1)
    volume = unitball * np.power(knn_distmat(responses, k), dim)
    density = (k / (numCell_ineachIndex(indexlist).reshape(numIndex(indexlist), 1) * volume))
    density = density.T
    indexlist = sorted(indexlist)
    density.index = indexlist
    if save:
        density.to_csv(filename, header = False)
    return density