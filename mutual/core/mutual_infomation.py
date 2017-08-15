import numpy as np
import pandas as pd
import math
import scipy.optimize as opt
import scipy.spatial.distance as spd

from .base import *
from .knn import knn_density

def H_RS(density, prior):
    entropy = -np.log2(density)
    numCell_ineach = density.groupby(level= 0).count()[0].values
    return (np.diag(entropy.groupby(level = 0).sum().values) * prior / numCell_ineach).sum()

def H_R(density, prior):
    entropy = -np.log2((density * prior).sum(axis = 1))
    numCell_ineach = density.groupby(level= 0).count()[0].values
    return (entropy.groupby(level=0).sum() * prior / numCell_ineach).sum()

def mutual_with_knowndensity(density, prior):
    return H_R(density, prior) - H_RS(density, prior)

def mutual_memorysaved(responses, k, prior):
    cellframe = responses.as_dataframe().sort_index()
    indexlist = sorted(responses.input)
    numIdx = numIndex(indexlist)
    u_idx = uniqueindex(indexlist)
    numCells = numCell_ineachIndex(indexlist)
    del indexlist
    Hrs = 0
    Hr = 0
    tmpHr = []
    offset = 0
    distlist = []

    if hasattr(responses.response[0], '__iter__'):
        dim = len(responses.response[0])
    else:
        dim = 1
    unitball = (math.pi ** (dim/2)) / math.gamma(dim / 2 +1)


    for idx in range(numIdx):
        dist = pd.DataFrame(spd.cdist(cellframe.ix[u_idx[idx]], cellframe, 'euclidean'))
        dist = dist.where(dist > 0, np.Inf)
        dist = (np.equal(dist.rank(), k) * dist).sum()
    
        density = k / (numCells[idx] * (unitball * np.power(dist, dim)))
        del dist

        Hrs += -np.log2(density[offset:offset + numCells[idx]]).sum()*prior[idx]/numCells[idx]
        tmpHr += [density*prior[idx]]
        del density
        offset += numCells[idx]

        
    tmpHr = -np.log2(np.sum(pd.DataFrame(tmpHr), axis=0))
    offset = 0
    for idx in range(numIdx):
        Hr += tmpHr[offset:offset+numCells[idx]].sum()*prior[idx]/numCells[idx]
        offset += numCells[idx]
    return Hr - Hrs


def mutual_info(a, prior = None, k = None,  known = 'prior'):

	if known == 'prior':
		if (k is None) or (prior is None):
			raise ValueError('k (int) and prior (list or array) must be specified when prior known method is chosen.')
		return mutual_memorysaved(a, k, prior)

	elif known == 'density':
		return optim_prior(a)
	
	elif known == 'both':
		if prior is None:
			raise ValueError('prior (list or array) must be specified when prior known method is chosen.')
		return mutual_with_knowndensity(a, prior)

	elif known == 'nothing':
		if k is None:
			raise ValueError('k (int) must be specified when prior known method is chosen.')
		return optim_prior(knn_density(a, k))


def optim_prior(density, hist = True, accuracy = 1e-06, epsilon=1.4901161193847656e-08, maxiter=100):
    numX = len(set(list(density.index)))
    obj = lambda x:-mutual_with_knowndensity(density = density, prior = x)
    x0 = np.ones(numX)/numX
    def eq_con(x):
        return x.sum()-1

    def ieq_con(x):
        return x
    x_opt = opt.fmin_slsqp(obj, x0, f_eqcons = eq_con, f_ieqcons = ieq_con, acc = accuracy, epsilon = epsilon, iter = maxiter)
    channel_capacity =mutual_with_knowndensity(density, x_opt)
    
    input_v = list(set(list(density.index)))
    x_opt_df = pd.DataFrame(x_opt, index = input_v)
    
    if hist:
        x_opt_df.plot.bar(legend = False, ylim = (0,1))
    return x_opt, input_v, channel_capacity


def _mutual(responses, res, k, prior):
    return (res), mutual_memorysaved(responses.select_response(res), k, prior)

def p_mutual(argv):
    return _mutual(*argv)