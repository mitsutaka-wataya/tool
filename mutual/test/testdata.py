import numpy as np
from ..framework.responses import create_responses

def testdata(i):
	if i == 1:
		cov = np.eye(2)/2

		z1 = np.random.multivariate_normal([3, 3], cov, 1000)
		z2 = np.random.multivariate_normal([-3, 3], cov, 1000)
		z3 = np.random.multivariate_normal([-3, -3], cov, 1000)
		z4 = np.random.multivariate_normal([3, -3], cov, 1000)
		celllist = np.concatenate((z1, z2, z3, z4))
		indexlist = [0]*1000 + [1]*1000 + [2]*1000 + [3]*1000
		return create_responses(celllist, indexlist)

	elif i == 2:
		cov = np.eye(2)/2

		z1 = np.random.multivariate_normal([3, 3], cov, 1000)
		z2 = np.random.multivariate_normal([-3, -3], cov, 1000)
		z3 = np.random.multivariate_normal([3, 3], cov, 1000)
		z4 = np.random.multivariate_normal([-3, -3], cov, 1000)
		celllist = np.concatenate((z1, z2, z3, z4))
		indexlist = [0]*1000 + [1]*1000 + [2]*1000 + [3]*1000
		return create_responses(celllist, indexlist)

	elif i == 3:
		cov = np.eye(3)/2

		z1 = np.random.multivariate_normal([3, 3, 3], cov, 1000)
		z2 = np.random.multivariate_normal([-3, 3, 3], cov, 1000)
		z3 = np.random.multivariate_normal([3, 3, -3], cov, 1000)
		z4 = np.random.multivariate_normal([-3, -3 ,-3], cov, 1000)
		celllist = np.concatenate((z1, z2, z3, z4))
		indexlist = [0]*1000 + [1]*1000 + [2]*1000 + [3]*1000
		return create_responses(celllist, indexlist)