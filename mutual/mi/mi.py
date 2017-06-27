
def knn_density(responses, k, save = False, filename = 'density.csv'):
	"""
	Create dataframe of conditioned densities of each response by k nearest neighbor.

	Parameters
	-----------------
	responses: Responses instance.
	k: int
		Parameter for k nearest neighbor.
	save: Bool. The default is False
		If True, the result dataframe is saved as csv file.
	filename: str. The default is 'density.csv'
		The name of the csv file. It is used when and only when save is True.

	"""
	from ..core import knn
	return knn.knn_density(responses, k, save, filename)


def mutual_info(response, prior = None, k = None, known = 'prior'):
	"""
	Calculate mutual information of input v.s. the whole of outputs

	Parameters
	----------------
	response: nested list or dataframe
		Response data
		Responses instance when known value is 'prior' or 'nothing'
		density dataframe from knn_density function when known value is 'density' or 'both'
	prior: list
		List whose prior probability. It is required when known is 'prior' or 'both'
	k: int
		Parameter for k nearest neighbor. It is required when known is 'prior' or 'nothing'
	indexlist: list
		Input list which has the same order with response. It is required when known is 'prior' or 'nothing'
	known: str
		The method for calulation. The default is 'prior'
		'nothing': Estimate channel capacity and optimal prior disttribution from celllist and indexlist.
		'density': Estimate channel capacity and optimal prior distribution from knn density.
		'prior': Calulate mutual information with the given prior distribution from celllist and indexlist.
		'both': Calculate mutual information with the given prior distribution from knn density.


	Returns
	----------------
	known value-dependent
	known == 'prior' or 'both':
		mutual information:	float. 
	known == 'density' or 'nothing':
		(x_opt, input_v, channel_capacity)

		x_opt:	list. optimized prior distribution
		input_v:	input list. The order is the same with that of x_opt
		channel_capacity: float. channel capacity, whish is the mutual information when the prior is optimal.

	"""
	from ..core import mutual_infomation
	return mutual_infomation.mutual_info(response, prior, k, known)
