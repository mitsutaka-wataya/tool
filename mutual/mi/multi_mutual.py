def mutual_allsubsets(response, k, prior):
	"""
    Calulate the values of peripheral mutual information of input v.s. all subsets of responses

    Parallelly computed

    Parameters
    ----------------------
    response: Responses
    k: int. k nearest neighbor parameter
    prior: list. prior distribution
    """
	from ..core import multi_mutual_info
	return multi_mutual_info.mutual_allsubsets(response, k, prior)

def independent_mi(mi_allsubset, response):
	"""
	Calculate all independent multivariate mutual information.

	Parameters
	----------------------
	mi_allsubset:
		The result of the function mutual_allsubsets
	response:
		Responses instance

	Returns
	----------------------
	dict {given variables: (not-given variables, mutual_information)}
	"""
	from ..core import independent_mi
	return independent_mi.multivariate_mi_ind(mi_allsubset, response)

def show_multivariate_ind(multi_mi_ind):
	"""
	show the result of the function independent_mi

	Parameters
	----------------------
	muti_mi_ind:
		The result of independent_mi	
	"""
	from ..core import independent_mi
	return independent_mi.show_multivariate_ind(multi_mi_ind)
