import numpy as np
import pandas as pd

def create_responses(response_list, input_list):
	"""
	create Response instance.

	Parameters
	----------------
	response_list: list or 2-dimensional ndarray
		The list of responses. When multivariate mutual information is to be calculated, 
		nested list whose inner list is composed of some outputs.
	input_list: list
		The list of input. The order must be the same with that of response_list

	ex)
	# response distribution of five responses. The dimension of each response is 1.
	response_list = [1,2,3,4,5]

	# response distribution of five responses. The dimension of each response is 4.
	response_list = [
		[1, 1, 1, 1],
		[2, 2, 2, 2],
		[3, 3, 3, 3],
		[4, 4, 4, 4],
		[5, 5, 5, 5]
	]

	# It is OK that response_list is either list or ndarray
	response_list = numpy.array(response_list)
	"""

	return Responses(response_list, input_list)

class Responses:
	"""
	The class that contains inputs and outputs

	Created by the function, create_responses

	Atrributes
	-------------------
	response:	ndarray
		response list
	input:		list
		input list


	Methods
	-------------------
	select_response:
		Create Responses instance whose dimension is diminished.
	fullset:
		Return the set of all responses
	as_dataframe:
		Return the response as dataframe
	"""
	def __init__(self, response_list, input_list):
		self.response = np.array(response_list)
		self.input = input_list


	def select_response(self, column_v):
		"""
		Create Responses instance whose dimension is diminished..

		Parameters
		---------------
		column_v: int list
			The list of the response column number to be kept.

		ex)
		select_response(celllist, column_v = [0, 1, 3, 4])	# the celllist whose columns are 0, 1, 3, and 4th of the original celllist.
		"""
		return Responses(pd.DataFrame(self.response).loc[:,column_v].values, self.input)
		
	def fullset(self):
		"""
		Return the set of all responses
		"""

		return set(range(self.response.shape[1]))

	def as_dataframe(self):
		"""
		Return the response as dataframe
		"""
		return pd.DataFrame(self.response, index = self.input)
