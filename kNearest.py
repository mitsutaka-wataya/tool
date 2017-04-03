from sklearn.neighbors import NearestNeighbors
import math
# -*- coding: utf-8 -*-

class kNearest:
# 時系列配列の配列を与えるとk近傍を求め密度推定を行う。
#返り値はすべてそのタイムコースのインデックスをkeyとし、k近傍までの距離や密度などをvalueとする辞書型

	def __init__(self, k, timecourselist):
		self.dim = len(timecourselist[0])
		self.n = len(timecourselist)
		self.k = k
		neighbors = NearestNeighbors(n_neighbors = k+1 , algorithm = "ball_tree").fit(timecourselist)
		self.distances, self.indices = neighbors.kneighbors(timecourselist)

	def index(self):
		return {index: k_neighbor_index for index, k_neighbor_index in zip(self.indices[:,0], self.indices[:,self.k])}

	def distance(self):
		return {index: k_neighbor_distance for index, k_neighbor_distance in zip(self.indices[:,0], self.distances[:, self.k])}

	def volume(self):
		unit_n_ball_volume = math.pi ** (self.dim / 2) / math.gamma(self.dim / 2 + 1)
		return {index: unit_n_ball_volume * self.distance()[index] ** self.dim for index in self.indices[:,0]}

	def density(self):
		return {index: self.k / self.volume()[index] / self.n for index in self.indices[:,0]}