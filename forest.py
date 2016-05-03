import tree
import numpy as np
import analysis

class forest:
	#initializes a forest, sets z to 1.96 a .95 confidence interval if it is not given
	def __init__(self, dataObj=None, categories=None, trees=[], z=1.96):
		self.trees = trees
		self.dataObj = dataObj
		if self.dataObj != None and categories != None:
			self.build(dataObj, categories, depth, z)
	
	#use Adaboost to build a forest with given depth of trees	
	def build(self, train_data, categories, depth, z):
		data_size = train_data.get_raw_num_rows()
		if data_size != categories.shape[0]:
			print "training data and categories have different sizes."
			return
		weights = np.matrix(np.ones(shape=(data_size, 1)))
		ensemble = np.matrix(np.zeros(shape=(data_size, 1)))
		correctCount = data_size
		treeCount = 0
		while correctCount > 0.5*data_size:
			correctCount = 0
			tree = tree.Tree(train_data, categories, depth, z)
			tree.prune()
			self.trees.append(tree)
			treeCount += 1
			cats = tree.classify(train_data)
			#increase the weight of data points that the previous tree classifies incorrectly
			weights *= 1/data_size
			#calculate error
			E = np.exp(-np.multiply(self.categories, cats))
			error = np.sum(np.multiply(weights, E), axis=0)
			a = 0.5*np.log((1 - error)/error)
			#add to ensemble
			if ensemble.shape[1] == 1:
				ensemble[:, 0] = a*cats
			else:
				ensemble = np.hstack((ensemble, ensemble[;, -1] + a*cats))
			#update weights
			weights = np.multiply(weights, np.exp(-np.multiply(self.categories, cats)*a))
			#normalize weights
			m = np.min(weights)
			M = np.max(weights)
			for i in range(data_size):
				weights[i, 0] -= m
				weights[i, 0] *= 1/(M - m)
			'''
			for j in range(len(weights)):
				if cats[i] == self.categories:
					weights[i] *= 0.75
					correctCount += 1
				else:
					weights[i] *= 1.25
			'''
	
	#classify a given data set and return a matrix of categories	
	def classify(self, dataMatrix):
		cats = np.matrix(np.zeros(shape=(dataMatrix.shape[0], 1)))
		unique, mapping = np.unique(np.array(categories), return_inverse = True)
		forest_cats = np.matrix(np.zeros(shape=(dataMatrix.shape[0], len(self.trees)))
		votes = np.matrix(np.zeros(shape=(dataMatrix.shape[0], len(unique))))
		for t in range(len(self.trees)):
			tree_cats = self.trees[t].classify(dataMatrix)
			forest_cats[:, t] = tree_cats
		for i in range(dataMatrix.shape[0]):
			for j in range(len(self.trees)):
				votes[i, mapping[forest_cats[i, j]]] += 1
			cat = 0
			count = 0
			for k in range(len(unique)):
				if votes[i, k] > count:
					count = votes[i, k]
					cat = unique[k]
			cats[i, 0] = cat
		return cat
		
	#use k-fold cross validation to test the forest
	def test(self, headers, k):
		n = int(self.dataObj.get_raw_num_rows()/k)
		for i in range(k):
			train = self.dataObj.get_data(headers, rows=range(i*n))
			train = np.vstack((train, self.dataObj.get_data(headers, rows=range(i*n + n, k*n))))
			test = self.dataObj.get_data(headers, rows=range(i*n, i*n + n))
			self.build
