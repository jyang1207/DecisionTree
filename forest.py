import tree
import numpy as np
import analysis

class Forest:
	#initializes a forest, sets z to 1.96 a .95 confidence interval if it is not given
	def __init__(self, dataMatrix=None, categories=None, depth=20, trees=[], z=1.96):
		self.trees = trees
		self.dataMatrix = dataMatrix
		if self.dataMatrix != None and categories != None:
			self.build(dataMatrix, categories, depth, z)
	
	#use Adaboost to build a forest with given depth of trees	
	def build(self, train_data, categories, depth=20, z=1.96):
		data_size = train_data.shape[0]
		if data_size != categories.shape[0]:
			print "training data and categories have different sizes."
			return
		weights = np.matrix(np.ones(shape=(data_size, 1)))
		ensemble = np.matrix(np.zeros(shape=(data_size, 1)))
		correctCount = data_size
		treeCount = 0
		while correctCount > 0.5*data_size:
			correctCount = 0
			t = tree.Tree(self.dataObj, categories, depth, z)
			t.prune()
			self.trees.append(t)
			treeCount += 1
			cats = tree.classify(train_data)
			#increase the weight of data points that the previous tree classifies incorrectly
			weights *= 1/data_size
			#calculate error
			E = np.exp(-np.multiply(categories, cats))
			error = np.sum(np.multiply(weights, E), axis=0)
			a = 0.5*np.log((1 - error)/error)
			#add to ensemble
			if ensemble.shape[1] == 1:
				ensemble[:, 0] = a[0,0]*cats
			else:
				ensemble = np.hstack((ensemble, ensemble[:, -1] + a[0,0]*cats))
			#update weights
			weights = np.multiply(weights, np.exp(-np.multiply(categories, cats)*a[0,0]))
			#normalize weights
			m = np.min(weights)
			M = np.max(weights)
			for i in range(data_size):
				weights[i, 0] -= m
				weights[i, 0] *= 1/(M - m)
	
	#classify a given data set and return a matrix of categories	
	def classify(self, dataMatrix):
		cats = np.matrix(np.zeros(shape=(dataMatrix.shape[0], 1)))
		unique, mapping = np.unique(np.array(categories), return_inverse = True)
		forest_cats = np.matrix(np.zeros(shape=(dataMatrix.shape[0], len(self.trees))))
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
	def test(self, categories, headers, k):
		n = int(self.dataObj.get_raw_num_rows()/k)
		for i in range(k):
			train = self.dataObj.get_data(headers, rows=range(i*n))
			train = np.vstack((train, self.dataObj.get_data(headers, rows=range(i*n + n, k*n))))
			test = self.dataObj.get_data(headers, rows=range(i*n, i*n + n))
			forest = Forest(categories)
