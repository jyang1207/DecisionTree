import tree
importy numpy as np

class forest:
	
	def __init__(self, dataObj=None, categories=None, trees=[], z=0):
		self.trees = trees
		self.dataObj = dataObj
		if self.dataObj != None:
			self.build(dataObj, categories, depth, z)
	
	#use Adaboost to build a forest with given depth of trees	
	def build(self, train_data, categories, depth, z):
		data_size = train_data.get_raw_num_rows()
		if data_size != categories.shape[0]:
			print "training data and categories have different sizes."
			return
		weights = np.ones_like(categories)
		correctCount = data_size
		while correctCount > 0.5*data_size:
			correctCount = 0
			tree = tree.Tree(train_data, categories, depth, z)
			tree.prune()
			self.trees.append(tree)
			cats = tree.classify(train_data)
			#increase the weight of data points that the previous tree classifies incorrectly
			for j in range(len(weights)):
				if cats[i] == self.categories:
					weights[i] *= 0.75
					correctCount += 1
				else:
					weights[i] *= 1.25
	
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
		
