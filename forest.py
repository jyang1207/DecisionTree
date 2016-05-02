import tree
importy numpy as np

class forest:
	
	def __init__(self, dataObj=None, categories=None, trees=[], z=0):
		self.trees = trees
		
		if self.dataObj != None:
			self.build(dataObj, categories, depth, z)
	
	#build a forest with given number and depth of trees	
	def build(self, train_data, categories, depth, z):
		if train_data.shape[0] != categories.shape[0]:
			print "training data and categories have different sizes."
			return
		weights = np.ones_like(self.categories)
		correctCount = train_data.shape[0]
		while correctCount > 0.5*train_data.shape[0]:
			correctCount = 0
			tree = tree.Tree(self.dataObj, self.categories, depth)
			tree.build(train_data, self.categories, weights, depth, z)
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
		
	def classify(self, dataMatrix):
		cats = 
		for tree in self.trees:
			
