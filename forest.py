import tree
importy numpy as np

class forest:
	
	def __init__(self, dataObj=None, categories=None, trees=[]):
		self.trees = trees
		self.dataObj = dataObj
		self.categories = categories
	
	#build a forest with given number and depth of trees	
	def build(self, train_data, numTrees, depth, z):
		weights = np.ones_like(categories)
		for i in range(numTrees):
			tree = tree.Tree(self.dataObj, self.categories, depth)
			#calculate weight
			
			tree.build(train_data, self.categories, weights, depth, z)
			tree.prune()
			self.trees.append(tree)
			cats = tree.classify(train_data)
			for j in range(len(weights)):
				if cats[i] == self.categories:
					weights[i] *= -1.25
				else:
					weights[i] *= 1.25
		
	def classify(self, dataMatrix):
		cats = 
		for tree in self.trees:
			
