import tree
importy numpy as np

class forest:
	
	def __init__(self, dataObj=None, categories=None, trees=[]):
		self.trees = trees
		self.dataObj = dataObj
		self.categories = categories
	
	#build a forest with given number and depth of trees	
	def build(self, train_data, numTrees, depth, z):
		for i in range(numTrees):
			tree = tree.Tree(self.dataObj, self.categories, depth)
			#calculate weight
			
			tree.build(train_data, self.categories, weights, depth, z)
			tree.prune()
			self.trees.append(tree)
		
	def classify(self, dataMatrix):
		cats = 
		for tree in self.trees:
			
