import numpy as np
import Node
import data

class Tree:

	def __init__(self, dataMatrix=None, categories=None, depth=20, z=1.96, weights = None):
		
		self.dataMatrix = dataMatrix
		self.depth = depth
		
		self.root = None
		if self.dataMatrix != None:
			self.build(dataMatrix, categories, weights, depth, z)
			
			
	# build a decision tree using the training data set
	def build(self, train_data, categories, weights, depth, z):
		self.root = Node.Node(depth, z)
		if weights is None:
			weights = np.ones_like(categories)
		unique, mapping = np.unique( np.array(categories.T), return_inverse = True)
		self.root.build(train_data, mapping, weights, unique)
		self.root.split()
	
	# prune the tree so that the information gain of children nodes are larger than the parent one
	def prune(self):
		self.root.prune()
	
	# take in a data set and return a list of classes
	def classify(self, data):
		cats = np.matrix(np.zeros(shape = (data.shape[0], 1)))
		for i in range(data.shape[0]):
			cat = self.root.classify(data[i, :])
			cats[i, 0] = cat
		return cats
	
	# test the tree 
	def test(self):
		pass

class randomTree(tree):
	def __init__(self, dataMatrix=None, categories=None, numFeatures=5, depth=20, z=1.96, weights = None):
		self.dataMatrix = dataMatrix
		self.depth = depth
		
		self.root = None
		if self.dataMatrix != None:
			self.build(dataMatrix, categories, numFeatures, weights, depth, z)
			
			
	# build a decision tree using the training data set
	def build(self, train_data, categories, numFeatures, weights, depth, z):
		self.root = Node.Node(depth, z)
		if weights is None:
			weights = np.ones_like(categories)
		unique, mapping = np.unique( np.array(categories.T), return_inverse = True)
		self.root.build(train_data, mapping, weights, unique)
		self.root.split(numFeatures)
