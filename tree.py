import numpy as np
import Node
import data

class Tree:

	def __init__(self, dataObj=None, categories=None depth=20):
		
		self.dataObj = dataObj
		self.depth = depth
		
		self.root = None
		if self.dataObj != None:
			self.build(dataObj, categories depth)
			
			
	# build a decision tree using the training data set
	def build(self, train_data, categories, depth):
		self.root = Node.Node(depth)
		self.root.build(train_data.get_data(train_data.get_headers()), categories)
		self.root.split()
	
	# prune the tree so that the information gain of children nodes are larger than the parent one
	def prune(self):
		self.root.prune()
	
	# take in a data set and return a list of classes
	def classify(self, data):
		cats = np.matrix(np.zeros(shape = (data.shape[0], 1)))
		for i in range(data.shape[0]):
			class = self.root.classify(data[i, :])
			cats[i, 0] = class
		return cats
	
	# test the tree 
	def test(self):