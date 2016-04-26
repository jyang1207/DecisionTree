class Tree:

	def __init__(self, dataObj=None, depth=20):
		
		self.dataObj = dataObj
		self.depth = depth
		
		if self.dataObj != None:
			self.build(dataObj, depth)
			
	# build a decision tree using the training data set
	def build(train_data, depth):
		
	
	# prune the tree so that the information gain of children nodes are larger than the parent one
	def prune():
	
	
	# take in a data set and return a list of classes
	def classify(data):
	
	
	# test the tree 
	def test():