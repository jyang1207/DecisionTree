class Tree:

	def __init__(self, dataObj=None, depth=20):
		
		self.dataObj = dataObj
		self.depth = depth
		
		if self.dataObj != None:
			self.build(dataObj, depth)
			
		self.root = node.Node(depth)
			
	# build a decision tree using the training data set
	def build(self, train_data, depth):
		
	
	# prune the tree so that the information gain of children nodes are larger than the parent one
	def prune(self):
	
	
	# take in a data set and return a list of classes
	def classify(self, data):
		cats = np.matrix(np.zeros(shape = (data.shape[0], 1)))
		for i in range(data.shape[0]):
			class = self.root.classify(data[i, :])
			cats[i, :] = class
		return cats
	
	# test the tree 
	def test(self):
