import numpy as np

def Node:
	#feature is an int
	def __init__(self, depth, feature, threshold):
		self.right = None
		self.left = None
		self.feature = feature
		self.threshold = threshold
		self.entropy = None
		self.classCounts = []
		self.depth = depth
	
	def build(self, dataMatrix, categories):
		self.data = dataMatrix
		self.categories = categories
		col = dataMatrix[:,self.feature]
		unique, mapping = np.unique(categories, return_inverse = True)
		for i in range(len(unique)):
			self.classCounts.append(0)
		for thing in mapping:
			self.classCounts[thing]+=1
		#calculate entropy
		
	def split(self):
		if self.depth<0:
			return
		#have it create them one at a time instead of all of them and only storing the "best one"
		best = Node(depth-1, 0, self.data[0,0])
		newDat = self.data[self.data[0]>best.threshold])
		newCat = self.categories[self.data[0]>best.threshold])
		best.build(newDat, newCat)
		for i in range(self.data.shape[0]):
			for j in range(self.data.shape[1]):
				new = Node(depth-1, j, self.data[i,j])
				newDat = self.data[self.data[j]>new.threshold])
				newCat = self.categories[self.data[j]>new.threshold])
				new.build(newDat, newCat)
				if new.entropy<best.entropy:
					best = new
		self.right = best
		self.right.split()
		self.left = Node(depth-1, self.right.feature, self.right.threshold)
		leftDat = self.data[self.data[self.right.feature]<self.right.threshold])
		leftCat = self.categories[self.data[self.right.feature]<self.right.threshold])
		self.left.build(leftDat, leftCat)
		self.left.split()
		
	def prune(self):
		if self.depth == 0:
			return
		self.right.prune()
		self.left.prune()
		
	def classify(self, point):
		if self.right is None:
			index = 0
			for i in range(self.classCounts):
				if self.classCounts[i]>self.classCounts[index]:
					index = i
			unique,mapping = np.unique(self.categories, return_inverse = True)
			return unique[index]
		if point[self.feature]>self.threshold:
			return self.right.classify(point)
		else:
			return self.left.classify(point)
