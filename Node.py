import numpy as np

def Node:
	#feature is an int
	def __init__(self, depth):
		self.right = None
		self.left = None
		self.feature = None
		self.threshold = None
		self.entropy = None
		self.error = None
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
		#calculate error
		
	def split(self):
		if self.depth<0:
			return
		#have it create them one at a time instead of all of them and only storing the "best one"
		best = Node(depth-1)
		bestFeature = 0
		bestThreshold = self.data[bestFeature,0]
		newDat = self.data[self.data[0]>bestThreshold)
		newCat = self.categories[self.data[0]>bestThreshold)
		best.build(newDat, newCat)
		for i in range(self.data.shape[0]):
			for j in range(self.data.shape[1]):
				new = Node(depth-1)
				newFeature = j
				newThreshold self.data[i,j]
				newDat = self.data[self.data[j]>newThreshold])
				newCat = self.categories[self.data[j]>newThreshold])
				new.build(newDat, newCat)
				if new.entropy<best.entropy:
					best = new
					bestThreshold = newThreshold
					bestFeature = newFeature
		self.threshold = bestThreshold
		self.feature = newFeature
		self.right = best
		self.right.split()
		self.left = Node(depth-1)
		leftDat = self.data[self.data[self.feature]<self.threshold])
		leftCat = self.categories[self.data[self.feature]<self.threshold])
		self.left.build(leftDat, leftCat)
		self.left.split()
		
	def prune(self):
		if self.depth == 0:
			return
		self.right.prune()
		self.left.prune()
		childerror = 0
		for cat in self.right.categories:
			childerror+= self.right.error * cat
		for cat in self.left.categories:
			childerror+= self.left.error * cat
		parerror = 0
		for cat in self.categories:
			parerror += self.error*cat
		if childerror>parerror:
			self.right, self.left = None
			
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
