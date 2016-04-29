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
		self.categories
		col = dataMatrix[:,self.feature]
		unique, mapping = np.unique(categories, return inverse = True)
		for i in range(len(unique)):
			classCounts.append(0)
		for thing in mapping:
			classCounts[thing]+=1
		#calculate entropy
		
	def split(self):
		if depth<0:
			return
		#have it create them one at a time instead of all of them and only storing the "best one"
		best = Node(depth-1, 0, self.data[0,0])
		newDat = self.data[self.data[0]>best.threshold])
		newCat = self.categories[self.data[0]>best.threshold])
		best.build(newDat, newCat)
		for j in range(self.data.shape[1]):
			for i in range(self.data.shape[0]):
				new = Node(depth-1, j, self.data[i,j])
				newDat = self.data[self.data[j]>new.threshold])
				newCat = self.categories[self.data[0]>new.threshold])
				new.build(newDat, newCat)
				if new.entropy<best.entropy:
					best = new
		self.right = best
		self.right.split()
		self.left = Node(depth-1, self.right.feature, self.right.threshold)
		leftDat = self.data[self.data[0]<self.right.threshold])
		leftCat = self.categories[self.data[0]<self.right.threshold])
		self.left.build(leftDat, leftCat)
		self.left.split()
		
	def prune(self):
		
	def classify(self, point):
		if point[self.feature]>self.threshold:
			if self.right is not None:
				self.right.classify(point)
			else
				pass
				#return greater of class counts
		else:
			if self.left is not None:
				self.left.classify(point)
		else:
			pass
			#return greater of class counts
