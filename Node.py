import numpy as np

def Node:
	#feature is an int
	def __init__(self, depth, z):
		self.z = z
		self.right = None
		self.left = None
		self.feature = None
		self.threshold = None
		self.entropy = None
		self.error = None
		self.classCounts = []
		self.depth = depth
	
	#set the data and categories of the node and calculate entropy and error
	def build(self, dataMatrix, categories, weights):
		self.data = dataMatrix
		self.categories = categories
		self.weights = weights
		#need to do something with z what is it
		col = dataMatrix[:,self.feature]
		unique, mapping = np.unique(np.array(categories), return_inverse = True)
		for i in range(len(unique)):
			self.classCounts.append(0)
		for i in range(len(mapping)):
			self.classCounts[mapping[i]]+=weights[i]
		#calculate entropy
		p = []
		for i in range(categories.shape[0]):
			p[i] = self.classCounts[i]/dataMatrix.shape[0]
		self.entropy = np.sum(np.multiply(-p, np.log2(p)), axis=1)
		#calculate error
		f = (np.sum(self.classCounts) - np.max(self.classCounts))/np.sum(self.classCounts)
		N = dataMatrix.shape[0]
		self.error = f + self.z*self.z/(2*N)
		self.error = self.error + self.z*np.sqrt(f/N - f*f/N + self.z*self.z/4/N/N)
		self.error = self.error/(1 + self.z*self.z/N)
	
	#calculate the optimal feature and threshold and split the data to two children nodes
	def split(self):
		if self.depth<0:
			return
		#have it create them one at a time instead of all of them and only storing the "best one"
		best = Node(depth-1, self.z)
		bestFeature = 0
		bestThreshold = self.data[bestFeature,0]
		newDat = self.data[self.data[0]>bestThreshold])
		newCat = self.categories[self.data[0]>bestThreshold])
		newWeight = self.weights[self.data[0]>bestThreshold])
		best.build(newDat, newCat, newWeight)
		for i in range(self.data.shape[0]):
			for j in range(self.data.shape[1]):
				new = Node(depth-1, self.z)
				newFeature = j
				newThreshold self.data[i,j]
				newDat = self.data[self.data[j]>newThreshold])
				newCat = self.categories[self.data[j]>newThreshold])
				newWeight = self.weights[self.data[j]>newThreshold])
				new.build(newDat, newCat, newWeight)
				if new.entropy<best.entropy:
					best = new
					bestThreshold = newThreshold
					bestFeature = newFeature
		self.threshold = bestThreshold
		self.feature = newFeature
		self.right = best
		self.right.split()
		self.left = Node(depth-1,self.z)
		leftDat = self.data[self.data[self.feature]<self.threshold])
		leftCat = self.categories[self.data[self.feature]<self.threshold])
		leftWeight = self.weights[self.data[self.feature]<self.threshold])
		self.left.build(leftDat, leftCat, leftWeight)
		self.left.split()
	
	#go through all the nodes bottom-up and relace redundant subtrees with leaf nodes
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
	
	#take in a data point and return its class
	def classify(self, point):
		if self.right is None:
			index = 0
			for i in range(self.classCounts):
				if self.classCounts[i]>self.classCounts[index]:
					index = i
			unique,mapping = np.unique(np.array(self.categories), return_inverse = True)
			return unique[index]
		if point[self.feature]>self.threshold:
			return self.right.classify(point)
		else:
			return self.left.classify(point)
