import numpy as np

class Node:
	#feature is an int
	index = 0
	def __init__(self, depth, z):
		self.z = z
		self.kids = [None, None]
		self.feature = None
		self.threshold = None
		self.entropy = None
		self.error = None
		self.classCounts = []
		self.depth = depth
		self.unique = None
	
	#set the data and categories of the node and calculate entropy and error
	def build(self, dataMatrix, categories, weights, unique):
		self.unique = unique
		self.data = dataMatrix
		self.categories = categories
		self.weights = weights
		col = dataMatrix[:,self.feature]
		self.classCounts = np.zeros(shape = len(unique))
		for i in range(len(categories)):
			self.classCounts[categories[i] == unique]+=weights[i]
		p = []
		for i in range(len(unique)):
			p.append(self.classCounts[i]/dataMatrix.shape[0])
		self.entropy = 0
		for prob in p:
			if prob != 0:
				self.entropy += -prob*np.log2(prob)
		f = (np.sum(self.classCounts) - np.max(self.classCounts))/np.sum(self.classCounts)
		N = dataMatrix.shape[0]
		self.error = f + self.z*self.z/(2*N)
		self.error = self.error + self.z*np.sqrt(f/N - f*f/N + self.z*self.z/4/N/N)
		self.error = self.error/(1 + self.z*self.z/N)
	
	#calculate the optimal feature and threshold and split the data to two children nodes
	def split(self):
		if self.depth<=0:
			return
		if self.data.shape[0] <=2:
			return
		#have it create them one at a time instead of all of them and only storing the "best one"
		best = [None, None]
		startFeature = 0
		startRow = 0
		bestThreshold = self.data[startRow, startFeature]
		rightDat = self.data[self.data[:,startFeature]>bestThreshold]
		rightCat = self.categories[self.data[:,startFeature]>bestThreshold]
		rightWeight = self.weights[self.data[:,startFeature]>bestThreshold]
		
		leftDat = self.data[self.data[:,startFeature]<bestThreshold]
		leftCat = self.categories[self.data[:,startFeature]<bestThreshold]
		leftWeight = self.weights[self.data[:,startFeature]<bestThreshold]
		
		
		minimum = np.min(self.data, axis =0)
		maximum = np.max(self.data, axis = 0)
		while bestThreshold == maximum[startFeature] or bestThreshold == minimum[startFeature]:
			startFeature = np.random.randint(self.data.shape[1])
			startRow = np.random.randint(self.data.shape[0])
			bestThreshold = self.data[startRow, startFeature]
			rightDat = self.data[self.data[:,startFeature]>bestThreshold]
			rightCat = self.categories[self.data[:,startFeature]>bestThreshold]
			rightWeight = self.weights[self.data[:,startFeature]>bestThreshold]
					
			leftDat = self.data[self.data[:,startFeature]<bestThreshold]
			leftCat = self.categories[self.data[:,startFeature]<bestThreshold]
			leftWeight = self.weights[self.data[:,startFeature]<bestThreshold]
		
		bestFeature = startFeature
		best[0] = Node(self.depth-1, self.z)
		best[1] = Node(self.depth-1, self.z)
		
		best[0].build(rightDat, rightCat, rightWeight, self.unique)
		best[1].build(leftDat, leftCat, leftCat, self.unique)
		
		for i in range(self.data.shape[0]):
			for j in range(self.data.shape[1]):
				right = Node(self.depth-1, self.z)
				left = Node(self.depth-1, self.z)
				newFeature = j
				newThreshold=self.data[i,j]
				if not(newThreshold == maximum[newFeature] or newThreshold == minimum[newFeature]):
					
					rightDat = self.data[self.data[:,j]>newThreshold]
					rightCat = self.categories[self.data[:,j]>newThreshold]
					rightWeight = self.weights[self.data[:,j]>newThreshold]
					
					leftDat = self.data[self.data[:,j]<newThreshold]
					leftCat = self.categories[self.data[:,j]<newThreshold]
					leftWeight = self.weights[self.data[:,j]<newThreshold]
					
					right.build(rightDat, rightCat, rightWeight, self.unique)
					left.build(leftDat, leftCat, leftWeight, self.unique)
					if right.entropy*right.data.shape[0]+left.entropy*left.data.shape[0]<best[0].entropy * best[0].data.shape[0] + best[1].entropy * best[1].data.shape[0]:
						best = [right,left]
						bestThreshold = newThreshold
						bestFeature = newFeature
						
		self.threshold = bestThreshold
		self.feature = newFeature
		self.kids = best
		self.kids[0].split()
		self.kids[1].split()
		
	
	#go through all the nodes bottom-up and relace redundant subtrees with leaf nodes
	def prune(self):
		if self.kids[0] is None:
			return
		self.kids[0].prune()
		self.kids[1].prune()
		childerror = 0
		for cat in self.kids[0].categories:
			childerror+= self.kids[0].error * cat
		for cat in self.kids[1].categories:
			childerror+= self.kids[1].error * cat
		parerror = 0
		for cat in self.categories:
			parerror += self.error*cat
		if childerror>parerror:
			self.kids[0] = None
			self.kids[1] = None
	
	#take in a data point and return its class
	def classify(self, point):
		if self.kids[0] is None:
			index = 0
			for i in range(len(self.classCounts)):
				if self.classCounts[i]>self.classCounts[index]:
					index = i
			return self.unique[index]
		if point[self.feature]>self.threshold:
			return self.kids[0].classify(point)
		else:
			return self.kids[1].classify(point)
	
	def __str__(self):
		#print self.classCounts
		#if self.kids[1] is not None:
		#	print self.kids[1]
		#if self.kids[0] is not None:
			#print self.kids[0]
		print 'work in progress'
