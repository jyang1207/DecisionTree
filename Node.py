#file Node.py
#Authors: Jason Gurevitch and Jianing Yang
#date: 5/8/2016

import numpy as np
import random

class Node:
	
	#initializes a decision tree node with a given depth and a given z value
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
	#unique is passed to every node in a tree so that class counts always
	#has the same shape, and that each class count is in the same place relative to the other nodes
	def build(self, dataMatrix, categories, weights, unique):
		self.unique = unique
		self.data = dataMatrix
		self.categories = categories
		self.weights = weights
		self.classCounts = np.zeros(shape = len(unique))
		for i in range(len(categories)):
			self.classCounts[categories[i] == unique]+=weights[i,0]
		#calcualte entropy
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
		
	#calculate the optimal feature and threshold and split the data to two children nodes by checking every option
	#if given a number of features it will check that number of random features, instead of all of them
	def split(self, num_features = None):
		#base case
		if self.depth<=0:
			return
		#base case as to not create an infinate loop since if the data point chosen is the minimum there would be a node with no information, which crashes the program
		#we chose 3 as an arbitrary number, but since there could be repeats we thought it was a good idea to not just say a smaller one, and we'd often get loops with lower ones
		if self.data.shape[0] <=3:
			return
		#if the node is for a random tree or not
		if num_features is None or num_features >= self.data.shape[1]:
			features =range(self.data.shape[1])
		else:
			basefeatures = range(self.data.shape[1])
			features = []
			for i in range(num_features-1):
				features.append(basefeatures.pop(np.random.randint(self.data.shape[1]- i)))
		best = [None, None]
		startFeature = features[0]
		startRow = 0
		
		#pick starting nodes that aren't the minimum or maximum
		minimum = np.min(self.data, axis =0)
		maximum = np.max(self.data, axis = 0)
		for i in range(1000):
			startFeature = random.choice(features)
			startRow = np.random.randint(self.data.shape[0])
			#print startRow, startFeature
			bestThreshold = self.data[startRow, startFeature]
			rightDat = self.data[self.data[:,startFeature]>bestThreshold]
			rightCat = self.categories[self.data[:,startFeature]>bestThreshold]
			rightWeight = self.weights[self.data[:,startFeature]>bestThreshold]
					
			leftDat = self.data[self.data[:,startFeature]<=bestThreshold]
			leftCat = self.categories[self.data[:,startFeature]<=bestThreshold]
			leftWeight = self.weights[self.data[:,startFeature]<=bestThreshold]
			
			if  not(bestThreshold == maximum[startFeature]):
				break
		if i == 999:
			print 'you were probably in an infinite loop due to the last few datapoints being the same so we saved you that trouble'
			raise
		
		#build the starting nodes
		bestFeature = startFeature
		best[0] = Node(self.depth-1, self.z)
		best[1] = Node(self.depth-1, self.z)
		
		best[0].build(rightDat, rightCat, rightWeight, self.unique)
		best[1].build(leftDat, leftCat, leftWeight, self.unique)
		
		#find the best threshold to split by
		for i in range(self.data.shape[0]):
			for j in features:
				right = Node(self.depth-1, self.z)
				left = Node(self.depth-1, self.z)
				newFeature = j
				newThreshold=self.data[i,j]
				if not(newThreshold == maximum[newFeature]):
					
					rightDat = self.data[self.data[:,j]>newThreshold]
					rightCat = self.categories[self.data[:,j]>newThreshold]
					rightWeight = self.weights[self.data[:,j]>newThreshold]
					
					leftDat = self.data[self.data[:,j]<=newThreshold]
					leftCat = self.categories[self.data[:,j]<=newThreshold]
					leftWeight = self.weights[self.data[:,j]<=newThreshold]
					
					right.build(rightDat, rightCat, rightWeight, self.unique)
					left.build(leftDat, leftCat, leftWeight, self.unique)
					if right.entropy*right.data.shape[0]+left.entropy*left.data.shape[0]<best[0].entropy * best[0].data.shape[0] + best[1].entropy * best[1].data.shape[0]:
						best = [right,left]
						bestThreshold = newThreshold
						bestFeature = newFeature
						
		self.threshold = bestThreshold
		self.feature = bestFeature
		self.kids = best
		self.kids[0].split(num_features = num_features)
		self.kids[1].split(num_features = num_features)
				
	
	#go through all the nodes bottom-up and relace redundant subtrees with leaf nodes
	def prune(self):
		if self.kids[0] is None:
			return
		self.kids[0].prune()
		self.kids[1].prune()
		childerror = 0
		for kid in self.kids:
			for cat in kid.classCounts:
				childerror+=kid.error*cat
		parerror = 0
		for cat in self.classCounts:
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

	
