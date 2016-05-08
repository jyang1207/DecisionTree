# Template by Bruce Maxwell
# Spring 2015
# CS 251 Project 8
#
# Classifier class and child definitions

import sys
import data
import analysis as an
import numpy as np
import scipy.cluster.vq as vq
import time
import tree
import forest

class Classifier:

	def __init__(self, type):
		'''The parent Classifier class stores only a single field: the type of
		the classifier.	 A string makes the most sense.

		'''
		self._type = type

	def type(self, newtype = None):
		'''Set or get the type with this function'''
		if newtype != None:
			self._type = newtype
		return self._type

	def confusion_matrix( self, truecats, classcats ):
		'''Takes in two Nx1 matrices of zero-index numeric categories and
		computes the confusion matrix. The rows represent true
		categories, and the columns represent the classifier output.

		'''
		
		unique, mapping = np.unique(np.array(truecats), return_inverse = True)
		#print unique
		confMatrix = np.matrix(np.zeros(shape = (self.num_classes, self.num_classes)))
		
		for i in range(truecats.shape[0]):
			#print int(classcats[i,0]),np.argwhere(unique == (truecats[i,0]))
			confMatrix[int(classcats[i,0]),np.argwhere(unique == (truecats[i,0]))]+=1
			#print 'time for %d pass'%(i), time.time()-begin
		return confMatrix

	def confusion_matrix_str( self, cmtx ):
		'''Takes in a confusion matrix and returns a string suitable for printing.'''
		s = 'actual ->'
		unique, mapping = np.unique(np.array(self.categories.T), return_inverse = True)
		for i in range(len(unique)):
			#print thing
			s+= '%d'%(unique[i])
			for c in range(len(str(np.amax(cmtx[:,i])))-2):
				s+= ' '
		s+= '\n'
		for char in s:
			s+='-'
		s+= '\n'
		for i in  range(cmtx.shape[0]):
			s+= 'pred %d |'%(unique[i])
			for j in range(cmtx.shape[1]):
				s+= '%d'%(cmtx[i,j])
				extra_chars = len(str(np.amax(cmtx[:,j])))-len(str(cmtx[i,j]))
				for c in range(extra_chars+1):
					s+=' '
			s+= '\n'
		
		return s

	def __str__(self):
		'''Converts a classifier object to a string.  Prints out the type.'''
		return str(self._type)

class DecisionTreeClassifier(Classifier):
	
	def __init__(self, dataMatrix = None, categories = None):
		Classifier.__init__(self, 'Decision Tree Classifier')
		self.categories = categories
		self.num_classes = 0
		self.class_labels = None
		if dataMatrix is not None:
			self.build(dataMatrix, categories)
	
	def build(self, A, categories):
		self.class_labels, mapping = np.unique( np.array(categories.T), return_inverse = True)
		self.num_classes = self.class_labels.shape[0]
		self.tree = tree.Tree(A, categories)
		self.tree.prune()
	
	def classify(self, A):
		return self.tree.classify(A)
		


class RandomTreeClassifier(Classifier):
	
	def __init__(self, dataMatrix = None, categories = None, numfeatures = 5):
		Classifier.__init__(self, 'Decision Tree Classifier')
		self.categories = categories
		self.num_classes = 0
		self.class_labels = None
		if dataMatrix is not None:
			self.build(dataMatrix, categories, numfeatures)
	
	def build(self, A, categories, numfeatures= 5):
		self.class_labels, mapping = np.unique( np.array(categories.T), return_inverse = True)
		self.num_classes = self.class_labels.shape[0]
		self.tree = tree.randomTree(A, categories, numfeatures)
		self.tree.prune()
	
	def classify(self, A):
		return self.tree.classify(A)


class ForestClassifier(Classifier):
	
	def __init__(self, dataMatrix = None, categories = None):
		Classifier.__init__(self, 'ForrestClassifier')
		self.categories = categories
		self.num_classes = 0
		self.class_labels = None
		if dataMatrix is not None:
			self.build(dataMatrix, categories)
			
	def build(self, A, categories):
		self.class_labels, mapping = np.unique( np.array(categories.T), return_inverse = True)
		self.num_classes = self.class_labels.shape[0]
		self.forest = forest.Forest(A, categories)
	
	def classify(self, A):
		return self.forest.classify(A)
		
	#use k-fold cross validation to test the forest
	def cross_validation(self, dataMatrix, categories, headers, k):
		n = int(dataMatrix.shape[0]/k)
		temp_matrix = dataMatrix[:, headers]
		for i in range(k):
			print i, 'fold'
			if i == 0:
				train = temp_matrix[range(i*n + n, min(k*n, categories.shape[0]-1)), :]
				train_cats = categories[range(i*n + n, min(k*n, categories.shape[0]-1)), :]
			else:
				train = temp_matrix[range(i*n),:]
				train2 = temp_matrix[range(i*n + n, min(k*n, categories.shape[0]-1)), :]
				train = np.vstack((train, train2))
				train_cats = categories[range(i*n), :]
				train_cats = np.vstack((train_cats, categories[range(i*n + n, min(k*n, categories.shape[0]-1)), :]))
			test = temp_matrix[range(i*n, i*n + n),:]
			f = forest.Forest(train, train_cats)
			if i == 0:
				results = f.classify(test)
			else:
				results = np.vstack((results, f.classify(test)))
		cm = self.confusion_matrix(categories[:results.shape[0], :], results)
		print self.confusion_matrix_str(cm)
		
		return cm
		
	#stratified k-fold cross validation where each class is distributed evenly to each fold
	def stratified_cv(self, dataMatrix, categories, headers, k):
		unique, mapping = np.unique(np.array(categories), return_inverse = True)
		classIndices = []
		for c in range(len(unique)):
			classIndices.append([])
		for m in range(categories.shape[0]):
			classIndices[mapping[m]].append(m)
		cmatrices = []
		temp_matrix = dataMatrix[:, headers]
		for i in range(k):
			trainlist = []
			testlist = []
			for j in range(len(classIndices)):
				n = int(len(classIndices[j])/k)
				trainlist += classIndices[j][:i*n]
				trainlist += classIndices[j][(i*n+n):]
				testlist += classIndices[j][i*n : i*n+n]
			train = temp_matrix[trainlist, :]
			train_cats = categories[trainlist, :]
			f = forest.Forest(train, train_cats)
			test = temp_matrix[testlist, :]
			if i == 0:
				test_cats = categories[testlist, :]
				results = f.classify(test)
			else:
				test_cats = np.vstack((test_cats, categories[testlist, :]))
				results = np.vstack((results, f.classify(test)))
		cm = self.confusion_matrix(test_cats, results)
		print self.confusion_matrix_str(cm)
		
		return cm

		
class NaiveBayes(Classifier):
	'''NaiveBayes implements a simple NaiveBayes classifier using a
	Gaussian distribution as the pdf.

	'''

	def __init__(self, dataObj=None, headers=[], categories=None):
		'''Takes in a Data object with N points, a set of F headers, and a
		matrix of categories, one category label for each data point.'''

		# call the parent init with the type
		Classifier.__init__(self, 'Naive Bayes Classifier')
		
		
		# store the headers used for classification
		self.headers = headers
		self.categories = categories
		# number of classes and number of features
		self.F = 0
		self.num_classes = 0
		# original class labels
		self.class_labels = None
		# unique data for the Naive Bayes: means, variances, scales
		self.class_means = None
		self.class_vars = None
		self.class_scales = None
		
		if dataObj is not None:
			if categories is not None:
				A = dataObj.get_data(headers)
				self.build(A, categories)

	def build( self, A, categories ):
		'''Builds the classifier give the data points in A and the categories'''
		self.class_labels, mapping = np.unique( np.array(categories.T), return_inverse = True)
		
		self.F = A.shape[1]
		
		self.num_classes = self.class_labels.shape[0]
		
		self.class_means = np.asmatrix(np.zeros(shape= (self.num_classes,self.F)))
		self.class_vars = np.asmatrix(np.zeros(shape= (self.num_classes,self.F)))
		self.class_scales = np.asmatrix(np.zeros(shape= (self.num_classes,self.F)))
		
		for i in range(self.num_classes):
			row = A[(mapping ==i),:]
			self.class_means[i] = np.mean(row,axis = 0)
			self.class_vars[i] = np.var(row, axis = 0, ddof =1)
			self.class_scales[i] = 1/np.sqrt(2*np.pi*self.class_vars[i])
			
		
		
		# figure out how many categories there are and get the mapping (np.unique)
		# create the matrices for the means, vars, and scales
		# the output matrices will be categories (C) x features (F)
		# compute the means/vars/scales for each class
		# store any other necessary information: # of classes, # of features, original labels

		return

	def classify( self, A, return_likelihoods=False ):
		'''Classify each row of A into one category. Return a matrix of
		category IDs in the range [0..C-1], and an array of class
		labels using the original label values. If return_likelihoods
		is True, it also returns the NxC likelihood matrix.

		'''

		# error check to see if A has the same number of columns as
		# the class means
		
		if A.shape[1] != self.class_means.shape[1]:
			print 'means is not the same shape as you\'r data'
			return
		
		# make a matrix that is N x C to store the probability of each
		# class for each data point
		P = np.asmatrix(np.zeros(shape=(A.shape[0],self.num_classes)))
		
		# calculate the probabilities by looping over the classes
		#  with numpy-fu you can do this in one line inside a for loop
		for i in range(self.num_classes):
			P[:,i] = np.prod(np.multiply(self.class_scales[i],np.exp(-np.square(A-self.class_means[i])/self.class_vars[i])), axis = 1)
			
		# calculate the most likely class for each data point
		cats = np.argmax(P, axis = 1)

		# use the class ID as a lookup to generate the original labels
		labels = self.class_labels[cats]

		if return_likelihoods:
			return cats, labels, P

		return cats, labels

	def __str__(self):
		'''Make a pretty string that prints out the classifier information.'''
		s = "\nNaive Bayes Classifier\n"
		for i in range(self.num_classes):
			s += 'Class %d --------------------\n' % (i)
			s += 'Mean	: ' + str(self.class_means[i,:]) + "\n"
			s += 'Var	: ' + str(self.class_vars[i,:]) + "\n"
			s += 'Scales: ' + str(self.class_scales[i,:]) + "\n"

		s += "\n"
		return s
		
	def write(self, filename):
		'''Writes the Bayes classifier to a file.'''
		# extension
		return

	def read(self, filename):
		'''Reads in the Bayes classifier from the file'''
		# extension
		return

	
class KNN(Classifier):

	def __init__(self, dataObj=None, headers=[], categories=None, K=None):
		'''Take in a Data object with N points, a set of F headers, and a
		matrix of categories, with one category label for each data point.'''

		# call the parent init with the type
		Classifier.__init__(self, 'KNN Classifier')
		
		# store the headers used for classification		
		self.headers = headers
		self.categories = categories
		# number of classes and number of features
		self.F = 0
		# original class labels
		self.class_labels = None
		# unique data for the KNN classifier: list of exemplars (matrices)
		self.exemplars =[]
		if dataObj is not None:
			A = dataObj.get_data(headers)
			self.build(A,categories,K)
		
	def build( self, A, categories, K = None ):
		'''Builds the classifier give the data points in A and the categories'''
		
		# figure out how many categories there are and get the mapping (np.unique)
		self.class_labels, mapping = np.unique( np.array(categories.T), return_inverse = True)
		
		self.F = A.shape[1]
		
		self.num_classes = self.class_labels.shape[0]
		
		for i in range(self.num_classes):
			if K is None:
				self.exemplars.append(A[mapping == i,:])
				#append to exemplars a matrix with all of the rows of A where the category/mapping is i
			else:
				W = A[mapping == i,:]
				codebook = an.kmeans_init(W, K)
				codebook,codes,errors = an.kmeans_algorithm(W, codebook)
				#print codebook
				self.exemplars.append(codebook)
				# run K-means on the rows of A where the category/mapping is i
				# append the codebook to the exemplars

		# store any other necessary information: # of classes, # of features, original labels

		return

	def classify(self, A, K=3, return_distances=False):
		'''Classify each row of A into one category. Return a matrix of
		category IDs in the range [0..C-1], and an array of class
		labels using the original label values. If return_distances is
		True, it also returns the NxC distance matrix.

		The parameter K specifies how many neighbors to use in the
		distance computation. The default is three.'''

		# error check to see if A has the same number of columns as the class means
		if A.shape[1] != self.exemplars[0].shape[1]:
			print 'something\'s wrong'
			return

		# make a matrix that is N x C to store the distance to each class for each data point
		D = np.matrix(np.zeros(shape = (A.shape[0],self.num_classes))) 
		
		for i in range(self.num_classes):
			# make a temporary matrix that is N x M where M is the number of examplars (rows in exemplars[i])
			temp_matrix = np.matrix(np.zeros(shape = (A.shape[0],self.exemplars[i].shape[0])))
			# calculate the distance from each point in A to each point in exemplar matrix i (for loop)
			for j in range(self.exemplars[i].shape[0]):
				try:
					temp_matrix[:,j] = np.asmatrix(np.sum(np.square(A-self.exemplars[i][j,:]),axis = 1)).T
				except:
					temp_matrix[:,j] = np.asmatrix(np.sum(np.square(A-self.exemplars[i][j,:]),axis = 1))
			# sort the distances by row
			temp_matrix.sort(axis = 1)
			# sum the first K columns
			first_K = temp_matrix[:,:K]
			sum = np.sum(first_K, axis = 1)
			# this is the distance to the first class
			#print temp_matrix
			D[:,i] = sum
			

		# calculate the most likely class for each data point
		cats = np.argmin(D, axis = 1) # take the argmin of D along axis 1

		# use the class ID as a lookup to generate the original labels
		labels = self.class_labels[cats]

		if return_distances:
			return cats, labels, D

		return cats, labels

	def __str__(self):
		'''Make a pretty string that prints out the classifier information.'''
		s = "\nKNN Classifier\n"
		for i in range(self.num_classes):
			s += 'Class %d --------------------\n' % (i)
			s += 'Number of Exemplars: %d\n' % (self.exemplars[i].shape[0])
			s += 'Mean of Exemplars	 :' + str(np.mean(self.exemplars[i], axis=0)) + "\n"

		s += "\n"
		return s


	def write(self, filename):
		'''Writes the KNN classifier to a file.'''
		# extension
		return

	def read(self, filename):
		'''Reads in the KNN classifier from the file'''
		# extension
		return
		
class KNN_FuzzyC(Classifier):

	def __init__(self, dataObj=None, headers=[], categories=None, K=None):
		'''Take in a Data object with N points, a set of F headers, and a
		matrix of categories, with one category label for each data point.'''

		# call the parent init with the type
		Classifier.__init__(self, 'KNN Classifier')
		
		# store the headers used for classification		
		self.headers = headers
		self.categories = categories
		# number of classes and number of features
		self.F = 0
		# original class labels
		self.class_labels = None
		# unique data for the KNN classifier: list of exemplars (matrices)
		self.exemplars =[]
		if dataObj is not None:
			A = dataObj.get_data(headers)
			self.dataObj = dataObj
			self.build(A,categories,K,headers)
		
	def build( self, A, categories, headers, K = None ):
		'''Builds the classifier give the data points in A and the categories'''
		
		# figure out how many categories there are and get the mapping (np.unique)
		self.class_labels, mapping = np.unique( np.array(categories.T), return_inverse = True)
		
		self.F = A.shape[1]
		
		self.num_classes = self.class_labels.shape[0]
		
		for i in range(self.num_classes):
			if K is None:
				self.exemplars.append(A[mapping == i,:])
				#append to exemplars a matrix with all of the rows of A where the category/mapping is i
			else:
				W = A[mapping == i,:]
				#centroids, partitionMatrix = an.fuzzyCmeans(self.dataObj, self.dataObj.get_headers(), K)
				#print len(headers)
				#print W.shape[0]
				centroids, partitionMatrix = an.fuzzyCinit(W, self.num_classes, headers)
				partitionMatrix, centroids = an.fuzzyC_algorithm(W, centroids, partitionMatrix)
				#codebook = np.asmatrix(np.zeros(shape = partitionMatrix.shape[0])).T
				#print partitionMatrix
				#for i in range(partitionMatrix.shape[0]):
					#codebook[i] = np.argmax(partitionMatrix[i,:])
				#print centroids
				self.exemplars.append(centroids)
				# run K-means on the rows of A where the category/mapping is i
				# append the codebook to the exemplars

		# store any other necessary information: # of classes, # of features, original labels

		return

	def classify(self, A, K=3, return_distances=False):
		'''Classify each row of A into one category. Return a matrix of
		category IDs in the range [0..C-1], and an array of class
		labels using the original label values. If return_distances is
		True, it also returns the NxC distance matrix.

		The parameter K specifies how many neighbors to use in the
		distance computation. The default is three.'''

		# error check to see if A has the same number of columns as the class means
		"""
		if A.shape[1] != self.exemplars[0].shape[1]:
			print A
			print self.exemplars
			print 'something\'s wrong'
			return
		"""
		# make a matrix that is N x C to store the distance to each class for each data point
		D = np.matrix(np.zeros(shape = (A.shape[0],self.num_classes))) 
		
		for i in range(self.num_classes):
			
			temp_matrix,centroids = an.fuzzyCclassify(A,self.exemplars[i])
			temp_matrix = -temp_matrix
			temp_matrix.sort(axis = 1)
			#print temp_matrix
			# sum the first K columns
			first_K = temp_matrix[:,:K]
			sum = np.sum(first_K, axis = 1)
			# this is the distance to the first class
			#print temp_matrix
			try:
				D[:,i] = sum
			except:
				D[:,i] = np.asmatrix(sum).T
			
		
	
		
		# calculate the most likely class for each data point
		cats = np.argmin(D, axis = 1) # take the argmin of D along axis 1

		# use the class ID as a lookup to generate the original labels
		labels = self.class_labels[cats]

		if return_distances:
			return cats, labels, D

		return cats, labels

	def __str__(self):
		'''Make a pretty string that prints out the classifier information.'''
		s = "\nKNN Classifier\n"
		for i in range(self.num_classes):
			s += 'Class %d --------------------\n' % (i)
			s += 'Number of Exemplars: %d\n' % (self.exemplars[i].shape[0])
			s += 'Mean of Exemplars	 :' + str(np.mean(self.exemplars[i], axis=0)) + "\n"

		s += "\n"
		return s


	def write(self, filename):
		'''Writes the KNN classifier to a file.'''
		# extension
		return

	def read(self, filename):
		'''Reads in the KNN classifier from the file'''
		# extension
		return

