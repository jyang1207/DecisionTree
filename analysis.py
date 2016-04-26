#File analysis.py
#Author Jason Gurevitch

#collection of functions to do things to data
import numpy as np
import data
import scipy.stats
import PCAData
import scipy.cluster.vq as vq
import random
import sys
import math
import time
import scipy.spatial.distance as spdist

#returns the data range of a data object for each header given
def data_range(data, headers):
	result = []
	for header in headers:
		colmax = np.amax(data.get_data((header,)))
		#colmax = np.amax(data.census_strip_totals((header,)))
		#print 'max'
		#print colmax
		colmin = np.amin(data.get_data((header,)))
		#colmin = np.amin(data.census_strip_totals((header,)))
		result.append((colmax,colmin))
		#print 'min'
		#print colmin
	return result

	
#returns the mean of a data object for each header given
def mean(data, headers):
	result = []
	for header in headers:
		result.append(np.mean(data.get_data((header,))))
		#result.append(np.mean(data.census_strip_totals((header,))))
	return result

	
#returns the standard deviation of a data object for each header given
def stdev(data, headers):
	result = []
	for header in headers:
		result.append(np.std(data.get_data((header,))))
		#result.append(np.std(data.census_strip_totals((header,))))
	return result
	
#normalizes each column individually from 0 to 1 first by translating each col by its minimum value, and then scaling that value from 0 to 1
def normalize_columns_separately(data, headers):
	temp_matrix = data.get_data(headers)
	rows = len(temp_matrix)
	homogenous_coordinates= np.ones(shape =(rows, 1))
	temp_matrix = np.hstack((temp_matrix, homogenous_coordinates))
	min_max = data_range(data, headers)
	Tx = np.eye(len(headers)+1)
	for i in range(len(headers)):
		Tx[i, len(headers)] = -min_max[i][1]
	#print 'Tx'
	#print Tx
	Ss = np.eye(len(headers)+1)
	for i in range(len(headers)):
		colrange = min_max[i][0] - min_max[i][1]
		Ss[i, i] = 1/colrange
	#print 'Ss'
	#print Ss
	result = None
	#for i in range(data.get_raw_num_rows()):
	for i in range(rows):
		temp_row = np.matrix(temp_matrix[i, :]).T
		#temp_row = temp_matrix[i,:]
		#print 'row as vector'
		#print temp_row
		row = Tx * temp_row
		row = Ss * row
		#print temp_matrix[i, :].T * TransformationMatrix
		if result is None:
			result = row.T
		else:
			result = np.vstack((result, row.T))
		#print 'row added'
	#print result[:,range(len(headers))]
	return result[:,range(len(headers))]

#normalizes all of the columns together by translating each col by the minimum value of the data set and then scales them by the range of the data set	
def normalize_columns_together(data, headers):
	temp_matrix = data.get_data(headers)
	rows = len(temp_matrix)
	homogenous_coordinates= np.ones(shape =(rows, 1))
	temp_matrix = np.hstack((temp_matrix, homogenous_coordinates))
	min_max = data_range(data, headers)
	mins = []
	mins = []
	for i in range(len(headers)):
		 mins.append(min_max[i][1])
	totmin = min(float(num) for num in mins)
	maxes = []
	for i in range(len(headers)):
		maxes.append(min_max[i][0])
	totmax = max(float(num) for num in maxes)
	totrange = totmax - totmin
	
	Tx = np.eye(len(headers)+1)
	for i in range(len(headers)):
		Tx[i, len(headers)] = -totmin
	Ss = np.eye(len(headers)+1)
	for i in range(len(headers)):
		Ss[i, i] = 1/totrange
	result = None
	for i in range(rows):
		temp_row = np.matrix(temp_matrix[i, :]).T
		row = Tx * temp_row
		row = Ss * row
		if result is None:
			result = row.T
		else:
			result = np.vstack((result, row.T))
	return result[:,range(len(headers))]
	
	
def linear_regression(d, ind, dep):
	y = d.get_data(dep)
	A = d.get_data(ind)
	#print A
	A = np.hstack((np.ones(shape=(len(A),1)), A))
	AAinv = np.linalg.inv(np.dot(A.T, A))
	x = np.linalg.lstsq(A, y)
	#print x
	b = x[0]
	#print b
	#print np.linalg.solve(np.dot(A.T, A), np.dot(A.T, y))
	N = len(y)
	C = len(b)
	df_e = N-C
	df_r = C-1
	error = y-np.dot(A, b)
	sse = np.dot(error.T, error)/df_e
	stderr = np.sqrt(np.diagonal(sse[0,0] * AAinv))
	t = b.T/stderr
	p = 2*(1-scipy.stats.t.cdf(abs(t), df_e))
	r2 = 1-error.var()/y.var()
	#print 'r squared'
	#print r2
	result = [b, sse, r2, t, p]
	#return result

	#does a PCA analysis
def pca(d, headers, normalize = True):
	if normalize:
		matrix = normalize_columns_separately(d, headers)
	else:
		matrix = d.get_data(headers)
	#makes the covariance matrix
	begin = time.time()
	covMatrix = np.cov(matrix, rowvar = False)
	print "time to create covMatrix", time.time()-begin 
	begin = time.time()
	temp_eigenvalues, temp_eigenvectors = np.linalg.eig(covMatrix)
	print "time to perform eig", time.time()-begin
	begin = time.time()
	order = np.argsort(temp_eigenvalues).tolist()
	order.reverse()
	eigenvalues = [temp_eigenvalues[i] for i in order]
	eigenvectors = temp_eigenvectors[:,order].T
	
	mean = np.mean(matrix, axis = 0)
	difference_matrix = np.matrix(np.zeros(shape=matrix.shape))
	for row in range(len(matrix)):
		for col in range(len(matrix[0].T)):
			#print row,col
			#I'm not sure why np.mean gives me a 2d array when using noramlize cols but not when getting data, but this is a simple fix to it
			if normalize:
				difference_matrix[row,col] = matrix[row,col] - mean[0,col]
			else:
				difference_matrix[row,col] = matrix[row,col] - mean[col]
	print "time to make matrix", time.time()-begin
	begin - time.time()
	projected_data =  difference_matrix * eigenvectors.T
	print "time to perform cross prodcut", time.time()-begin
	return PCAData.PCAData(headers, projected_data, eigenvalues, eigenvectors, mean)

def kmeans_numpy(d, headers, K, whiten = True):
	A = d.get_data(headers)
	if whiten:
		W = vq.whiten(A)
	else:
		W = A
	codebook, bookerror = vq.kmeans(W, K)
	codes, error = vq.vq(W, codebook)
	return codebook,codes,error

def kmeans(data, headers, K, whiten = True, categories = None):
	A = data.get_data(headers)
	if whiten:
		W = vq.whiten(A)
	else:
		W = A
	codebook = kmeans_init(W, K, categories)
	codebook,codes,errors = kmeans_algorithm(W, codebook)
	return codebook, codes, errors

def kmeans_init(data, K, categories = None):
	#print categories
	#print 'data'
	#print data
	matrix = np.zeros(shape =len(data.T))
	if categories is None:
		for i in range(K):
			matrix = np.vstack((matrix, data[random.randint(0,K-1)]))
	else:
		data_cats = []
		for i in range(K):
			data_cats.append(np.zeros(shape = len(data[0].T)))
		for i in range(len(categories)):
			data_cats[int(categories[i,0])] = np.vstack((data_cats[int(categories[i,0])], data[i]))
		for cat in data_cats:
			matrix = np.vstack((matrix, np.mean(cat[1:].T, axis = 1)))
		
	return matrix[1:]
			
def kmeans_classify(data, cluster_means):
	#begin = time.time()
	IDs = []
	distances = []
	#is there a way to do this faster??
	for row in data:
		distance = sys.maxint
		ID = 0
		for index in range(cluster_means.shape[0]):
			newdist = 0
			clustrow = cluster_means[index]
			
			#numpy-foo is magical
			newdist = row-clustrow
			newdist = np.sum(np.square(newdist))
			
			if newdist < distance:
				distance = newdist
				ID = index
				#print 'changed'
		IDs.append(ID)
		#if distance < 0:
		#	print distance
		distances.append(np.sqrt(distance))
	#print "time for one analysis", time.time()-begin
	#print data.shape[0]*cluster_means.shape[0]*cluster_means.shape[1]
	return np.asmatrix(IDs).T, np.asmatrix(distances).T
		
def kmeans_algorithm(A, means):
	# set up some useful constants
	MIN_CHANGE = 1e-7
	MAX_ITERATIONS = 100
	D = means.shape[1]
	K = means.shape[0]
	N = A.shape[0]

	# iterate no more than MAX_ITERATIONS
	for i in range(MAX_ITERATIONS):
		# calculate the codes
		codes, errors = kmeans_classify( A, means )

		# calculate the new means
		newmeans = np.zeros_like( means )
		counts = np.zeros( (K, 1) )
		for j in range(N):
			newmeans[codes[j,0],:] += A[j,:]
			counts[codes[j,0],0] += 1.0

		# finish calculating the means, taking into account possible zero counts
		for j in range(K):
			if counts[j,0] > 0.0:
				newmeans[j,:] /= counts[j, 0]
			else:
				newmeans[j,:] = A[random.randint(0,A.shape[0]-1),:]

		# test if the change is small enough
		diff = np.sum(np.square(means - newmeans))
		means = newmeans
		if diff < MIN_CHANGE:
			break

	# call classify with the final means
	codes, errors = kmeans_classify( A, means )

	# return the means, codes, and errors
	print i, "iterations"
	return (means, codes, errors)

def fuzzyCmeans(data, headers, C):
	A = data.get_data(headers)
	
	centroids,partitionMatrix = fuzzyCinit(A, C, headers)
	partitionMatrix,centroids = fuzzyC_algorithm(A,centroids,partitionMatrix)
	#print centroids
	#print partitionMatrix
	return partitionMatrix, centroids
	
def fuzzyCinit(data, C, headers):
	centroids = np.zeros(shape =data.shape[1])
	for i in range(C):
		centroids = np.vstack((centroids, data[random.randint(0,data.shape[0]-1)]))
	centroids = centroids[1:]
	"""
	partitionMatrix = np.asmatrix(np.random.rand(data.shape[0],C))
	print partitionMatrix
	"""
	partitionMatrix = np.zeros(shape = (data.shape[0],C))
	
	#print centroids
	
	#C = centroids.shape[0]
	F = data.shape[1]
	N = data.shape[0]
	"""
	for i in range(N):
		for j in range(C):
			m = 2
			sum = 0
			for c in range(F):
				sum += ((np.sum(np.square(centroids[c]-data[i])))/((1+np.sum(np.square(centroids[j]-data[i]))))**(2/(m-1)))
			#print 'first',j,c,sum
			#print sum
			partitionMatrix[i,j]=1/sum
	#print partitionMatrix[0]
	"""
	#somehow got numpyfoo to work
	for j in range(C):
		m = 2
		thing = np.asmatrix(np.zeros(shape = N),dtype = np.complex128)
		#print thing.shape
		for c in range(C):
			top =data-centroids[c]
			bot =data-centroids[j]
			thing += np.power((np.sum(np.square(top),axis = 1)/((1+np.sum(np.square(bot),axis = 1)))),(2/(m-1)))
		#print 'second',j,c,thing
		partitionMatrix[:,j] =1/thing
	
	#print partitionMatrix[0]
	#print centroids
	return centroids, partitionMatrix 

def fuzzyCclassify(data, centroids, partitionMatrix = None):

	F = data.shape[1]
	N = data.shape[0]
	C = centroids.shape[0]
	newCentroids = centroids
	newPartMat = np.zeros(shape = (data.shape[0],C))
	errors = np.zeros_like(newPartMat)

	if partitionMatrix is not None:
		#compute centroids
		for j in range(C):
			numerator = 0
			denominator = 0
			for i in range(N):
				m = 2
				Wi = np.power(partitionMatrix[i,j],m)
				numerator +=data[i]*Wi
				denominator +=Wi
			newCentroids[j] = numerator/denominator
	
	#compute weights
	"""
	for i in range(N):
		for j in range(C):
			m = 2
			sum = 0
			topd =(np.sum(np.square(centroids[j]-data[i]))+1)
			for h in range(F):
				#sum += spdist.euclidean(newCentroids[c],data[i]+1)/(spdist.euclidean(centroids[j],data[i])+1)**(2/(m-1))
				sum += (topd/((np.sum(np.square(centroids[h]-data[i])))+1))**(2/(m-1))
			newPartMat[i,j]=1/sum
	"""
	#print centroids.shape
	#print data.shape
	for j in range(C):
		m = 2
		thing = np.asmatrix(np.zeros(shape = N))
		for c in range(C):
			top =data-newCentroids[c]
			bot =data-newCentroids[j]
			thing += (np.sum(np.square(top),axis = 1)/((1+np.sum(np.square(bot),axis = 1))))**(2/(m-1))
		newPartMat[:,j] = 1/thing
		#newPartMat[:,j] =1/np.sum(((np.sum(np.square(top),axis = 1))/(1+(np.sum(np.square(bot),axis = 1))))**(2/(m-1)))
	
	if partitionMatrix is not None:
		for i in range(C):
			if np.sum(np.square(newPartMat[:,i]))>np.sum(np.square(partitionMatrix[:,i])):
				newCentroids[i] = centroids[i]
				newPartMat[:,i] = partitionMatrix[:,i]
				#print 'the thing'
		
			
	
	#print newPartMat
	#print newCentroids
	
	return newPartMat,newCentroids

def fuzzyC_algorithm(A, centroids, partitionMatrix):
	# set up some useful constants
	MIN_CHANGE = 1e-7
	MAX_ITERATIONS = 25
	D = partitionMatrix.shape[1]
	K = partitionMatrix.shape[0]
	N = A.shape[0]
	oldPartMat = None
	# iterate no more than MAX_ITERATIONS
	for i in range(MAX_ITERATIONS):
		begin = time.time()
		#print i
		# calculate the codes
		newPartMat, newCentroids = fuzzyCclassify(A,centroids,partitionMatrix)
		
		# test if the change is small enough
		
		if oldPartMat is not None:
			diff = np.sum(np.square(partitionMatrix - newPartMat))
			#print diff
			if diff < MIN_CHANGE:
				break
		
		#partitionMatrix = newPartMat
		#centroids = newCentroids
		#print 'time for one pass',time.time()-begin

	# call classify with the final means
	partitionMatrix, centroids = fuzzyCclassify( A, centroids, partitionMatrix )
	
	#remove the largest thing from each partition matrix (since I'm getting one huge value in some of them)
	"""
	for i in range(D):
		col = partitionMatrix[:,i]
		index = np.argmax(partitionMatrix[:,i])
		#print col[index]
		col[index] = np.amax(col[:index-1:])
		#print col[index]
	"""	
	# return the means, codes, and errors
	print i, "iterations"
	return (partitionMatrix, centroids)

#function to test the functions of the data
def test(data):
	print 'data_range(data, data.get_headers())'
	print data_range(data, data.get_headers())
	print 'mean(data, data.get_headers())'
	print mean(data, data.get_headers())
	print 'stdev(data, data.get_headers())'
	print stdev(data, data.get_headers())
	print 'normalize_columns_together(data, data.get_headers())'
	print normalize_columns_together(data, data.get_headers())
	
	
if __name__ == "__main__":
	#data = data.Data('hw.csv')
	#print linear_regression(data, ['X1', 'X2', 'X3'], 'Y')
	#filename = 'data-clean.csv'
	#filename = 'data-good.csv'
	filename = 'data-noisy.csv'
	print filename
	data = data.Data(filename)
	print linear_regression(data, ['X0', 'X1'],'Y')
	#test(data)