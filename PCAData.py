#File PCAData.py
#Author Jason Gurevitch
#date 3/29/2016

#subclass of Data, takes in headers and dataset to create a data object

import numpy as np
import data
import time

class PCAData(data.Data):
	def __init__(self, headers, dataMatrix, eigenvalues, eigenvectors, means):
		begin = time.time()
		data.Data.__init__(self)
		self.matrix_data = dataMatrix
		self.orig_headers = headers
		self.raw_headers = []
		self.raw_types = []
		self.header2raw = {}
		self.means = means
		print 'time to init PCAdata', time.time()-begin
		for i in range(len(headers)):
			header ="PCA%d"%(i)
			self.raw_headers.append(header)
			self.raw_types.append("numeric")
			self.header2raw[header] = i
		begin = time.time()
		self.raw_data = []
		for i in range(dataMatrix.shape[0]):
			data_row = []
			for j in range(dataMatrix.shape[1]):
				data_row.append("%.6f"%(dataMatrix[i,j]))
				#sped this up about 5 times by using 
			self.raw_data.append(data_row)
		print 'time to make raw rows', time.time()-begin
		self.header2matrix = self.header2raw
		self.eigenvectors = eigenvectors
		self.eigenvalues = eigenvalues
		self.calculate_energies()
	#calculates the percentages that each eigenvalues accounts for in the data
	def calculate_energies(self):
		total = 0.0
		for value in self.eigenvalues:
			total+= value
		self.energies = []
		for value in self.eigenvalues:
			self.energies.append(value/total)
			
	#returns the data headers
	def get_data_headers(self):
		return self.orig_headers
	#returns the eigenvalues
	def get_eigenvalues(self):
		return self.eigenvalues
	#returns the eigenvectors
	def get_eigenvectors(self):
		return self.eigenvectors
	#returns the data means
	def get_data_means(self):
		return self.means
	#returns the energies
	def get_energies(self):
		return self.energies
	
	
	def test(self):
		# Test all of the various new functions
		print "Eigenvalues:"
		print pcad.get_eigenvalues()

		print "\nEigenvectors:"
		print pcad.get_eigenvectors()

		print "\nMeans:"
		print pcad.get_data_means()

		print "\nOriginal Headers:"
		print pcad.get_data_headers()

		# Test old functions
		print "\nProjected data:"
		print pcad.get_data( pcad.get_headers() )

		print "\nRaw headers:"
		print pcad.get_raw_headers()

		print "\nRaw types:"
		print pcad.get_raw_types()

		print "\nNumber of rows:"
		print pcad.get_raw_num_rows()
		
if __name__ == "__main__":
	# This is the proper set of eigenvalues and eigenvectors for the small
	# data set of four points.
	headers = ['A','B']

	# original data
	orgdata = np.matrix([ [1,2],
						  [2,4],
						  [5,9.5],
						  [4,8.5] ])

	# means of the original data
	means = np.matrix([ 3.,  6.])

	# eigenvalues of the original data
	evals = np.matrix([16.13395443, 0.03271224])

	# eigenvectors of the original data as rows
	evecs = np.matrix([[ 0.4527601,   0.89163238],
					   [-0.89163238,  0.4527601 ]])

	# the original data projected onto the eigenvectors.
	# pdata = (evecs * (orgdata - means).T).T
	pdata = np.matrix([[-4.4720497,  -0.02777563],
					   [-2.23602485, -0.01388782],
					   [ 4.02623351, -0.19860441],
					   [ 2.68184104,  0.24026787]])
	# create a PCAData object
	pcad = PCAData( headers, pdata, evals, evecs, means)
	pcad.test()
