#File Data.py
#Author Jason Gurevitch


#-9999 represents missing data
#if any row starts with # treat is as a comment
#numeric data stored in numpy matrix as float
import csv
import numpy as np
import time

class Data:
	
	def __init__(self, filename = None):
		if filename != None:
			self.read(filename)
	
	#reads headers+types
	#then reads in raw data
	#identifies numeric data
	#builds matrix
	def read(self, filename):
		begin = time.time()
		self.raw_headers = []
		self.raw_types = []
		self.raw_data = []
		self.header2raw = {}
		fp = file(filename, 'rU')
		if filename.lower().endswith('.csv'):
			creader = csv.reader(fp)
			print "time to read in", time.time()-begin
			begin = time.time()
			self.raw_headers = creader.next()
			self.raw_types = creader.next()
			for row in creader:
				if row != []:
					self.raw_data.append(row)
			print "time to make raw data", time.time()-begin
			begin = time.time()
			
		for i in range(len(self.raw_headers)):
			key = self.raw_headers[i]
			self.header2raw[key] = i
		headers = self.get_headers()
		self.header2matrix = {} #new field
		for i in range(self.get_num_columns()):
			self.header2matrix[headers[i]] = i
		#print "self.header2matrix"
		#print self.header2matrix
		self.matrix_data = np.zeros(shape =(self.get_raw_num_rows(), self.get_num_columns()),dtype = float)
		for i in range(self.matrix_data.shape[0]):
			for j in range(self.matrix_data.shape[1]):
				
				#if (i*self.get_num_columns()+j)%1000 == 0:
				#	print i*self.get_num_columns()+j
				header = headers[j]
				matIndex = self.header2matrix[header] 
				rawIndex = self.header2raw[header]
				
				try:
					self.matrix_data[i,matIndex] = float(self.raw_data[i][rawIndex])
				except:
					#print 'adding empty value'
					self.matrix_data[i,matIndex] = -9999
				
								
		fp.close()
		print "time to make matrix", time.time()-begin
	
	def write(self, filename = None):
		if filename is None:
			doc = file("Data.csv", 'wU')
		else:
			doc = file(filename+".csv", 'w')
		writer =csv.writer(doc, delimiter=',', lineterminator='\n')
		writer.writerow(self.raw_headers)
		writer.writerow(self.raw_types)
		for row in self.raw_data:
			writer.writerow(row)
		doc.close
	
	#returns the headers that are of columns with the type numeric
	def get_headers(self):
		result = []
		for i in range(len(self.raw_headers)):
			if self.raw_types[i] == 'numeric':
				result.append(self.raw_headers[i])
		return result
			
	#returns the number of columns in the matrix
	def get_num_columns(self):
		return len(self.get_headers())
	
	#returns the row of the given index
	def get_row(self, rowIndex):
		#make this throw an exception/print an exeption
		return self.matrix_data[rowIndex,:]
		
	#returns the value at certian x,y points using the row index and header
	def get_value(self, rowIndex, header):
		#make this throw an exception/print an exeption
		return self.matrix_data[rowIndex, self.header2matrix[header]]
	
	#returns a matrix of the data with specific cols determined by the user
	#the rows can also be speicied so it will only return certain rows of the given headers
	def get_data(self, headers, rows = None):
		hindexes =[]
		for header in headers:
			hindexes.append(self.header2matrix[header])
		if rows is None:
			return self.matrix_data[:,hindexes]
		else:
			result = None
			for row in rows:
				if result is None:
					result =self.matrix_data[row, hindexes]
				else:
					result = np.vstack((result, self.matrix_data[row, hindexes]))
			return result
		
	#returns all of the raw headers
	def get_raw_headers(self):
		return self.raw_headers
	
	#returns all of the raw types
	def get_raw_types(self):
		return self.raw_types
	
	#returns the number of columns in the raw data
	def get_raw_num_columns(self):
		return len(self.raw_headers)
	
	#returns the number of rows of data
	def get_raw_num_rows(self):
		return len(self.raw_data)
	
	#returns a a raw row of a given index
	def get_raw_row(self, index):
		return self.raw_data[index]
	
	#returns a raw value of a given index and header
	def get_raw_value(self, index, header):
		if self.header2raw.has_key(header):
			return self.raw_data[index][self.header2raw[header]]
	
	
	#adds a column to the data set
	#adds -9999 to the end if there are empty values
	#deletes values at the end if there are extra
	def add_column(self, header, type, data_points):
		#print len(data_points)
		#print self.get_raw_num_rows()
		if len(data_points)<self.get_raw_num_rows():
			for i in range(self.get_raw_num_rows()-len(data_points)):
				#print 'adding empty values'
				data_points.append(-9999)
		elif len(data_points)>self.get_raw_num_rows():
			print 'cutting off last data points'
			for i in range(self.get_raw_num_rows() - len(self.raw_data)):
				data_points.pop()
		self.raw_headers.append(header)
		self.header2raw[header] = len(self.raw_headers)-1
		self.raw_types.append(type)
		for i in range(len(data_points)):
			self.raw_data[i].append(data_points[i])
		if type == 'numeric':
			self.header2matrix[header] = self.get_num_columns()-1
			#print 'the thing'
		numeric_data_points = []
		for datum in data_points:
			numeric_data_points.append(float(datum))
		#print 'before adding col'
		#print self.matrix_data
		self.matrix_data = np.hstack((self.matrix_data, np.asmatrix(numeric_data_points).T))
		#print 'after adding col'
		#print self.matrix_data
	
	#taken from lab2_test1 and expanded upon
	def test(self):
		print "\n Testing the fields and acessors for raw data\n"
		headers = self.get_raw_headers()
		print "self.get_raw_headers()"
		print self.get_raw_headers()
		print "self.get_raw_type"
		print self.get_raw_types()
		print "self.get_raw_num_columns"
		print self.get_raw_num_columns()

		try:
			print "d_get_raw_num_rows\n", self.get_raw_num_rows()
		except:
			print "class has no method get_num_raw_rows()"

		print "self.get_raw_row(1)"
		print self.get_raw_row(1)
		print "type( self.get_raw_row(1) )"
		print type( self.get_raw_row(1) )
		print "self.get_raw_value(0,headers(1))"
		print self.get_raw_value(0,headers[1])
		print "type(self.get_raw_value(0,headers(1)))"
		print type(self.get_raw_value(0,headers[1]))
		
		print "matrix"
		print self.matrix_data
		
		print "self.get_headers()"
		print self.get_headers()
		print "self.get_num_columns"
		print self.get_num_columns()
		print "self.get_row(0)"
		print self.get_row(0)
		#print "self.get_value(0, headers[1])"
		#print self.get_value(0, headers[1])
		#print "self.get_data([headers[0],headers[2]])"
		#print self.get_data([headers[0], headers[2]])
		#print "self.get_data([headers[0],headers[1]], [0,1,4,5])"
		#print self.get_data([headers[0],headers[1]], [0,1,4,5])
		
			
if __name__ == "__main__":
	#data = Data('censusdata.csv')
	data = Data('testdata2.csv')
	#data = Data('NYCZipcodeDemographics.xml')
	#data = Data('2010censusbyzipcode.csv')
	data.test()
