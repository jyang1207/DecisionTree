import sys
import data
import classifiers
import csv
import numpy as np
import analysis as an


#read in data and categories
#build classifier
#perform k-fold cross validation and prints confusion matrix

def main(argv):
	if len(argv) < 4:
		print 'usage: python %s <data CSV file><data categories CSV file><k in k-fold cross validation>' % (argv[0])
		exit(-1)
		
	try:
		dataObj = data.Data(argv[1])
	except:
		print 'Unable to open %s' % (argv[1])
		exit(-1)

	try:
		cat = data.Data(argv[2])
		cat = cat.get_data(cat.get_headers())
	except:
		print 'Unable to open %s' % (argv[2])
		exit(-1)
		
	try:
		k = argv[3]
	except:
		print 'No k input'
		exit(-1)
	
	print 'building forest classifier'
	classifier = classifiers.ForestClassifier(dataObj, dataObj.get_headers(), cat)
	
	print 'performing k-fold cross validation of the given data set'
	classifier.cross_validation(cat, dataObj.get_headers(), k)
	
	
if __name__ == "__main__":
	main(sys.argv)		  

	