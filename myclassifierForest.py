#my testing classifier based on the code from last lab

import sys
import data
import classifiers
import csv
import numpy as np
import analysis as an
#read in training set and categories

#read in test set and categories
#build classifier
#classifies training matrix and prints confusion matrix
#classifies testing matrix and prints confusion matrix

def main(argv):
	if len(argv) < 5:
		print 'usage: python %s <Train CSV file><Train categories CSV file><Test CSV file><Test categories CSV file>' % (argv[0])
		exit(-1)
	try:
		train = data.Data(argv[1])
		train = train.get_data(train.get_headers())
		#train = an.pca(train, train.get_headers())
	except:
		print 'Unable to open %s' % (argv[1])
		exit(-1)

	try:
		traincat = data.Data(argv[2])
		traincat = traincat.get_data(traincat.get_headers())
	except:
		print 'Unable to open %s' % (argv[2])
		exit(-1)
	
	try:
		test = data.Data(argv[3])
		test = test.get_data(test.get_headers())
		#test = an.pca(test, test.get_headers())
	except:
		print 'Unable to open %s' % (argv[3])
		exit(-1)

	try:
		testcat = data.Data(argv[4])
		testcat = testcat.get_data(testcat.get_headers())
	except:
		print 'Unable to open %s' % (argv[4])
		exit(-1)
	
	print 'making training data'
	classifier = classifiers.ForestClassifier(train, traincat)
	print 'classifying training data'
	predtraincats = classifier.classify(train)
	
	print 'classifying testing data'
	predtestcats = classifier.classify(test)
	
	print 'training data'
	print classifier.confusion_matrix_str(classifier.confusion_matrix(traincat,predtraincats))
	
	print 'test data'
	print classifier.confusion_matrix_str(classifier.confusion_matrix(testcat, predtestcats))
	
	#print np.array(testcats.T).tolist()[0]
	
	#test.add_column('categories', 'numeric', np.array(predtestcats.T).tolist()[0])
	#test.write('testing_data_PCA_w_cats')
	
if __name__ == "__main__":
	main(sys.argv)		  
