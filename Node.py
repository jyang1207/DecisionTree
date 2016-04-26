def Node:
	def __init__(self, depth):
		self.right = None
		self.left = None
		self.feature = None
		self.threshold = None
		self.entropy = None
		self.classCounts = []
		self.depth = depth
	
	def build(self, dataMatrix, categories):
		
	def split(self):
		if depth<0:
			return
		
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