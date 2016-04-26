
# Author: Jason Gurevitch
# Date: 3/29/2016
#
# CS 251
# Spring 2015
#
#creates postscript files by uncomenting the lines that say self.canvas.postscript
#these images of the canvas can be converted to jpg files using imagemagick and ghsotscript

import time
import Tkinter as tk
import tkFileDialog
import tkFont as tkf
import math
import random
import View
import numpy as np
import data
import analysis
import scipy.stats
import Dialogs

# create a class to build and manage the display
class DisplayApp:

	#initializes everything
	#since this function was getting very long, I split it up into a few smaller ones
	def __init__(self, width, height):
		# create a tk object, which is the root window
		self.root = tk.Tk()
		self.n=0
		# width and height of the window
		self.initDx = width
		self.initDy = height
		
		# set up the geometry for the window
		self.root.geometry( "%dx%d+50+30" % (self.initDx, self.initDy) )

		# set the title of the window
		self.root.title("Data Display Revolution")

		# set the maximum size of the window for resizing
		self.root.maxsize( 1600, 900 )

		# setup the menus
		self.buildMenus()

		# build the controls
		self.buildControls()

		# build the Canvas
		self.buildCanvas()

		# bring the window to the front
		self.root.lift()

		# - do idle events here to get actual canvas size
		self.root.update_idletasks()

		# now we can ask the size of the canvas
		print self.canvas.winfo_geometry()

		# set up the key bindings
		self.setBindings()
		
		self.initFields()
		
		self.view = View.View()
		self.view.screen[0, 0] = float(width-width/10)
		self.view.screen[0, 1] = float(height-height/10)
		self.initAxes()
		self.buildAxes()
		
	#initializes the fields
	def initFields(self):
		
		self.data = None
		
		self.objects = [] 
		self.label = None 
		self.baseClick = None 
		
		self.dataMatrix = None
		self.dataheaders = []
		self.colorMatrix = None
		self.colorResult = None
		self.sizeResult = None
		self.sizeMatrix = None	
		self.glinText = None
		
		self.slope = []
		self.intercept = None
		self.r_squared = None
		self.p_value = None
		self.std_err = None		
		
		self.gXticks = []
		self.gXlabels = []
		self.gYticks = []
		self.gYlabels=[]
		self.gZticks = []
		self.gZlabels =[]
		self.gAxes = []
		self.gLabels = []
		
		self.gRegressLine = None
		self.gRegressLines = []
		self.endpoints = None
		
		self.pastSelections = []
		
		self.PCA = None
		self.PCAanalysis = []
		self.PCAs = 0
		
		
		self.clusternumber = 0
		
		self.colors = ['red','saddle brown','orange','yellow','lime green','lawn green','cyan','blue','navy','blue violet','purple','grey','bisque2','black','aquamarine']
		#self.uniqueColors = False
	
	#initializes the axes and the tickmarks
	def initAxes(self):
		self.axes = np.matrix([[0, 0, 0, 1],
								[1, 0, 0, 1],
								[0, 0, 0, 1],
								[0, 1, 0, 1],
								[0, 0, 0, 1],
								[0, 0, 1, 1]])
								
		self.xticks = np.matrix([[0.25, 0, 0, 1],
								[0.25, 0.05, 0, 1],
								[0.5, 0, 0, 1],
								[0.5, 0.05, 0, 1],
								[0.75, 0, 0, 1],
								[0.75, 0.05, 0, 1],
								[1, 0, 0, 1],
								[1, 0.05, 0, 1]])
		
		self.yticks = np.matrix([[0, 0.25, 0, 1],
								[0.05, 0.25, 0, 1],
								[0, 0.5, 0, 1],
								[0.05, 0.5, 0, 1],
								[0, 0.75, 0, 1],
								[0.05, 0.75, 0, 1],
								[0, 1, 0, 1],
								[0.05, 1, 0, 1]])
				
		self.zticks = np.matrix([[0, 0, 0.25, 1],
								[0, 0.05, 0.25, 1],
								[0, 0, 0.5, 1],
								[0, 0.05, 0.5, 1],
								[0, 0, 0.75, 1],
								[0, 0.05, 0.75, 1],
								[0, 0, 1, 1],
								[0, 0.05, 1, 1]])
		
	#resets all of the fields
	def resetData(self):
		for obj in self.objects:
			self.canvas.delete(obj)
		self.objects = []
		for axis in self.gAxes:
			self.canvas.delete(axis)
		self.gAxes = []
		for label in self.gLabels:
			self.canvas.delete(label)
		self.gLabels = []
		
		for tick in self.gXticks:
			self.canvas.delete(tick)
		self.gXticks = []
		for tick in self.gYticks:
			self.canvas.delete(tick)
		self.gYticks = []	
		for tick in self.gZticks:
			self.canvas.delete(tick)
		self.gZticks = []
		
		for label in self.gXlabels:
			self.canvas.delete(label)
		self.gXlabels = []
		for label in self.gYlabels:
			self.canvas.delete(label)
		self.gYlabels = []
		for label in self.gZlabels:
			self.canvas.delete(label)
		self.gZlabels = []
		
		for line in self.gRegressLines:
			self.canvas.delete(line)
		self.gRegressLines = []
		
		if self.gRegressLine is not None:
			self.canvas.delete(self.gRegressLine)
		self.gRegressLine = None
		
		if self.glinText is not None:
			self.canvas.delete(self.glinText)
		self.glinText = None
		
		self.slope = []
		self.intercept = None
		self.r_squared = None
		self.p_value = None
		self.std_err = None
		
		
		
	def buildAxes(self):
		self.resetData()
		vtm = self.view.build()
		#print self.axes.T
		pts = (vtm * self.axes.T).T
		#create tick marks
		xpts = (vtm*self.xticks.T).T
		ypts = (vtm*self.yticks.T).T
		zpts = (vtm*self.zticks.T).T
		
		if self.dataMatrix is not None:
			#length =  len(self.dataMatrix.T)
			#temp_matrix = self.dataMatrix.T[[0,1,2,-1]]
			#print temp_matrix
			datapts = (vtm*self.dataMatrix.T).T
		#number2letter= {0: 'X', 1: 'Y', 2: 'Z'}
		number2axes= {0: [0, 10], 1: [10, 0], 2: [5,5]}
		
		for i in range(3):
			axis = self.canvas.create_line(pts[2*i,0], pts[2*i,1], pts[2*i+1,0], pts[2*i+1,1])
			self.gAxes.append(axis)
			
		for i in range(len(self.dataheaders)):
			text = self.canvas.create_text(pts[2*i+1,0]+number2axes[i][0]*1/self.view.extent[0,0], pts[2*i+1,1]+number2axes[i][1]*1/self.view.extent[0,1], text = self.dataheaders[i])
			self.gLabels.append(text)
		
		if self.dataMatrix is not None:
			#this part is so much more difficult incorporating both
			if self.dataheaders[0] in self.data.get_headers():
				xdrange = analysis.data_range(self.data, (self.dataheaders[0],)) 
			else:
				xdrange = analysis.data_range(self.PCA, (self.dataheaders[0],)) 
			xrange = xdrange[0][0] - xdrange[0][1]
			
			if self.dataheaders[1] in self.data.get_headers():
				ydrange = analysis.data_range(self.data, (self.dataheaders[1],)) 
			else:
				ydrange = analysis.data_range(self.PCA, (self.dataheaders[1],)) 
			yrange = ydrange[0][0] - ydrange[0][1]
			
			if len(self.dataheaders) >2:
				if self.dataheaders[2] in self.data.get_headers():
					zdrange = analysis.data_range(self.data, (self.dataheaders[2],)) 
				else:
					zdrange = analysis.data_range(self.PCA, (self.dataheaders[2],)) 
				zrange = zdrange[0][0] - zdrange[0][1]
				
			number2xlabel = {0:"%.2f"%(xdrange[0][1]+xrange/4.0), 1: "%.2f"%(xdrange[0][1]+2*xrange/4.0), 2: "%.2f"%(xdrange[0][1]+3*xrange/4.0), 3:"%.2f"%(xdrange[0][0])}
			number2ylabel = {0:"%.2f"%(ydrange[0][1]+yrange/4.0), 1: "%.2f"%(ydrange[0][1]+2*yrange/4.0), 2: "%.2f"%(ydrange[0][1]+3*yrange/4.0), 3:"%.2f"%(ydrange[0][0])}
			if len(self.dataheaders) >2:
				number2zlabel = {0:"%.2f"%(zdrange[0][1]+zrange/4.0), 1: "%.2f"%(zdrange[0][1]+2*zrange/4.0), 2: "%.2f"%(zdrange[0][1]+3*zrange/4.0), 3:"%.2f"%(zdrange[0][0])}
			for i in range(4):
				tick = self.canvas.create_line(xpts[2*i,0], xpts[2*i,1], xpts[2*i+1,0], xpts[2*i+1,1])
				self.gXticks.append(tick)
				text = self.canvas.create_text(xpts[2*i+1,0], xpts[2*i+1,1], text= number2xlabel[i])
				self.gXlabels.append(text)
				
				tick = self.canvas.create_line(ypts[2*i,0], ypts[2*i,1], ypts[2*i+1,0], ypts[2*i+1,1])
				self.gYticks.append(tick)
				text = self.canvas.create_text(ypts[2*i+1,0], ypts[2*i+1,1], text= number2ylabel[i])
				self.gYlabels.append(text)
				
				if len(self.dataheaders) >2:
					tick = self.canvas.create_line(zpts[2*i,0], zpts[2*i,1], zpts[2*i+1,0], zpts[2*i+1,1])
					self.gZticks.append(tick)
					text = self.canvas.create_text(zpts[2*i+1,0], zpts[2*i+1,1], text= number2zlabel[i])
					self.gZlabels.append(text)
				
		
		if self.dataMatrix is not None:
			rows =self.rows
			lenx = 3*1.0/self.view.extent[0,0]
			leny = 3*1.0/self.view.extent[0,0]
			for i in range(rows):
				dx = 1
				r = 0.5*255
				g = 0.5*255
				b = 0.5*255
				color ="#%02x%02x%02x" %(r,g,b)
				if self.colorResult is not None:
					if self.colorVar.get() == 1:
						color = self.colors[int(self.colorMatrix[i,0])]
						#print self.colorMatrix[i,0]
					else:
						if self.colorResult in self.data.get_headers():
							dataRange = analysis.data_range(self.data, (self.colorResult,))
						elif self.colorResult in self.PCA.get_headers():
							dataRange = analysis.data_range(self.PCA, (self.colorResult,))
						else:
							print "something is wrong, your color does not exist"
						middle = (dataRange[0][0]+dataRange[0][1])/2
						alpha = 1.0/(1.0+math.e**(-10*(self.colorMatrix[i,0]-0.5))) 
						#alpha = 1.0/(1.0+math.e**(-(1.0/dataRange[0][1])*(self.colorMatrix[i,0]-middle)))
						#print alpha
						r = (1.0- alpha)*255
						g = (1.0 - alpha) *255
						b = alpha*255
						color ="#%02x%02x%02x" %(r,g,b)
				if self.sizeResult is not None:
					dx = self.sizeMatrix[i,0]*5
					#print dx
				self.objects.append(self.canvas.create_oval(datapts[i,0]-lenx*dx, 
										datapts[i,1]-leny*dx,
										datapts[i,0] + lenx*dx,
										datapts[i,1]+leny*dx,
										fill = color,
										outline ='',))
		#self.canvas.postscript(file = "imag%d.ps"%(self.n) , colormode = 'color')
		#self.n+=1
		self.scaleText.set("%.2f, %.2f"%(self.view.extent[0,0], self.view.extent[0,1]))
		self.resetAxes()
		
			
	def updateAxes(self):
		#begin = time.time()
		vtm = self.view.build()
		pts = (vtm * self.axes.T).T	

		xpts = (vtm*self.xticks.T).T
		ypts = (vtm*self.yticks.T).T
		zpts = (vtm*self.zticks.T).T
		
		if self.dataMatrix is not None:
			#temp_matrix = self.dataMatrix.T[0:4]
			datapts = (vtm*self.dataMatrix.T).T
		number2axes= {0: [0, 10], 1: [10, 0], 2: [5,5]}
		for i in range(3):
			self.canvas.coords(self.gAxes[i],pts[2*i,0],pts[2*i,1],pts[2*i+1,0],pts[2*i+1,1])
		for i in range(len(self.gLabels)):
			self.canvas.coords(self.gLabels[i], pts[2*i+1,0]+number2axes[i][0]*1/self.view.extent[0,0], pts[2*i+1,1]++number2axes[i][1]*1/self.view.extent[0,1])
		
		if self.dataMatrix is not None:
			for i in range(4):
				self.canvas.coords(self.gXticks[i], xpts[2*i,0], xpts[2*i,1], xpts[2*i+1,0], xpts[2*i+1,1])
				self.canvas.coords(self.gXlabels[i], xpts[2*i+1,0], xpts[2*i+1,1])

				self.canvas.coords(self.gYticks[i], ypts[2*i,0], ypts[2*i,1], ypts[2*i+1,0], ypts[2*i+1,1])
				self.canvas.coords(self.gYlabels[i], ypts[2*i+1,0], ypts[2*i+1,1])				
				
				if len(self.dataheaders) >2:
					self.canvas.coords(self.gZticks[i], zpts[2*i,0], zpts[2*i,1], zpts[2*i+1,0], zpts[2*i+1,1])
					self.canvas.coords(self.gZlabels[i], zpts[2*i+1,0], zpts[2*i+1,1])
				
		lenx = 3*1.0/self.view.extent[0,0]
		leny = 3*1.0/self.view.extent[0,0]
		#begin = time.time()
		for i in range(len(self.objects)):
			dx = 1
			if self.sizeResult is not None:
				dx = self.sizeMatrix[i,0]*5
				#print dx
			self.canvas.coords(self.objects[i], datapts[i,0]-(lenx*dx), 
										datapts[i,1]-(leny*dx),
										datapts[i,0] + (lenx*dx),
										datapts[i,1]+(leny*dx))		
		self.scaleText.set("%.2f, %.2f"%(self.view.extent[0,0], self.view.extent[0,1]))
		
		#I did this instead of an updateFits method because I didn't want to have to create another VTM
		if self.gRegressLine is not None: 
			pts2 = (vtm*self.endpoints.T).T
			self.canvas.coords(self.glinText, pts2[1,0], pts2[1,1])
			if len(self.dataheaders) == 2:
				self.canvas.coords(self.gRegressLine, pts2[0,0], pts2[0,1], pts2[1,0], pts2[1,1])
			else:
				#this came from the multiple ways I tried to make my plane
				#self.canvas.coords(self.gRegressLine, pts2[0,0],pts2[2,1],pts2[1,0],pts2[3,1])
				#self.canvas.coords(self.gRegressLine, pts2[0,0],pts2[0,1],
				#								pts2[1,0], pts2[1,1],
				#								pts2[2,0], pts2[2,1],
				#								pts2[3,0], pts2[3,1])
				pass
				
		if self.gRegressLines != []:
			pts2 = (vtm*self.endpoints.T).T
			self.canvas.coords(self.gRegressLines[0],pts2[0,0], pts2[0,1], pts2[1,0], pts2[1,1])
			self.canvas.coords(self.gRegressLines[1],pts2[2,0], pts2[2,1], pts2[3,0], pts2[3,1])
			self.canvas.coords(self.gRegressLines[2],pts2[0,0], pts2[0,1], pts2[2,0], pts2[2,1])
			self.canvas.coords(self.gRegressLines[3],pts2[1,0], pts2[1,1], pts2[3,0], pts2[3,1])		
				
			self.canvas.coords(self.glinText, pts2[1,0], pts2[1,1])
		

		#print time.time()-begin
		#self.canvas.postscript(file = "imag%d.ps"%(self.n) , colormode = 'color')
		#self.n+=1
			
	def buildMenus(self):
		
		# create a new menu
		menu = tk.Menu(self.root)

		# set the root menu to our new menu
		self.root.config(menu = menu)

		# create a variable to hold the individual menus
		menulist = []

		# create a file menu
		filemenu = tk.Menu( menu )
		menu.add_cascade( label = "File", menu = filemenu )
		menulist.append(filemenu)

		# create another menu for kicks
		cmdmenu = tk.Menu( menu )
		menu.add_cascade( label = "Regression", menu = cmdmenu )
		menulist.append(cmdmenu)
		
		pcamenu = tk.Menu(menu)
		menu.add_cascade( label = "PCA", menu = pcamenu )
		menulist.append(pcamenu)

		clustmenu = tk.Menu(menu)
		menu.add_cascade( label = "Clustering", menu = clustmenu )
		menulist.append(clustmenu)
		
		# menu text for the elements
		# the first sublist is the set of items for the file menu
		# the second sublist is the set of items for the option menu
		menutext = [ ['Open \xE2\x8C\x98-O', 'Build Data \xE2\x8C\x98-B', 'Quit\xE2\x8C\x98-Q' ],
					 [ 'Linear Regression', 'Display Linear Analysis', '-' ] ,
					 [ 'Execute PCA', 'Show PCA table', 'Align data to EigenVectors' ] ,
					 ['K means', 'K means on PCA', 'Fuzzy C means']]

		# menu callback functions (note that some are left blank,
		# so that you can add functions there if you want).
		# the first sublist is the set of callback functions for the file menu
		# the second sublist is the set of callback functions for the option menu
		menucmd = [ [self.handleOpen, self.handlePlotData, self.handleQuit],
					[self.handleLinearRegression, self.viewLinAna, None],
					[self.createPCA, self.showPCATable, self.alignPCA],
					[self.cluster, self.clusterPCA, self.fuzzyCluster]]
		
		# build the menu elements and callbacks
		for i in range( len( menulist ) ):
			for j in range( len( menutext[i]) ):
				if menutext[i][j] != '-':
					menulist[i].add_command( label = menutext[i][j], command=menucmd[i][j] )
				else:
					menulist[i].add_separator()

	# create the canvas object
	def buildCanvas(self):
		self.canvas = tk.Canvas( self.root, width=self.initDx, height=self.initDy, bg= 'white' )
		self.canvas.pack( expand=tk.YES, fill=tk.BOTH )
		return

	# build a frame and put controls in it
	def buildControls(self):

		### Control ###
		# make a control frame on the right
		rightcntlframe = tk.Frame(master = self.root, width = 100, height= self.initDy)
		rightcntlframe.pack(side = tk.RIGHT, padx=2, pady=2, fill=tk.Y)

		# make a separator frame
		sep = tk.Frame( self.root, height=self.initDy, width=2, bd=1, relief=tk.SUNKEN )
		sep.pack( side=tk.RIGHT, padx = 2, pady = 2, fill=tk.Y)

		# make a menubutton
		"""
		label = tk.Label( rightcntlframe, text="Scaling Selector", width=20 )
		label.pack( side=tk.TOP )
		self.scaleOption = tk.StringVar( self.root )
		self.scaleOption.set("Together")
		scaleMenu = tk.OptionMenu( rightcntlframe, self.scaleOption, 
		"Together", "X", "Y") # can add a command to the menu
		scaleMenu.pack(side=tk.TOP)"""
		
		label = tk.Label( rightcntlframe, text="Control Panel", width=20 )
		label.pack( side=tk.TOP )
		
		ZtoHeight = tk.Button(rightcntlframe, text ="Set Z to the height", command = self.ZtoHeight)
		ZtoHeight.pack(side = tk.TOP)
		
		#make something here to determine if they use unique colors
		
		# make a button in the frame
		# and tell it to call the handleButton method when it is pressed.
		button1 = tk.Button(rightcntlframe, text = "Reset Axes", command = self.resetAxes)
		button1.pack(side=tk.TOP)	
		
		#makes a frame inside the rightcntlframe to have a listbox and a scrollbar of old selections
		label3 = tk.Label(rightcntlframe, text = "past selections", width = 20)
		label3.pack(side = tk.TOP)
		pastFrame = tk.Frame(rightcntlframe)
		scrollbar = tk.Scrollbar(pastFrame, orient = tk.VERTICAL)
		self.pastListbox = tk.Listbox(pastFrame, selectmode = tk.SINGLE, exportselection = 0, yscrollcommand = scrollbar.set)
		scrollbar.config(command = self.pastListbox.yview)
		scrollbar.pack(side = tk.RIGHT, fill = tk.Y)
		self.pastListbox.pack()
		pastFrame.pack(side = tk.TOP)
		
		selectOldButton = tk.Button(rightcntlframe, text = "open old data", command = self.openOld)
		selectOldButton.pack(side = tk.TOP)
				
		label4 = tk.Label(rightcntlframe, text = "PCA analysis", width = 20)
		label4.pack(side = tk.TOP)
		PCAFrame = tk.Frame(rightcntlframe)
		PCAscrollbar = tk.Scrollbar(PCAFrame, orient = tk.VERTICAL)
		self.PCAListbox = tk.Listbox(PCAFrame, selectmode = tk.SINGLE, exportselection = 0, yscrollcommand = PCAscrollbar.set)
		scrollbar.config(command = self.PCAListbox.yview)
		PCAscrollbar.pack(side = tk.RIGHT, fill = tk.Y)
		self.PCAListbox.pack()
		PCAFrame.pack(side = tk.TOP)
		"""
		selectOldPCAButton = tk.Button(rightcntlframe, text = "open PCA analysis", command = self.openOldPCA)
		selectOldPCAButton.pack(side = tk.TOP)
		"""
		deletePCAButton = tk.Button(rightcntlframe, text = "delete PCA analysis", command = self.deletePCA)
		deletePCAButton.pack(side = tk.TOP)
		
		"""
		clusteringButton = tk.Button(rightcntlframe, text = "perform clustering analysis", command = self.cluster)
		clusteringButton.pack(side = tk.TOP)

		clusteringButton2 = tk.Button(rightcntlframe, text = "perform clustering analysis on PCA", command = self.clusterPCA)
		clusteringButton2.pack(side = tk.TOP)
		
		fuzzyButton = tk.Button(rightcntlframe, text = "perform fuzzy C means clustering", command = self.fuzzyCluster)
		fuzzyButton.pack()
		"""
		writeButton = tk.Button(rightcntlframe, text = "write out data", command = self.write_data)
		writeButton.pack(side = tk.TOP)
		
		var = tk.IntVar()
		checkbox = tk.Checkbutton(rightcntlframe, text = "Use Distinct Colors", variable = var)
		self.colorVar = var
		checkbox.pack(side = tk.TOP)
		
		self.scaleText = tk.StringVar()
		self.scaleText.set("1,1")
		scaleLabel = tk.Label(rightcntlframe, textvariable =self.scaleText , width = 10)
		scaleLabel.pack(side = tk.BOTTOM)
		
		label2 = tk.Label(rightcntlframe, text = "Amount scaled:", width = 12)
		label2.pack(side = tk.BOTTOM)
		
		
		
		return
	
	def write_data(self):
		if self.data is None:
			print 'you don\'t have any data to save'
			return
		variables = Dialogs.WriteDialog(self.root)
		if variables.result == []:
			print 'you cancelled out of the writing process'
			return
		if variables.pca:
			if self.PCA is None:
				print 'you didn\'t perform a PCA analysis'
				return
			self.PCA.write(variables.result)
		else:
			self.data.write(variables.result)
	
	def fuzzyCluster(self):
		if self.data is None:
			print 'you have no data'
			return
		variables = Dialogs.ClusterDialog(self.root, self.data.get_headers())
		if variables.result == []:
			print 'you didn\'t pick anything'
			return
		self.clusternumber+=1
		#print variables.numclusters
		#self.uniqueColors = variables.distColors
		#print self.uniqueColors
		partitionMatrix, centroids = analysis.fuzzyCmeans(self.data, variables.result, variables.numclusters)
		#print partitionMatrix
		#print centroids
		#print self.data.matrix_data
		#I add the col to the data so I can cluster the data itself to the PCA, rather than the transformed data
		#self.data.add_column('clustering%d'%(self.clusternumber),'numeric', codes)
		i = 0
		for col in partitionMatrix.T:
			self.data.add_column('afinity to cluster %d'%(i), 'numeric', col)
			i+=1
		
	
	def cluster(self):
		if self.data is None:
			print 'you have no data'
			return
		variables = Dialogs.ClusterDialog(self.root, self.data.get_headers())
		if variables.result == []:
			print 'you didn\'t pick anything'
			return
		self.clusternumber+=1
		#print variables.numclusters
		#self.uniqueColors = variables.distColors
		#print self.uniqueColors
		codebook, codes, errors = analysis.kmeans(self.data, variables.result, variables.numclusters)
		#I add the col to the data so I can cluster the data itself to the PCA, rather than the transformed data
		self.data.add_column('clustering%d'%(self.clusternumber),'numeric', codes)
		
	def clusterPCA(self):
		if self.PCAanalysis != [] and self.PCAListbox.curselection() != ():
			self.PCA = self.PCAanalysis[self.PCAListbox.curselection()[0]]
		
		if self.PCA is None:
			print 'you have no data'
			return
		variables = Dialogs.ClusterDialog(self.root, self.PCA.get_headers())
		if variables.result == []:
			print 'you didn\'t pick anything'
			return
		self.clusternumber+=1
		#print variables.numclusters
		#self.uniqueColors = variables.distColors
		#print self.uniqueColors
		codebook, codes, errors = analysis.kmeans(self.PCA, variables.result, variables.numclusters)
		self.PCA.add_column('clustering%d'%(self.clusternumber),'numeric', codes)

		
	def ZtoHeight(self):
		self.view.vrp = np.matrix([1.0, 0.5, 0.5])
		self.view.u = np.matrix([0.0, -1.0, 0.0])
		self.view.vup = np.matrix([0.0, 0, 1.0])
		self.view.vpn = np.matrix([-1.0, 0.0, 0.0])
		self.updateAxes()
	
	#displays the information of the linear analysis
	def viewLinAna(self):
		infoView = tk.Toplevel()
		infoView.title("Linear Analysis")
		infoView.geometry("%dx%d%+d%+d" % (300, 100, 250, 250))
		if self.gRegressLine is not None:
			info = tk.Message(infoView, width = 250, text = "Slope: %.2f Intercept: %.2f \n R Squared: %.2f P value: %.2f Error: %.2f"%(self.slope[0],self.intercept,self.r_squared,self.p_value,self.std_err))
		elif self.gRegressLines != []:
			info = tk.Message(infoView, width = 250, text = "Slope X0: %s Slope X1: %s, Intercept: %s \n R Squared: %s P value: %s Error: %s"%
			(np.array_str(self.slope[0], precision = 2),np.array_str(self.slope[1],precision = 2), np.array_str(self.intercept, precision = 2),
			np.array_str(self.r_squared,precision = 2),np.array_str(self.p_value,precision = 2),np.array_str(self.std_err, precision = 2)))
		else :
			info = tk.Message(infoView, width = 250, text = "you are not doing a linear analysis")
	
		info.pack()
		button = tk.Button(infoView, text = "close", command = infoView.destroy)
		button.pack()
		
	def setBindings(self):
		# bind mouse motions to the canvas
		self.canvas.bind( '<Button-1>', self.handleMouseButton1 )
		self.canvas.bind( '<Control-Button-1>', self.handleMouseButton2 )
		self.canvas.bind( '<Button-2>', self.handleMouseButton2 )
		self.canvas.bind( '<Button-3>', self.handleMouseButton3)
		self.canvas.bind( '<B1-Motion>', self.handleMouseButton1Motion )
		self.canvas.bind( '<B2-Motion>', self.handleMouseButton2Motion )
		self.canvas.bind( '<B3-Motion>', self.handleMouseButton3Motion )
		self.canvas.bind( '<Control-B1-Motion>', self.handleMouseButton2Motion )
		self.canvas.bind( '<Configure>', self.resize )
		
		
		# bind command sequences to the root window
		self.root.bind( '<Control-q>', self.handleQuit )
		self.root.bind('<Control-o>', self.handleOpen)
		self.root.bind('<Control-b>', self.handlePlotData)
		
		
	def resize(self, event= None):
		self.view.screen[0, 0] = float(self.canvas.winfo_width())
		self.view.screen[0, 1] = float(self.canvas.winfo_height())
		self.updateAxes()
		
	#resets the axes
	def resetAxes(self):
		self.view.reset()
		self.view.screen[0, 0] = float(self.canvas.winfo_width())
		self.view.screen[0, 1] = float(self.canvas.winfo_height())
		self.updateAxes()
		
	def handleQuit(self, event=None):
		print 'Terminating'
		self.root.destroy()
		
	#opens a file from the Past Selections listbox
	def openOld(self):
		if self.pastSelections == []:
			print "you have no old data"
			return
		if self.pastListbox.curselection != ():
			self.data = data.Data(self.pastSelections[self.pastListbox.curselection()[0]])
		else:
			print 'you have no data selected'
	
	#sets up the linear regression, but does not build it(passes it into the build function)
	def handleLinearRegression(self):
		if self.data == None:
			print "you don't have data"
			return
		variables = Dialogs.LinRegressDialog(self.root, self.data.get_headers())
		colorbox = Dialogs.ColorDialog(self.root, self.data.get_headers())
		if variables.result == []:
			return
		if colorbox.resultc != []:
			self.colorMatrix =analysis.normalize_columns_separately(self.data, (colorbox.resultc,))
			self.colorResult = colorbox.resultc
		else:
			self.colorResult = None
			self.colorMatrix = None
		if colorbox.results != []:
			self.sizeMatrix =analysis.normalize_columns_separately(self.data, (colorbox.results,))
			self.sizeResult = colorbox.results
		else:
			self.sizeResult = None
			self.sizeMatrix = None

		self.dataheaders = variables.result
		self.resetData()
		#self.resetAxes()
		self.view.reset()
		self.view.screen[0, 0] = float(self.canvas.winfo_width())
		self.view.screen[0, 1] = float(self.canvas.winfo_height())
		self.buildLinearRegression()
	
	#builds a linear regression, can either create a line for 2 Dimensional data, or a plane(displayed as a rectangle) for 3 Dimensional data
	def buildLinearRegression(self):
		#self.uniqueColors = False
		if self.gRegressLine is not None:
			self.canvas.delete(self.gRegressLine)
			self.canvas.delete(self.glinText)
		self.gRegressLine = None
		temp_matrix = analysis.normalize_columns_separately(self.data, self.dataheaders)
		self.rows = len(temp_matrix)
		if len(self.dataheaders) == 2:
			temp_matrix = np.hstack((temp_matrix, np.zeros(shape=(self.rows,1))))
		self.dataMatrix = np.hstack((temp_matrix, np.ones(shape=(self.rows,1))))
		self.buildAxes()
		if len(self.dataheaders) == 2: 
			slope, self.intercept, r_value, self.p_value, self.std_err = scipy.stats.linregress(self.data.get_data(self.dataheaders))
			self.slope.append(slope)
			self.r_squared = r_value**2
			data_range = analysis.data_range(self.data, self.dataheaders)
			high = ((data_range[0][0]*self.slope[0] + self.intercept)-data_range[1][1])/(data_range[1][0]-data_range[1][1])
			low =  ((data_range[0][1]*self.slope[0] + self.intercept)-data_range[1][1])/(data_range[1][0]-data_range[1][1])
			#print low,high
			self.endpoints = np.matrix([[0, low, 0, 1],
											[1, high, 0, 1]])
			vtm = self.view.build()
			pts = (vtm * self.endpoints.T).T
			self.gRegressLine = self.canvas.create_line(pts[0,0], pts[0,1], pts[1,0], pts[1,1], fill = "red")
			linText = ("Slope: %.3f, Intercept: %.3f, R Squared: %.3f"%(slope, self.intercept, r_value**2))
			self.glinText = self.canvas.create_text(pts[1,0], pts[1,1], text = linText)
		else:
			regressstuffs = analysis.linear_regression(self.data, self.dataheaders[:2], [self.dataheaders[2],])
			self.intercept = regressstuffs[0][0]
			self.slope.append(regressstuffs[0][1])
			self.slope.append(regressstuffs[0][2])
			self.std_err = regressstuffs[1]
			self.r_squared = regressstuffs[2]
			self.p_value = regressstuffs[4]
			#print intercept
			
			data_range = analysis.data_range(self.data, self.dataheaders)
			highx0 = ((data_range[0][0]*self.slope[0] + self.intercept)-data_range[2][1])/(data_range[2][0]-data_range[2][1])
			lowx0 =  ((data_range[0][1]*self.slope[0] + self.intercept)-data_range[2][1])/(data_range[2][0]-data_range[2][1])
			#print lowx0, highx0

			highx1 = ((data_range[1][0]*self.slope[1] + self.intercept)-data_range[2][1])/(data_range[2][0]-data_range[2][1])
			lowx1 =  ((data_range[1][1]*self.slope[1] + self.intercept)-data_range[2][1])/(data_range[2][0]-data_range[2][1])
			#print lowx1,highx1
			
			#x1 goes in the x direction, x2 in y, dep goes in Z
			self.endpoints = np.matrix([[0, 0, lowx0, 1],
										[1, 0, highx0, 1],
										[0, 0, lowx1, 1],
										[0, 1, highx1, 1]])
			vtm = self.view.build()
			pts = (vtm * self.endpoints.T).T
			#print pts
			#self.gRegressLine = self.canvas.create_rectangle(pts[0,0],pts[2,1],pts[1,0],pts[3,1])
			self.gRegressLines = []
			#I made each line in the plane a different color because I wasn't sure if things were working right so I wanted to be able to differentiate them
			#I think this should be a 3D visualization of the linear regression, but I might have done something horribly wrong(it seems to work as a plane) for 
			self.gRegressLines.append(self.canvas.create_line(pts[0,0], pts[0,1], pts[1,0], pts[1,1], fill = "red"))
			self.gRegressLines.append(self.canvas.create_line(pts[2,0], pts[2,1], pts[3,0], pts[3,1], fill = "green"))
			self.gRegressLines.append(self.canvas.create_line(pts[0,0], pts[0,1], pts[2,0], pts[2,1], fill = "blue"))
			self.gRegressLines.append(self.canvas.create_line(pts[1,0], pts[1,1], pts[3,0], pts[3,1], fill = "black"))
			#self.gRegressLine = self.canvas.create_polygon(pts[0,0],pts[0,1],
			#												pts[1,0], pts[1,1],
			#												pts[2,0], pts[2,1],
			#												pts[3,0], pts[3,1], fill = '', outline = "red")
			linText = ("X0 Slope: %.3f, X1 Slope: %.3f, Intercept: %.3f, R Squared: %.3f"%(self.slope[0], self.slope[1], self.intercept, self.r_squared))
			self.glinText = self.canvas.create_text(pts[1,0], pts[1,1], text = linText)
	"""
	def openOldPCA(self):
		if self.PCAanalysis != [] and self.PCAListbox.curselection() != ():
			self.PCA = self.PCAanalysis[self.PCAListbox.curselection()[0]]
		else:
			print "you have no data selected"
	"""
	def deletePCA(self):
		if self.PCAanalysis != [] and self.PCAListbox.curselection() != ():
			self.PCAanalysis.pop(self.PCAListbox.curselection()[0])
			self.PCAListbox.delete(self.PCAListbox.curselection()[0])
		else:
			print "you have no data selected"
	
	#performs a PCA analysis on the current data
	def createPCA(self):
		if self.data == None:
			print "you don't have data"
			return
		variables = Dialogs.PCADialog(self.root, self.data.get_headers())
		if variables.result == [] and not variables.all:
			print "you didn't pick anything"
			return
		headers = variables.result
		if variables.all:
			headers = self.data.get_headers()
		pca = analysis.pca(self.data, headers, variables.normalize)
		self.PCA = pca
		
		if pca not in self.PCAanalysis:
			self.PCAs +=1
			self.PCAanalysis.append(pca)
			self.PCAListbox.insert(tk.END, "PCA%d"%(self.PCAs))
		
	
	#aligns the data to the eigenvectors in the selected axes
	def alignPCA(self):
		if self.PCAanalysis != [] and self.PCAListbox.curselection() != ():
			self.PCA = self.PCAanalysis[self.PCAListbox.curselection()[0]]		
		
		if self.PCA is None:
			print 'you don\'t have any data'
			return
		
		headers = self.PCA.get_headers() + self.data.get_headers()
		
		variables = Dialogs.selectPCAData(self.root, headers)
		if variables.result == []:
			print "you didn't pick anything"
			return
			
		self.dataheaders = []
		
		if variables.result[0] < len(self.PCA.get_headers()):
			header = self.PCA.get_headers()[variables.result[0]]
			self.dataMatrix =analysis.normalize_columns_separately(self.PCA, (header,))
			#self.dataMatrix =self.PCA.get_data((header,))
		else:
			header = self.data.get_headers()[variables.result[0]-len(self.PCA.get_headers())]
			self.dataMatrix = analysis.normalize_columns_separately(self.data, (header,))
		self.dataheaders.append(header)
		
		for index in variables.result[1:]:
			if index < len(self.PCA.get_headers()):
				header = self.PCA.get_headers()[index]
				self.dataMatrix = np.hstack((self.dataMatrix, analysis.normalize_columns_separately(self.PCA, (header,))))
				#self.dataMatrix =np.hstack((self.dataMatrix, self.PCA.get_data((header,))))
			else:
				header = self.data.get_headers()[index-len(self.PCA.get_headers())]
				self.dataMatrix = np.hstack((self.dataMatrix, analysis.normalize_columns_separately(self.data, (header,))))
			print header
			self.dataheaders.append(header)
			
		if len(variables.result) == 2:
			self.dataMatrix = np.hstack((self.dataMatrix, np.zeros(shape=(len(self.dataMatrix),1))))
		
		#self.dataMatrix = self.PCA.get_data(headers)
		homogenous_coordinates = np.ones(shape =(len(self.dataMatrix), 1))
		self.dataMatrix = np.hstack((self.dataMatrix , homogenous_coordinates))
		
		if variables.resultc is not None:
			if variables.resultc < len(self.PCA.get_headers()):
				header = self.PCA.get_headers()[variables.resultc]
				if header[:7] == 'cluster':
					self.colorMatrix = self.PCA.get_data((header,))
				else:
					self.colorMatrix =analysis.normalize_columns_separately(self.PCA, (header,))
				self.colorResult = header
			else:
				header = self.data.get_headers()[variables.resultc-len(self.PCA.get_headers())]
				self.colorMatrix =analysis.normalize_columns_separately(self.data, (header,))
				self.colorResult = header
		else:
			self.colorResult = None
			self.colorMatrix = None
		
		if variables.results is not None:
			if variables.resultc < len(self.PCA.get_headers()):
				header = self.PCA.get_headers()[variables.results]
				self.sizeMatrix =analysis.normalize_columns_separately(self.PCA, (header,))
				self.sizeResult = header
			else:
				header = self.data.get_headers()[variables.results-len(self.PCA.get_headers())]
				self.sizeMatrix =analysis.normalize_columns_separately(self.data, (header,))
				self.sizeResult = header
		else:
			self.sizeResult = None
			self.sizeMatrix = None	
		self.rows = len(self.dataMatrix)
		self.buildAxes()
			
		
	#displays all of the data about the selected PCA analysis
	def showPCATable(self):	
		if self.PCA is None:
			print 'you don\'t have any data'
			return
		infoView = tk.Toplevel(background = "white")
		infoView.title("PCA Analysis")
		
		#width is function of eigen
		
		headers = ['E-vec', 'E-val', 'Energy']
		for header in self.PCA.get_data_headers():
			headers.append(header)
		
		print headers
		
		#twidth = len(headers)*17
		#theight = len(headers)*7
		#woff = self.canvas.winfo_width()-twidth/2 +25
		#hoff = self.canvas.winfo_height()-theight/2+25
		#infoView.geometry("%dx%d%+d%+d" % (twidth, theight, woff, hoff))
			
		for i in range(len(headers)):
			header = tk.Message(infoView, width = 50, text = headers[i], background = "white")
			header.grid(row = 0, column = i, sticky = tk.N+tk.E+tk.W+tk.S)
			
		for i in range(len(self.PCA.get_headers())):
			if i%2 != 0:
				color = "white"
			else:
				color = "#CAE1FF"
			Eheader = tk.Message(infoView, width = 50, text = self.PCA.get_headers()[i], background = color)
			Eheader.grid(row = i+1, column = 0)
			#Im sure there is a much "better" way to do this but this was the way I found to be able to control the precision
			value =  np.array_str(np.asmatrix(self.PCA.get_eigenvalues()[i]), precision = 4)
			Evalue = tk.Message(infoView, width = 50, text = value[2:8], background = color)
			Evalue.grid(row = i+1, column = 1, sticky = tk.N+tk.E+tk.W+tk.S)
			value =  np.array_str(np.asmatrix(self.PCA.get_energies()[i]), precision = 4)
			Energy =tk.Message(infoView, width = 50, text = value[2:8], background = color)
			Energy.grid(row = i+1, column = 2, sticky = tk.N+tk.E+tk.W+tk.S)
			for j in range(len(self.PCA.eigenvectors[0])):
				value = np.array_str(np.asmatrix(self.PCA.eigenvectors[i,j]), precision =5)
				for index in range(len(value)):
					if value[index] == "]":
						break
				message = tk.Message(infoView, width= 50, text = value[2:index], background = color)
				message.grid(row = i+1, column = j+3, sticky = tk.N+tk.E+tk.W+tk.S)
		
		button = tk.Button(infoView, text = "close", command = infoView.destroy)
		if len(headers)%2 == 0:			
			button.grid(column = len(headers)/2-1, columnspan = 2)
		else:
			button.grid(column = len(headers)/2)
		
		
	#plots the data
	def handlePlotData(self, event = None):
		if self.data is None:
			print 'you don\'t have any data'
			return
		headerbox = Dialogs.AxesDialog(self.root, self.data.get_headers())
		colorbox = Dialogs.ColorDialog(self.root, self.data.get_headers())
		#result = headerbox.result + colorbox.result
		#print headerbox.result
		if headerbox.result != []:
			#the point of new data is for when the user tries to plot new data, but cancels out of it, since the headerboxes need the new data, but the canvas needs the old data
			#print headerbox.result
			self.dataheaders = headerbox.result
			temp_matrix = analysis.normalize_columns_separately(self.data, headerbox.result)
			if colorbox.resultc != []:
				if self.colorVar.get() == 1:
					temp_matrix2 = self.data.get_data((colorbox.resultc,))
				else:
					temp_matrix2 =analysis.normalize_columns_separately(self.data, (colorbox.resultc,))
			if colorbox.results != []:
				temp_matrix3 =analysis.normalize_columns_separately(self.data, (colorbox.results,))
			self.rows = len(temp_matrix)
			#print self.rows
			if len(headerbox.result) == 2:
				temp_matrix = np.hstack((temp_matrix, np.zeros(shape=(self.rows,1))))
			homogenous_coordinates = np.ones(shape =(self.rows, 1))
			self.dataMatrix = np.hstack((temp_matrix , homogenous_coordinates))
			if colorbox.resultc != []:
				self.colorMatrix = temp_matrix2
				self.colorResult = colorbox.resultc
			else:
				self.colorResult = None
				self.colorMatrix = None
			
			if colorbox.results != []:
				self.sizeMatrix = temp_matrix3
				self.sizeResult = colorbox.results
			else:
				self.sizeResult = None
				self.sizeMatrix = None			
			self.buildAxes()
			
	#opens a file
	def handleOpen(self, event=None):
		#self.uniqueColors = False
		filename = tkFileDialog.askopenfilename(parent = self.root, title = "Choose a data file", initialdir = '.')
		if not filename.endswith(('.csv', '.xml')):
			print 'thats not a valid file'
			return
		self.data = data.Data(filename)
		if filename not in self.pastSelections:
			self.pastSelections.append(filename)
			#this cuts off the directory of the filename for display purposes
			index = len(filename)-1
			while index>0:
				if filename[index] == '/':
					break
				else:
					index -=1
			filename = filename[index+1:]
			self.pastListbox.insert(tk.END, filename)
					
	#handles the mouse button 1
	def handleMouseButton1(self, event):
		print 'handle mouse button 1: %d %d' % (event.x, event.y)
		self.baseClick = (event.x, event.y)
		
	#handles the mouse button 2 being pressed 
	def handleMouseButton2(self, event):
		self.baseClick2 = (event.x, event.y)
		self.baseView = self.view.clone()
		print 'handle mouse button 2: %d %d' % (event.x, event.y)
		#self.canvas.postscript(file = "imag%d.ps"%(self.n) , colormode = 'color')
		#self.n+=1
		
	#handles button 3
	def handleMouseButton3(self, event):
		self.baseClick = (event.x, event.y)
		self.baseExtent = self.view.extent.copy()		
		
	# This is called if the first mouse button is being moved
	# changes the coordinates of all the objects but the change moved so the canvas moves
	def handleMouseButton1Motion(self, event):
		# calculate the difference
		diff = ( float(event.x - self.baseClick[0]), float(event.y - self.baseClick[1]) )
		delta0 = diff[0]/self.view.screen[0,0]*self.view.extent[0,0]*0.75
		delta1 = diff[1]/self.view.screen[0,1]*self.view.extent[0,1]*0.75
		self.view.vrp += delta0 * self.view.u + delta1* self.view.vup
		self.updateAxes()
		#self.canvas.postscript(file = "imag%d.ps"%(self.n) , colormode = 'color')
		#self.n+=1
		# update base click
		self.baseClick = ( event.x, event.y )
		print 'handle button1 motion %d %d' % (diff[0], diff[1])
			
	
	#rotates the data
	def handleMouseButton2Motion(self, event):
		delta0 = float(event.x-self.baseClick2[0])/(self.view.screen[0,0])*math.pi
		delta1 = float(event.y-self.baseClick2[1])/(self.view.screen[0,1])*math.pi
		self.view = self.baseView.clone()		
		self.view.rotateVRC(-delta0, delta1)
		self.updateAxes()
		#self.canvas.postscript(file = "imag%d.ps"%(self.n) , colormode = 'color')
		#self.n+=1
		
	#scales the data
	def handleMouseButton3Motion(self, event):
		dy = event.y - self.baseClick[1]
		dx = event.x - self.baseClick[0]
		"""
		if self.scaleOption.get() == "X":
			k = 1.0/self.canvas.winfo_height()
			f = 1.0+k*dx
			f = max(min(f, 3.0), 0.1)
			self.view.extent[0,0] = max(min(self.baseExtent[0,0]*f, 3.0), 0.1)
		elif self.scaleOption.get() == "Y":
			k = 1.0/self.canvas.winfo_height()
			f = 1.0+k*dy
			f = max(min(f, 3.0), 0.1)
			self.view.extent[0,1] = max(min(self.baseExtent[0,1]*f, 3.0), 0.1)
		else:"""
		k = 1.0/self.canvas.winfo_height()
		f = 1.0+k*dy
		f = max(min(f, 3.0), 0.1)
		self.view.extent[0,0] = max(min(self.baseExtent[0,0]*f, 3.0), 0.1)
		self.view.extent[0,1] = max(min(self.baseExtent[0,1]*f, 3.0), 0.1)
		self.updateAxes()
	
	def main(self):
		print 'Entering main loop'
		self.root.mainloop()


if __name__ == "__main__":
	dapp = DisplayApp(1200, 675)
	dapp.main()