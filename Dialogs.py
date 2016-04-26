import Tkinter as tk

#class copied in
class Dialog(tk.Toplevel):

	def __init__(self, parent, title = "Distribution Selection"):

		tk.Toplevel.__init__(self, parent)
		self.transient(parent)

		if title:
			self.title(title)

		self.parent = parent

		self.result = []

		body = tk.Frame(self)
		self.initial_focus = self.body(body)
		body.pack(padx=5, pady=5)

		self.buttonbox()

		self.grab_set()

		if not self.initial_focus:
			self.initial_focus = self

		self.protocol("WM_DELETE_WINDOW", self.cancel)

		self.geometry("+%d+%d" % (parent.winfo_rootx()+50,
								  parent.winfo_rooty()+50))

		self.initial_focus.focus_set()

		self.wait_window(self)

	#
	# construction hooks

	def body(self, master):
		# create dialog body.  return widget that should have
		# initial focus.  this method should be overridden

		pass

	def buttonbox(self):
		# add standard button box. override if you don't want the
		# standard buttons

		box = tk.Frame(self)

		w = tk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
		w.pack(side=tk.LEFT, padx=5, pady=5)
		w = tk.Button(box, text="Cancel", width=10, command=self.cancel)
		w.pack(side=tk.LEFT, padx=5, pady=5)

		self.bind("<Return>", self.ok)
		self.bind("<Escape>", self.cancel)

		box.pack()

	#
	# standard button semantics

	def ok(self, event=None):

		if not self.validate():
			self.initial_focus.focus_set() # put focus back
			return

		self.withdraw()
		self.update_idletasks()

		self.apply()

		self.cancel()

	def cancel(self, event=None):

		# put focus back to the parent window
		self.parent.focus_set()
		self.destroy()

	#
	# command hooks

	def validate(self):

		return 1 # override

	def apply(self):

		pass # override
	
class AxesDialog(Dialog):
	
	def __init__(self, parent, headers):
		self.headers =headers
		Dialog.__init__(self, parent)
	
	def body(self, master):
		framex = tk.Frame(master)
		scrollbarx = tk.Scrollbar(framex, orient=tk.VERTICAL)
		label = tk.Label(master, text = "Select 3 different Axes for X, Y, and Z")
		labelx = tk.Label(master, text = "Select X axis")
		self.listboxX = tk.Listbox(framex, selectmode = tk.SINGLE, exportselection = 0, yscrollcommand = scrollbarx.set)
		scrollbarx.config(command = self.listboxX.yview)
		scrollbarx.pack(side = tk.RIGHT, fill = tk.Y)
		for header in self.headers:
			self.listboxX.insert(tk.END, header)
		self.listboxX.pack()
		
		framey = tk.Frame(master)
		labely = tk.Label(master, text = "Select Y axis")
		scrollbary = tk.Scrollbar(framey, orient=tk.VERTICAL)
		self.listboxY = tk.Listbox(framey, selectmode = tk.SINGLE, exportselection = 0, yscrollcommand = scrollbary.set)
		scrollbary.config(command = self.listboxY.yview)
		scrollbary.pack(side = tk.RIGHT, fill = tk.Y)
		for header in self.headers:
			self.listboxY.insert(tk.END, header)
		self.listboxY.pack()
			
		framez = tk.Frame(master)
		labelz = tk.Label(master, text = "Select Z axis")		
		scrollbarz = tk.Scrollbar(framez, orient=tk.VERTICAL)
		self.listboxZ = tk.Listbox(framez, selectmode = tk.SINGLE, exportselection = 0, yscrollcommand = scrollbarz.set)
		scrollbarz.config(command = self.listboxZ.yview)
		scrollbarz.pack(side = tk.RIGHT, fill = tk.Y)
		for header in self.headers:
			self.listboxZ.insert(tk.END, header)
		self.listboxZ.pack()
		
		self.listboxX.selection_set(0)
		self.listboxY.selection_set(1)
		""" so they can only select 2 if they REALLY want to
		if len(self.headers) >= 3:
			self.listboxZ.selection_set(2)"""

		label.pack()
		labelx.pack()
		framex.pack()
		labely.pack()
		framey.pack()
		labelz.pack()
		framez.pack()
		
	def apply(self):
		self.result = []
		self.result.append(self.headers[self.listboxX.curselection()[0]])
		self.result.append(self.headers[self.listboxY.curselection()[0]])
		if self.listboxZ.curselection() != ():
			self.result.append(self.headers[self.listboxZ.curselection()[0]])
		
		
	def validate(self):
		result = []
		result.append(self.headers[self.listboxX.curselection()[0]])
		result.append(self.headers[self.listboxY.curselection()[0]])
		if self.listboxZ.curselection() != ():
			#print self.listboxZ.curselection()
			result.append(self.headers[self.listboxZ.curselection()[0]])

		for i in range(len(result)):
			copy = result[:]
			value = result[i]
			copy.pop(i)
			if value in copy:
				return 0
		return 1
		
#listbox for creating a linear regression, it was origionally more distinct from the listbox to select data in general, but then I added a third dimension
class LinRegressDialog(Dialog):
	def __init__(self, parent, headers):
		self.headers =headers
		Dialog.__init__(self, parent, 'Create Linear Regression')
	
	def body(self, master):
		label = tk.Label(master, text = "Pick and independent and dependent variable")
		
		labelInd = tk.Label(master, text = "Select your independent variable")
		frameInd= tk.Frame(master)
		scrollbarInd = tk.Scrollbar(frameInd, orient = tk.VERTICAL)
		self.listboxInd = tk.Listbox(frameInd, selectmode = tk.SINGLE, exportselection = 0, yscrollcommand = scrollbarInd.set)
		scrollbarInd.config(command = self.listboxInd.yview)
		scrollbarInd.pack(side = tk.RIGHT, fill = tk.Y)
		for header in self.headers:
			self.listboxInd.insert(tk.END, header)
		self.listboxInd.pack()
		
		labelInd2 = tk.Label(master, text = "Select your second independent variable (optional)")
		frameInd2= tk.Frame(master)
		scrollbarInd2 = tk.Scrollbar(frameInd2, orient = tk.VERTICAL)
		self.listboxInd2 = tk.Listbox(frameInd2, selectmode = tk.SINGLE, exportselection = 0, yscrollcommand = scrollbarInd2.set)
		scrollbarInd2.config(command = self.listboxInd2.yview)
		scrollbarInd2.pack(side = tk.RIGHT, fill = tk.Y)
		for header in self.headers:
			self.listboxInd2.insert(tk.END, header)
		self.listboxInd2.pack()
		
		labelDep = tk.Label(master, text = "Select your dependent variable")
		frameDep= tk.Frame(master)
		scrollbarDep = tk.Scrollbar(frameDep, orient = tk.VERTICAL)
		self.listboxDep = tk.Listbox(frameDep, selectmode = tk.SINGLE, exportselection = 0, yscrollcommand = scrollbarDep.set)
		scrollbarDep.config(command = self.listboxDep.yview)
		scrollbarDep.pack(side = tk.RIGHT, fill = tk.Y)
		for header in self.headers:
			self.listboxDep.insert(tk.END, header)
		self.listboxDep.pack()
		
		label.pack()
		labelInd.pack()
		frameInd.pack()
		labelInd2.pack()
		frameInd2.pack()
		labelDep.pack()
		frameDep.pack()
	
	def apply(self):
		self.result = []
		self.result.append(self.headers[self.listboxInd.curselection()[0]])
		if self.listboxInd2.curselection() != ():
			self.result.append(self.headers[self.listboxInd2.curselection()[0]])
		self.result.append(self.headers[self.listboxDep.curselection()[0]])
				
	def validate(self):
		if self.listboxInd.curselection() == () or self.listboxDep.curselection() == (): 
			return 0
		if self.listboxInd2.curselection() == ():
			if self.listboxInd.curselection() == () or self.listboxDep.curselection() == () or self.listboxInd.curselection()[0] == self.listboxDep.curselection()[0]:
				return 0
			return 1
		else:
			result = []
			result.append(self.headers[self.listboxInd.curselection()[0]])
			result.append(self.headers[self.listboxInd2.curselection()[0]])
			result.append(self.headers[self.listboxDep.curselection()[0]])
			
			for i in range(len(result)):
				copy = result[:]
				value = result[i]
				copy.pop(i)
				if value in copy:
					return 0
			return 1
		
class ColorDialog(Dialog):
	def __init__(self, parent, headers):
		self.headers =headers
		Dialog.__init__(self, parent, 'color selection')
		
	def body(self, master):
		label = tk.Label(master, text = "what you want to scale by color and size")
		#print self.headers
		labelc = tk.Label(master, text = "Select what to scale by color")
		framec = tk.Frame(master)
		scrollbarc = tk.Scrollbar(framec, orient=tk.VERTICAL)
		self.listboxC = tk.Listbox(framec, selectmode = tk.SINGLE, exportselection = 0, yscrollcommand = scrollbarc.set)
		scrollbarc.config(command = self.listboxC.yview)
		scrollbarc.pack(side = tk.RIGHT, fill = tk.Y)
		for header in self.headers:
			self.listboxC.insert(tk.END, header)
		self.listboxC.pack()
		
		frames = tk.Frame(master)
		scrollbars = tk.Scrollbar(frames, orient=tk.VERTICAL)		
		labels = tk.Label(master, text = "Select what to scale by size")
		self.listboxS = tk.Listbox(frames, selectmode = tk.SINGLE, exportselection = 0, yscrollcommand = scrollbars.set)
		scrollbars.config(command = self.listboxS.yview)
		scrollbars.pack(side = tk.RIGHT, fill = tk.Y)
		for header in self.headers:
			self.listboxS.insert(tk.END, header)
		self.listboxS.pack()
		
		
		label.pack()
		labelc.pack()
		framec.pack()
		labels.pack()
		frames.pack()
		
	def apply(self):
		self.resultc = []
		self.results = []
		try:
			self.resultc =self.headers[self.listboxC.curselection()[0]]
		except:
			pass
		try:
			self.results =self.headers[self.listboxS.curselection()[0]]
		except:
			pass
		
		
	def validate(self):
		return 1
		
class PCADialog(Dialog):
	def __init__(self, parent, headers):
		self.headers =headers
		Dialog.__init__(self, parent, 'PCA selection')
		self.all = False
		
	def body(self, master):
		label =tk.Label(master, text = "pick your PCA value")
		label.pack()
		frame =tk.Frame(master)
		scrollbar = tk.Scrollbar(frame, orient = tk.VERTICAL)	
		self.listbox = tk.Listbox(frame, selectmode = tk.MULTIPLE, exportselection = 0, yscrollcommand = scrollbar.set)
		scrollbar.config(command = self.listbox.yview)		
		scrollbar.pack(side = tk.RIGHT, fill = tk.Y)
		for header in self.headers:
			self.listbox.insert(tk.END, header)
		self.listbox.pack()				
		
		frame.pack()
		
		var = tk.IntVar()
		checkboxA = tk.Checkbutton(master, text = "Select All", variable = var)
		self.var = var
		
		checkboxA.pack()
		
		var2 = tk.IntVar()
		checkboxN = tk.Checkbutton(master, text="Normalize", variable=var2)
		self.var2 = var2
		
		checkboxN.pack()
		
	def apply(self):
		self.result = []
		for i in self.listbox.curselection():
			self.result.append(self.headers[i])
		if self.var.get() == 1:
			self.all = True
		else:
			self.all = False
		if self.var2.get() == 1:
			self.normalize = True	
		else:
			self.normalize = False
			
	def validate(self):
		if len(self.listbox.curselection())>1 or self.var.get() == 1:
			return 1
		else:
			return 0


			
class selectPCAData(Dialog):
	def __init__(self, parent, headers):
		self.headers =headers
		Dialog.__init__(self, parent, 'Select Data')
		
	def body(self, master):
		labelx =tk.Label(master, text = "pick your x value")
		#grid 0, 1
		framex =tk.Frame(master)
		#grid 0, 0
		scrollbarx = tk.Scrollbar(framex, orient = tk.VERTICAL)		
		self.listboxX = tk.Listbox(framex, selectmode = tk.SINGLE, exportselection = 0, yscrollcommand = scrollbarx.set, height = 5)
		scrollbarx.config(command = self.listboxX.yview)
		for header in self.headers:
			self.listboxX.insert(tk.END, header)
		self.listboxX.grid(row = 0, column = 0)
		scrollbarx.grid(row=0, column=1, sticky=tk.N+tk.S)
		
		labely =tk.Label(master, text = "pick your y value")
		framey =tk.Frame(master)
		scrollbary = tk.Scrollbar(framey, orient = tk.VERTICAL)		
		self.listboxY = tk.Listbox(framey, selectmode = tk.SINGLE, exportselection = 0, yscrollcommand = scrollbary.set, height = 5)
		scrollbary.config(command = self.listboxY.yview)
		for header in self.headers:
			self.listboxY.insert(tk.END, header)
		self.listboxY.grid(row = 0, column = 0)
		scrollbary.grid(row=0, column=1, sticky=tk.N+tk.S)
		
		labelz =tk.Label(master, text = "pick your z value")
		framez =tk.Frame(master)
		scrollbarz = tk.Scrollbar(framez, orient = tk.VERTICAL)		
		self.listboxZ = tk.Listbox(framez, selectmode = tk.SINGLE, exportselection = 0, yscrollcommand = scrollbarz.set, height = 5)
		scrollbarz.config(command = self.listboxZ.yview)
		for header in self.headers:
			self.listboxZ.insert(tk.END, header)		
		self.listboxZ.grid(row = 0, column = 0)
		scrollbarz.grid(row=0, column=1, sticky=tk.N+tk.S)
		
		labelc =tk.Label(master, text = "pick what to scale by color")
		framec =tk.Frame(master)
		scrollbarc = tk.Scrollbar(framec, orient = tk.VERTICAL)		
		self.listboxC = tk.Listbox(framec, selectmode = tk.SINGLE, exportselection = 0, yscrollcommand = scrollbarc.set, height = 5)
		scrollbarc.config(command = self.listboxC.yview)
		for header in self.headers:
			self.listboxC.insert(tk.END, header)
		self.listboxC.grid(row = 0, column = 0)
		scrollbarc.grid(row=0, column=1, sticky=tk.N+tk.S)
		
		labels =tk.Label(master, text = "pick what to scale by size")
		frames =tk.Frame(master)
		scrollbars = tk.Scrollbar(frames, orient = tk.VERTICAL)		
		self.listboxS = tk.Listbox(frames, selectmode = tk.SINGLE, exportselection = 0, yscrollcommand = scrollbars.set, height = 5)
		scrollbars.config(command = self.listboxS.yview)
		for header in self.headers:
			self.listboxS.insert(tk.END, header)
		self.listboxS.grid(row = 0, column = 0)
		scrollbars.grid(row=0, column=1, sticky=tk.N+tk.S)
		
		framex.grid(row=0, column =0)
		labelx.grid(row = 0, column =1)
		framey.grid(row=1, column =0)
		labely.grid(row = 1, column =1)				
		framez.grid(row=2, column =0)
		labelz.grid(row = 2, column =1)
		framec.grid(row=3, column =0)
		labelc.grid(row = 3, column =1)
		frames.grid(row=4, column =0)
		labels.grid(row = 4, column =1)
		
	def apply(self):
		#this listbox just gives me the numbers so I can figure out which matrix to take from
		self.result = []
		self.result.append(self.listboxX.curselection()[0])
		self.result.append(self.listboxY.curselection()[0])
		if 	self.listboxZ.curselection() != ():
			self.result.append(self.listboxZ.curselection()[0])
			
		self.resultc = None
		if 	self.listboxC.curselection() != ():
			self.resultc = self.listboxC.curselection()[0]
		
		self.results = None
		if 	self.listboxS.curselection() != ():
			self.results = self.listboxS.curselection()[0]
		
	def validate(self):
		if self.listboxX.curselection() == () or self.listboxY.curselection() == ():
			return 0
		if self.listboxZ.curselection() == ():
			if self.listboxX.curselection() != self.listboxZ.curselection():
				return 1
			else:
				return 0
		else:
			result = []
			result.append(self.headers[self.listboxX.curselection()[0]])
			result.append(self.headers[self.listboxY.curselection()[0]])
			result.append(self.headers[self.listboxZ.curselection()[0]])
			
			for i in range(len(result)):
				copy = result[:]
				value = result[i]
				copy.pop(i)
				if value in copy:
					return 0
			return 1
			
			
class ClusterDialog(Dialog):
	def __init__(self, parent, headers):
		self.headers =headers
		Dialog.__init__(self, parent, 'Cluster selection')
		
	def body(self, master):
		label =tk.Label(master, text = "pick your Headers to cluster")
		label.pack()
		frame =tk.Frame(master)
		scrollbar = tk.Scrollbar(frame, orient = tk.VERTICAL)	
		self.listbox = tk.Listbox(frame, selectmode = tk.MULTIPLE, exportselection = 0, yscrollcommand = scrollbar.set)
		scrollbar.config(command = self.listbox.yview)		
		scrollbar.pack(side = tk.RIGHT, fill = tk.Y)
		for header in self.headers:
			self.listbox.insert(tk.END, header)
		self.listbox.pack()				
		
		frame.pack()
		
		self.slider = tk.Scale(master, from_= 0, to= 15, sliderlength = 15, label = "Num Clusters",orient=tk.HORIZONTAL)
		self.slider.pack()
		"""
		var = tk.IntVar()
		checkbox = tk.Checkbutton(master, text = "Use Distinct Colors", variable = var)
		self.var = var
		checkbox.pack()"""
		
	def apply(self):
		self.result = []
		for i in self.listbox.curselection():
			self.result.append(self.headers[i])
		self.numclusters = self.slider.get()
	"""
		if self.var.get() == 1:
			self.distColors = True
		else:
			self.distColors = False
		"""	
	def validate(self):
		if len(self.listbox.curselection())>0:
			return 1
		else:
			return 0
			

class WriteDialog(Dialog):
	def __init__(self, parent):
		Dialog.__init__(self, parent, 'Write Data')
		self.all = False
		
	def body(self, master):
		label =tk.Label(master, text = "Give your file a name")
		label.pack()
		
		self.textbox = tk.Entry(master)
		self.textbox.pack()
		
		var = tk.IntVar()
		checkboxA = tk.Checkbutton(master, text = "Write out PCA instead", variable = var)
		self.var = var
		
		checkboxA.pack()
		
		
	def apply(self):
		self.result = self.textbox.get()
		if self.var.get() == 1:
			self.pca = True
		else:
			self.pca = False
		
			
	def validate(self):
		if self.textbox.get() == '':
			return 0
		else:
			return 1
