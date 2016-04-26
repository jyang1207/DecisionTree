#File: View.py
#Author: Jason Gurevitch

import numpy as np
import math

class View:
	
	def __init__(self):
		self.reset()
	
	#resets the fields
	def reset(self):
		self.vrp = np.matrix([0.5, 0.5, 1])
		self.vpn = np.matrix([0.0, 0.0, -1.0])
		self.vup = np.matrix([0.0, 1.0, 0.0])
		self.u = np.matrix([-1.0,0.0,0.0])
		self.extent = np.matrix([1.0,1.0,1.0])
		self.screen = np.matrix([400, 400])
		self.offset = np.matrix([20, 20])
	
	#creates a matrix to return based on the fields (3D pipleine)
	def build(self):
		vtm = np.identity(4, float)
		t1 = np.matrix( [[1,0,0, -self.vrp[0,0]],
						[0,1,0, -self.vrp[0,1]],
						[0,0,1, -self.vrp[0,2]],
						[0,0,0, 1]])
		vtm = t1 *vtm
		#print vtm
		tu = np.cross(self.vup, self.vpn)
		tvup = np.cross(self.vpn, tu)
		tvpn = self.vpn
		tu = self.normalize(tu)
		tvup = self.normalize(tvup)
		tvpn = self.normalize(tvpn)
		self.u = tu
		self.vup = tvup
		self.vpn = tvpn
		
		#align the axes
		r1 = np.matrix([[tu[0,0], tu[0,1], tu[0,2], 0.0],
						[tvup[0,0], tvup[0,1], tvup[0,2], 0.0],
						[tvpn[0,0], tvpn[0,1], tvpn[0,2], 0.0],
						[0.0, 0.0, 0.0, 1.0]])
		vtm = r1*vtm
		#print vtm
		
		t1 = np.identity(4, float)
		t1[0,3] = 0.5*self.extent[0,0]
		t1[1,3] = 0.5*self.extent[0,1]
		vtm = t1*vtm
		#print vtm
		
		s1 = np.identity(4, float)
		s1[0,0] = -self.screen[0,0]/self.extent[0,0]
		s1[1,1] = -self.screen[0,1]/self.extent[0,1]
		s1[2,2] = 1.0/self.extent[0,2]
		vtm = s1*vtm
		#print vtm

		t2 = np.identity(4, float)
		t2[0,3] = self.screen[0,0]+self.offset[0,0]
		t2[1,3] = self.screen[0,1]+self.offset[0,1]
		vtm = t2*vtm
		return vtm
		
	#normalizes the given vector
	def normalize(self, vector):
		length = math.sqrt(vector[0,0]**2+ vector[0,1]**2+ vector[0,2]**2)
		vector[0,0] = vector[0,0]/length
		vector[0,1] = vector[0,1]/length
		vector[0,2] = vector[0,2]/length
		return vector
		
	def rotateVRC(self,angleVUP, angleU):
		t1 = np.identity(4, float)
		for i in range(3):
			t1[i,3] = -(self.vrp[0,i]+self.vpn[0,i] *self.extent[0,i]*0.5)
		Rxyz  = np.matrix([[self.u[0,0], self.u[0,1], self.u[0,2], 0.0],
						[self.vup[0,0], self.vup[0,1], self.vup[0,2], 0.0],
						[self.vpn[0,0], self.vpn[0,1], self.vpn[0,2], 0.0],
						[0.0, 0.0, 0.0, 1.0]])
		r1 = np.matrix([[math.cos(angleVUP),0,math.sin(angleVUP),0],
						[0,1,0,0],
						[-math.sin(angleVUP),0,math.cos(angleVUP),0],
						[0,0,0,1]])
		r2 = np.matrix([[1,0,0,0],
						[0, math.cos(angleU), -math.sin(angleU),0],
						[0, math.sin(angleU), math.cos(angleU),0],
						[0,0,0,1]])
		t2 = np.identity(4, float)
		for i in range(3):
			t2[i,3] = self.vrp[0,i]+self.vpn[0,i] *self.extent[0,i]*0.5
		tvrc = np.matrix([[self.vrp[0,0],self.vrp[0,1],self.vrp[0,2],1],
						[self.u[0,0],self.u[0,1],self.u[0,2],0],
						[self.vup[0,0],self.vup[0,1],self.vup[0,2],0],
						[self.vpn[0,0],self.vpn[0,1],self.vpn[0,2],0]])
		tvrc = (t2*Rxyz.T*r2*r1*Rxyz*t1*tvrc.T).T
		self.vrp=tvrc[0,:3]
		self.u = self.normalize(tvrc[1,:3])
		self.vup = self.normalize(tvrc[2,:3])
		self.vpn = self.normalize(tvrc[3,:3])

		
	#returns a deep clone of the object
	def clone(self):
		result = View()
		result.vrp = self.vrp.copy()
		result.vpn = self.vpn.copy()
		result.vup = self.vup.copy()
		result.u = self.u.copy()
		result.extent = self.extent.copy()
		result.screen = self.screen.copy()
		result.offset = self.offset.copy()
		return result
		
if __name__ == "__main__":
	test = View()
	print test.build()