import numpy as np
import cost_functions

class ffa(object):
	'''
	Ffa
	'''
	def __init__(slef,population=50,alpha=0.2,gamma=1.0):
		self.population = population
		self.alpha = alpha
		self.gamma = gamma
		self.xn = np.array()
		self.xy = np.array()
		self.lightn = np.array()

	def initiate(rnges, self.population):
		x_range = rnges['xupper']-rnges['xlower']
		y_range = rnges['yupper']-rnges['ylower']
		self.xn = np.random.rand(1,self.population)*x_range+rnges['xlower']
		self.yn = np.random.rand(1,self.population)*y_range+rnges['ylower']
		self.lightn = np.zeros(self.yn.size)
		return np.array([self.xn, self.yn, self.lightn])

	def findrange(self.xn,self.yn,rnges):
		for i in range(self.yn.shape):
			if self.xn[i]<=rnges['xlower']:
				self.xn = rnges['xlower']
			if self.xn[i]>=rnges['xupper']:
				self.xn = rnges['xupper']
			if self.yn[i]<=rnges['ylower']:
				self.yn = rnges['ylower']
			if self.yn[i]>=rnges['yupper']:
				self.yn = rnges['yupper']

	def ffa_move(self.xn,self.yn,self.lightn,xo,yo,lighto):
		ni = self.yn.shape
		nj = yo.shape
		for i in range(ni):
			for j in range(nj):
				r = sqrt((self.xn[i]-xo[i])**2 + (self.yn[i]-yo[i])**2)
				if self.lightn[i]<lighto[i]:
					beta0 = 1
				beta = beta0*np.exp(-1*self.gamma*r**2)
				self.xn[i] = self.xn[i]*(1-beta)+xo[i]*beta+self.alpha*(np.random.rand()-0.5)
				self.yn[i] = self.yn[i]*(1-beta)+yo[j]*beta+self.alpha*(np.random.rand()-0.5)
		self.xn,self.yn = findrange(self.xn,self.yn,rnges)
		return np.array([self.xn,self.yn])

