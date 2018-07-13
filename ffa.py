import numpy as np
import cost_functions
import plot_graph

class ffa():
    '''
    Ffirefly algorithm.
    '''
    def __init__(self,rnges,population,alpha=0.2,gamma=1.0):
        self.population = population
        self.alpha = 0.5
        self.gamma = 1.0
        self.rnges = rnges
        self.xn = np.zeros(0)
        self.xy = np.zeros(0)
        self.lightn = np.zeros(0)

    def initiate(self,max_gen):
        x_range = self.rnges['xupper']-self.rnges['xlower']
        y_range = self.rnges['yupper']-self.rnges['ylower']
        self.xn = np.random.rand(self.population)*x_range+self.rnges['xlower']
        self.yn = np.random.rand(self.population)*y_range+self.rnges['ylower']
        self.lightn = np.zeros(self.yn.shape)

    def findrange(self):
        for i in range(self.yn.size):
            if self.xn[i]<=self.rnges['xlower']:
                self.xn[i] = self.rnges['xlower']
            if self.xn[i]>=self.rnges['xupper']:
                self.xn[i] = self.rnges['xupper']
            if self.yn[i]<=self.rnges['ylower']:
                self.yn[i] = self.rnges['ylower']
            if self.yn[i]>=self.rnges['yupper']:
                self.yn[i] = self.rnges['yupper']

    def ffa_move(self,xo,yo,lighto):
        ni = self.yn.shape[0]
        nj = yo.shape[0]
        for i in range(ni):
            for j in range(nj):
                r = np.sqrt((self.xn[i]-xo[j])**2 + (self.yn[i]-yo[j])**2)
                if self.lightn[i]<lighto[j]:
                    beta0 = 1
                    beta = beta0*np.exp(-1*self.gamma*r**2)
                    self.xn[i] = self.xn[i]*(1-beta)+xo[j]*beta+self.alpha*(np.random.rand()-0.5)
                    self.yn[i] = self.yn[i]*(1-beta)+yo[j]*beta+self.alpha*(np.random.rand()-0.5)
        self.findrange()

    def firefly_simple(self,max_gen):
        z = cost_functions.landmark(self.rnges, mode=1)[2]
        self.initiate(max_gen)
        plot_graph.plot_case(self.xn,self.yn)
        for i in range(max_gen):
            values = {'x':self.xn, 'y':self.yn}
            zn = cost_functions.landmark(values, mode=0)[0]
            lighto = np.sort(zn)
            indexes = np.argsort(zn)
            # lighto = lighto[::-1]         # for minima
            # indexes = indexes[::-1]       # for minima
            xo = np.array([self.xn[i] for i in indexes])
            yo = np.array([self.yn[i] for i in indexes])
            self.ffa_move(xo,yo,lighto)
        plot_graph.plot_case(self.xn,self.yn)


if __name__ == '__main__':
    rnges = {'xlower':-5, 'xupper':5, 'ylower':-5, 'yupper':5}
    obj = ffa(rnges=rnges, population=20)
    obj.firefly_simple(200)
