import numpy as np
import cost_functions
import plot_graph
from model import Model
from dataset_util import read_data_sets

class ffa(object):
    '''
    Ffirefly algorithm.
    '''
    def __init__(self,rnges,input_nodes,logits,population,alpha=0.2,gamma=1.0):
        self.population = population
        self.alpha = 0.5
        self.gamma = 1.0
        self.rnges = rnges
        self.xn = np.zeros(0)
        self.yn = np.zeros(0)
        self.zn = list()
        self.fireflies = list()
        self.input_nodes = [input_nodes]
        self.logits = logits
        self.lightn = np.zeros(0)
        self.data = read_data_sets("MNIST_data/", one_hot=True)

    def build_nodes(self):
        print("Building Nodes")
        for i in self.yn:
            self.zn.append(np.random.randint(low=self.rnges['node_count_lower'],high=self.rnges['node_count_upper'],size=self.rnges['h_layers_count_upper']))

    def collect_fireflies(self):
        print("Collecting fireflies")
        self.fireflies = list()
        for i in range(len(self.xn)):
            self.fireflies.append(Model(self.xn[i], self.yn[i], self.input_nodes[0], self.logits, self.zn[i], self.data))
    
    def initiate(self,max_gen):
        print("Called ffa initiate")
        learning_rate_range = self.rnges['learning_rate_upper']-self.rnges['learning_rate_lower']
        self.xn = np.random.rand(self.population)*learning_rate_range+self.rnges['learning_rate_lower']
        self.yn = np.random.randint(low=self.rnges['h_layers_count_lower']+2, high=self.rnges['h_layers_count_upper'],size=self.population)
        self.build_nodes()
        self.lightn = np.zeros(self.yn.shape)
        self.collect_fireflies()

    def findrange(self):
        for i in range(self.yn.size):
            if self.xn[i]<=self.rnges['learning_rate_lower']:
                self.xn[i] = self.rnges['learning_rate_lower']
            if self.xn[i]>=self.rnges['learning_rate_upper']:
                self.xn[i] = self.rnges['learning_rate_upper']
            if self.yn[i]<=self.rnges['h_layers_count_lower']:
                self.yn[i] = self.rnges['h_layers_count_lower']
            if self.yn[i]>=self.rnges['h_layers_count_upper']:
                self.yn[i] = self.rnges['h_layers_count_upper']
            for i in range(len(self.zn)):
                for j in range(len(self.zn[i])):
                    if self.zn[i][j]>=self.rnges['node_count_upper']:
                        self.zn[i][j] = self.rnges['node_count_upper']
                    if self.zn[i][j]<=self.rnges['node_count_lower']:
                        self.zn[i][j] = self.rnges['node_count_lower']

    def ffa_move(self,xo,yo,zo,lighto):
        ni = self.yn.shape[0]
        nj = yo.shape[0]
        temp = 0
        for i in range(ni):
            for j in range(nj):
                original_layers_count = self.yn[i]-2
                r1 = np.sqrt((self.xn[i]-xo[j])**2 + (self.yn[i]-yo[j])**2)
                if self.yn[i]<yo[j]:
                    for k in range(self.yn[i]):
                        temp+=(self.zn[i][k]-zo[j][k])**2
                else:
                    for k in range(yo[i]):
                        temp+=(self.zn[i][k]-zo[j][k])**2
                r2 = np.sqrt(temp)
                if self.lightn[i]<lighto[j]:
                    beta0 = 1
                    beta1 = beta0*np.exp(-1*self.gamma*r1**2)
                    beta2 = beta0*np.exp(-1*self.gamma*r2**2)
                    self.xn[i] = self.xn[i]*(1-beta1)+xo[j]*beta1+self.alpha*(np.random.rand()-0.5)
                    self.yn[i] = self.yn[i]*(1-beta1)+yo[j]*beta1+self.alpha*(np.random.randint(low=self.rnges['h_layers_count_lower'], high=self.rnges['h_layers_count_upper'])-10)
                    for k in range(self.yn[i]):
                        if k < original_layers_count:
                            if k<yo[j]:
                                self.zn[i][k] = self.zn[i][k]*(1-beta2)+zo[j][k]*beta2+self.alpha*(np.random.randint(low=self.rnges['node_count_lower'], high=self.rnges['node_count_upper'])-0.5)
                        elif k < yo[j]:
                            self.zn[i][k] = zo[j][k]
        print("INSIDE MOVE!")
        print(self.xn,self.yn)
        self.findrange()
        self.collect_fireflies()
        print("again insode MOVWE")
        print(self.xn,self.yn)
        print(self.fireflies)

    def firefly_simple(self,max_gen):
        self.initiate(max_gen)
        print("Inside ffa_simple")
        for i in range(max_gen):
            print(self.xn)
            print(self.yn)
            print(self.zn)
            qn = []
            for j in range(len(self.xn)):
                qn.append(self.fireflies[j].make_layer())
            lighto = np.sort(qn)
            indexes = np.argsort(qn)
            # lighto = lighto[::-1]         # for minima
            # indexes = indexes[::-1]       # for minima
            print("\n At step "+str(i)+"with values- ", qn)
            xo = np.array([self.xn[j] for j in indexes])
            yo = np.array([self.yn[j] for j in indexes])
            zo = list(np.array([self.zn[j] for j in indexes]))
            print(self.xn,xo)
            print(self.yn,yo)
            print(self.zn,zo)
            # print(type(zo),type(self.zn))
            self.ffa_move(xo,yo,zo,lighto)
            print("\n\n Moved "+str(i))


if __name__ == '__main__':
    rnges = {'learning_rate_lower':0.0001, 'learning_rate_upper':0.1, 'h_layers_count_upper':20, 'h_layers_count_lower':3, 'node_count_lower':10, 'node_count_upper':150}
    # rnges = {'xlower':-5, 'xupper':5, 'ylower':-5, 'yupper':5}
    obj = ffa(rnges=rnges, input_nodes=784, logits=10, population=5)
    obj.firefly_simple(max_gen=20)
