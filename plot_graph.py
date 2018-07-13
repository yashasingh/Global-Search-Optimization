import numpy as np
import matplotlib.pyplot as plt
import cost_functions
from mpl_toolkits.mplot3d import Axes3D

rnges = {'xlower':-5, 'xupper':5, 'ylower':-5, 'yupper':5}
values = cost_functions.landmark(rnges, mode=1)	# cost_functions.matays(rnges, mode=1)

def graph_3d():
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_surface(values[0],values[1],values[2])
	plt.show()

def contour():
	plt.contour(values[0],values[1],values[2])
	plt.show()

def plot_case(xn,yn):
	plt.contour(values[0],values[1],values[2])
	plt.scatter(xn,yn)
	plt.show()

if __name__ == '__main__':
	print("Choose the type 0/1: ",end=' ')
	n = int(input())
	if n:
		print(n,"here!")
		graph_3d()
	else:
		contour()
