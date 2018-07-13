import numpy as np

def landmark(inp_values,mode=0):
	result = list()
	if(mode):
		x = np.linspace(inp_values['xlower'], inp_values['xupper'])
		y = np.linspace(inp_values['ylower'], inp_values['yupper'])
		xn,yn = np.meshgrid(x,y)
		result.append(xn); result.append(yn)
	else:
		xn = inp_values['x']
		yn = inp_values['y']
	z = np.exp((-1*(xn-4)**2)-(yn-4)**2) + np.exp((-1*(xn+4)**2)-(yn-4)**2) + 2*(np.exp((-1*xn**2)-yn**2) + np.exp((-1*xn**2)-(yn+4)**2))
	result.append(z)
	return result

def matays(inp_values,mode=0):
	result = list()
	if(mode):
		x = np.linspace(inp_values['xlower'], inp_values['xupper'])
		y = np.linspace(inp_values['ylower'], inp_values['yupper'])
		xn,yn = np.meshgrid(x,y)
		result.append(xn); result.append(yn)
	else:
		xn = inp_values['x']
		yn = inp_values['y']
	z =  0.26*(xn**2 + yn**2) - 0.48*xn*yn
	result.append(z)
	return result
