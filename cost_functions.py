import numpy as np

def landmark(inp_values,mode=0):
	if(mode):
		x = np.linspace(inp_values['xlower'], inp_values['xupper'])
		y = np.linspace(inp_values['ylower'], inp_values['yupper'])
		xn,yn = np.meshgrid(x,y)
	else:
		xn = inp_values['x']
		yn = inp_values['y']
	
	z = np.exp((-1*(xn-4)**2)-(yn-4)**2) + np.exp((-1*(xn+4)**2)-(yn-4)**2) + 2*(np.exp((-1*xn**2)-yn**2) + np.exp((-1*xn**2)-(yn+4)**2))
	return z
