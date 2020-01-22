import numpy as np

def sum_(x, f):
	if x.size == 0:
		return(None)
	try:
		y = np.array(f(x))
	except:
		return(None)
	s = 0
	for i in y:
		s = float(s) + float(i)
	return(s)

a = np.array([1 , 2 , 3 , 4])
a = np.array([0, 15, -9, 7, 12, 3, -21])
a = np.array([])

print(a)

b = sum_(a, lambda x, y : x + y)

print(b)
	
