import numpy as np

def variance(x):
	if x.size == 0:
		return(None)
	m = 0
	for i in x:
		m = m + i
	m = m / x.size
	v = 0
	for i in x:
		v = v + (i - m)**2
	v = (v / x.size)**0.5

X = np.array([0, 15, -9, 7, 12, 3, -21])

print(np.var(X / 2))
		
	
