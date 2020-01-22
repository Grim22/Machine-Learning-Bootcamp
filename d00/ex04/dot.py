import numpy as np

def dot(x, y):
	if x.size == 0 or y.size == 0 or x.size != y.size:
		return(None)
	p = 0
	for i, j in zip(x, y):
		p = p + i * j
	return(p)

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])

print(dot(Y, Y))
