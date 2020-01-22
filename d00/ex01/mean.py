import numpy as np

def mean(x):
	if x.size == 0:
		return(None)
	s = 0
	for i in x:
		s = s + i
	return(s / x.size)

X = np.array([0, 15, -9, 7, 12, 3, -21])
print(mean(X))
X = np.array([0, 15, -9, 7, 12, 3, -21])
print(mean(X ** 2))
135.57142857142858
