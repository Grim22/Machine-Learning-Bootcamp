import numpy as np
from NumPyCreator import NumPyCreator
from dot import dot

def mat_mat_product(x, y):
	if x.shape[1] != y.shape[0] or x.size == 0 or y.size == 0:
		return (None)
	l = []
	for i in range (x.shape[0]):
		l.append([dot(x[i, :],y[:, j]) for j in range (y.shape[1])])
	return(np.array(l))

W = np.array([
[ -8, 8, -6, 14, 14, -9, -4],
[ 2, -11, -2, -11, 14, -2, 14],
[-13, -2, -5, 3, -8, -4, 13],
[ 2, 13, -14, -15, -14, -15, 13],
[ 2, -1, 12, 3, -7, -3, -6]])

Z = np.array([
[ -6, -1, -8, 7, -8],
[ 7, 4, 0, -10, -10],
[ 7, -13, 2, 2, -11],
[ 3, 14, 7, 7, -4],
[ -1, -3, -8, -4, -14],
[ 9, -14, 9, 12, -7],
[ -9, -4, -10, -3, 6]])

#print (W.shape[0])

print(mat_mat_product(W, Z))
print(W.dot(Z))
print(mat_mat_product(Z, W))
