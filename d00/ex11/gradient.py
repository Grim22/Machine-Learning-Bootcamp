import numpy as np

def gradient(x, y, theta):
    if x.shape[0] != y.size or x.shape[1] !=  theta.size:
        return(None)
    if x.size == 0 or y.size == 0 or theta.size == 0:
        return(None)
    l = []
    for j in range(theta.size):
        s = 0
        #print(theta)
        #print(x[0:1, :])
        for i in range(y.size):
            s = s + ((np.dot(x[i : i +1, :], theta) - y[i]) * x[i, j])
        l.append(s / y.size)
    return(np.array(l))

X = np.array([
    [ -6, -7, -9],
        [ 13, -2, 14],
        [ -7, 14, -1],
        [ -8, -4, 6],
        [ -5, -9, 6],
        [ 1, -5, 11],
        [ 9, -11, 8]])
Y = np.array([2, 14, -13, 5, 12, 4, -19])
Z = np.array([3,0.5,-6])
W = np.array([0, 0, 0])
print(gradient(X, Y, W))



    