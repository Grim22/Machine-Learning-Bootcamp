import numpy as np

def vec_mse(y , y_hat):
    if y.size == 0 or y_hat.size == 0 or y.size != y_hat.size:
        return(None)
    s = np.dot(y_hat - y, y_hat - y)
    return(s / y_hat.size)

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])

print(vec_mse(X, Y))