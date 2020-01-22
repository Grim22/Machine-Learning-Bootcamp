import numpy as np

def mse(y, y_hat):
    if y.size == 0 or y_hat.size == 0 or y.size != y_hat.size:
        return(None)
    s = 0
    for i, j in zip(y, y_hat):
        s = s + (j - i)**2
    s = s / y.size
    return(s)

X = np.array([0, 15, -9, 7, 12, 3, -21])
Y = np.array([2, 14, -13, 5, 12, 4, -19])

print(mse(X, Y))