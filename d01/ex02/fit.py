import numpy as np
from gradient import gradient
from pred import predict_

def fit_(theta, X, Y, alpha, n_cycle):
    """
    operates a gradient descent on theta, with n_cycle being the number of iterations
    """
    if X.shape[1] + 1 != theta.size or X.shape[0] != Y.size:
        return(None)
    X_N = np.full((X.shape[0], X.shape[1] + 1), 1, float)
    X_N[:, 1:] = X
    for i in enumerate(range(n_cycle -1)):
        theta_tmp = theta
        for j in range(theta.size):
            theta[j] = theta[j] - alpha * gradient(X_N, Y, theta_tmp)[j]
    return(theta)

if __name__ == "__main__":
    X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    Y1 = np.array([[2.], [6.], [10.], [14.], [18.]])
    theta1 = np.array([[1.], [1.]])
    theta1 = fit_(theta1, X1, Y1, alpha = 0.01, n_cycle=2000)
    print(theta1)
    print(predict_(theta1, X1))

    X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8.,80.]])
    Y2 = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta2 = np.array([[42.], [1.], [1.], [1.]])
    theta2 = fit_(theta2, X2, Y2, alpha = 0.0005, n_cycle=42000)
    print(theta2)
    print(predict_(theta2, X2))