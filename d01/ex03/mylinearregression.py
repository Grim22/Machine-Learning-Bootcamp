import numpy as np
import sys
sys.path.insert(1, '../ex02')
from gradient import gradient

class MyLinearRegression():
    def __init__(self, theta):
        """
        Description:
            generator of the class, initialize self.
        Args:
            theta: has to be a list or a numpy array, it is a vector of
dimension (number of features + 1, 1).
        Raises:
            This method should noot raise any Exception.
        """
        self.theta = np.array(theta)
    
    def predict_(self, X):
        """
        returns an estimate of the output with X as an input an theta as an hypothesis
        """
        if X.shape[1] + 1 != self.theta.size:
            print("Inc dim")
            return(None)
        a = np.full((X.shape[0], X.shape[1] + 1), 1, float)
        a[:, 1:] = X
        return(np.dot(a, self.theta))

    def cost_elem_(self, X, Y):
        if X.shape[1] + 1 != self.theta.size or X.shape[0] != Y.size:
            print("Inc dim")
            return
        Z = self.predict_(X)
        l = [0.5 / Y.size * (i - j)**2 for i, j in zip(Z, Y)]
        return(np.array(l))

    def cost_(self, X, Y):
        """
        calculates the cost of the model with MSE being the cost function, and theta being the hypothesis
        X: input
        Y: output
        """
        if X.shape[1] + 1 != self.theta.size or X.shape[0] != Y.size:
            print("Inc dim")
            return
        c = self.cost_elem_(X, Y)
        s = 0
        for i in c:
            s = s + i
        return(s)
    
    def mse_(self, X, Y):
         Y_hat = self.predict_(X)
         if Y.size == 0 or Y_hat.size == 0 or Y.size != Y_hat.size:
            return(None)
         s = 0
         for i, j in zip(Y, Y_hat):
            s = s + (j - i)**2
         s = s / Y.size
         return(s)

    def fit_(self, X, Y, alpha, n_cycle):
        """
        operates a gradient descent on theta, with n_cycle being the number of iterations
        """
        X_N = np.full((X.shape[0], X.shape[1] + 1), 1, float)
        X_N[:, 1:] = X
        for i in enumerate(range(n_cycle -1)):
            theta_tmp = self.theta
            for j in range(self.theta.size):
                self.theta[j] = self.theta[j] - alpha * gradient(X_N, Y, theta_tmp)[j]
        return(self.theta)

if __name__ == "__main__":
    MyLR = MyLinearRegression
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89.,144.]])
    Y = np.array([[23.], [48.], [218.]])
    mylr = MyLR([[1.], [1.], [1.], [1.], [1]])
    #print(mylr.predict_(X))
    #array([[8.], [48.], [323.]])
    #print(mylr.cost_elem_(X,Y))
    #array([[37.5], [0.], [1837.5]])
    #print(mylr.cost_(X,Y))
    #1875.0
    mylr.fit_(X, Y, alpha = 1.6e-4, n_cycle=200000)
    print(mylr.theta)
    print(mylr.predict_(X))
    print(mylr.cost_(X,Y))