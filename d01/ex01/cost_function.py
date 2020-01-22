import numpy as np
from pred import predict_

def cost_elem_(theta, X, Y):
    if X.shape[1] + 1 != theta.size or X.shape[0] != Y.size:
        print("Inc dim")
        return
    Z = predict_(theta, X)
    l = [0.5 / Y.size * (i - j)**2 for i, j in zip(Z, Y)]
    return(np.array(l))

def cost_(theta, X, Y):
    """
    calculates the cost of the model with MSE being the cost function, and theta being the hypothesis
    X: input
    Y: output
    """
    if X.shape[1] + 1 != theta.size or X.shape[0] != Y.size:
        print("Inc dim")
        return
    c = cost_elem_(theta, X, Y)
    s = 0
    for i in c:
        s = s + i
    return(s)

if __name__ == "__main__":
    X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    theta1 = np.array([[2.], [4.]])
    Y1 = np.array([[2.], [7.], [12.], [17.], [22.]])
    print(cost_elem_(theta1, X1, Y1))
    print(cost_(theta1, X1, Y1))

    X2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8.,
    80.]])
    theta2 = np.array([[0.05], [1.], [1.], [1.]])
    Y2 = np.array([[19.], [42.], [67.], [93.], [90.]])
    print(cost_elem_(theta2, X2, Y2))
    print(cost_(theta2, X2, Y2))