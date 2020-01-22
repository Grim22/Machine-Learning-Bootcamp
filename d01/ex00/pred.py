import numpy as np

def predict_(theta, x):
    """
    returns an estimate of the output with x as an input an theta as an hypothesis
    """
    if x.shape[1] + 1 != theta.size:
        print("Inc dim")
        return(None)
    a = np.full((x.shape[0], x.shape[1] + 1), 1, float)
    a[:, 1:] = x
    return(np.dot(a, theta))
   
if __name__ == "__main__":
    X1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    theta1 = np.array([[2.], [4.]])
    print(predict_(theta1, X1))

    X2 = np.array([[1], [2], [3], [5], [8]])
    theta2 = np.array([[2.]])
    print(predict_(theta2, X2))

    X3 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8.,
    80.]])
    theta3 = np.array([[0.05], [1.], [1.], [1.]])
    print(predict_(theta3, X3))