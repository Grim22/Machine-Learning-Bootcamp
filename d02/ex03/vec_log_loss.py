import numpy as np
import sys
sys.path.insert(1, '../ex00')
from sigmoid import sigmoid_

def vec_log_loss_(y_true, y_pred, m, eps=1e-15):
    """
    Compute the logistic loss value.
    Args:
        y_true: a scalar or a numpy ndarray for the correct labels
        y_pred: a scalar or a numpy ndarray for the predicted labels
        m: the length of y_true (should also be the length of y_pred)
        eps: epsilon (default=1e-15)
    Returns:
        The logistic loss value as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
"""
    if m > 1:
        if y_true.size != m or y_pred.size != y_true.size:
            return(None)
        return(-1/m * (np.dot(y_true, np.log(y_pred)) + np.dot((1 - y_true), np.log(1 - y_pred))))
    return(y_true * np.log(y_pred + eps) + (1 - y_true)*np.log(1 - y_pred))

if __name__ == "__main__":
    # Test n.1
    x= 4
    y_true = 1
    theta = 0.5
    y_pred = sigmoid_(x * theta)
    m = 1 # length of y_true is 1 
    print(vec_log_loss_(y_true, y_pred, m)) 
    # 0.12692801104297152
    
    # Test n.2
    x = np.array([1, 2, 3, 4])
    y_true = 0
    theta = np.array([-1.5, 2.3, 1.4, 0.7]) 
    y_pred = sigmoid_(np.dot(x, theta)) 
    m= 1
    print(vec_log_loss_(y_true, y_pred, m)) 
    # 10.100041078687479
    
    # Test n.3
    x_new = np.arange(1, 13).reshape((3, 4))
    y_true = np.array([1, 0, 1])
    theta = np.array([-1.5, 2.3, 1.4, 0.7])
    y_pred = sigmoid_(np.dot(x_new, theta))
    m = len(y_true)
    print(vec_log_loss_(y_true, y_pred, m))
    # 7.233346147374828