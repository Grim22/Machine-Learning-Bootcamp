import numpy as np
from sklearn.metrics import accuracy_score

def accuracy_score_(y_true, y_pred):
    s = 0
    for i in range(y_true.size):
        if y_true[i] == y_pred[i]:
            s = s + 1
    return(s / y_pred.size)


if __name__ == "__main__":
    # Test n.1
    y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
    y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    print(accuracy_score_(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))
    # 0.5 # 0.5
    # Test n.2
    y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog','dog', 'dog'])
    y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet','dog', 'norminet'])
    print(accuracy_score_(y_true, y_pred))
    print(accuracy_score(y_true, y_pred))
    # 0.625
    # 0.625