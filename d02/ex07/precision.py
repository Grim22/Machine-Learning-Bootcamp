import numpy as np
from sklearn.metrics import precision_score

def precision_score_(y_true, y_pred, pos_label=1):
    tp = 0
    fp = 0
    for i in range(y_pred.size):
        if y_pred[i] == pos_label:
            if y_true[i] == pos_label:
                tp = tp + 1
            else:
                fp = fp + 1
    return(float(tp/(tp + fp)))


if __name__ == "__main__":
        
    # Test n.1
    y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
    y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    print(precision_score_(y_true, y_pred))
    print(precision_score(y_true, y_pred))
    # 0.4 # 0.4
    # Test n.2
    y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog',
    'dog', 'dog'])
    y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet',
    'dog', 'norminet'])
    print(precision_score_(y_true, y_pred, pos_label='dog'))
    print(precision_score(y_true, y_pred, pos_label='dog'))
    # 0.6
    # 0.6
    # Test n.3
    print(precision_score_(y_true, y_pred, pos_label='norminet'))
    print(precision_score(y_true, y_pred, pos_label='norminet'))
    # 0.6666666666666666
    # 0.6666666666666666