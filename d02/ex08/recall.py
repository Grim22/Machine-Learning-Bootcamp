 
import numpy as np
from sklearn.metrics import recall_score


def recall_score_(y_true, y_pred, pos_label=1):
    tp = 0
    fn = 0
    for i in range(y_true.size):
        if y_true[i] == pos_label:
            if y_pred[i] == pos_label:
                tp = tp + 1
            else:
                fn = fn + 1
    return(float(tp / (tp + fn)))


if __name__ == "__main__":

    # Test n.1
    y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1])
    y_true = np.array([1, 0, 0, 1, 0, 1, 0, 0])
    print(recall_score_(y_true, y_pred))
    print(recall_score(y_true, y_pred))
    # 0.6666666666666666
    # 0.6666666666666666
    # Test n.2
    y_pred = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog',
    'dog', 'dog'])
    y_true = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet',
    'dog', 'norminet'])
    print(recall_score_(y_true, y_pred, pos_label='dog'))
    print(recall_score(y_true, y_pred, pos_label='dog'))
    # 0.75
    # 0.75
    # Test n.3
    print(recall_score_(y_true, y_pred, pos_label='norminet'))
    print(recall_score(y_true, y_pred, pos_label='norminet'))
    # 0.5
    # 0.5