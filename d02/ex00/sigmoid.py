import numpy as np

def sigmoid_(x):
    """
    Compute the sigmoid of a scalar or a list.
    Args:
        x: a scalar or list
    Returns:
        The sigmoid value as a scalar or list.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if isinstance(x, list):
        for i in range(len(x)):
            x[i] = 1 / (1 + np.exp(-x[i]))
        return(x)
    else:
        return(1 / (1 + np.exp(-x)))

if __name__ == "__main__":
    x = -4
    print(sigmoid_(x))
    # 0.01798620996209156
    x= 2
    print(sigmoid_(x))
    # 0.8807970779778823
    x = [-4, 2, 0]
    print(sigmoid_(x))
    # [0.01798620996209156, 0.8807970779778823, 0.5]