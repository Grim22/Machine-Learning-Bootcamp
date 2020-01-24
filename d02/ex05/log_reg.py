    import numpy as np
import pandas as pd

class LogisticRegressionBatchGd:
    def __init__(self, alpha=0.001, max_iter=1000, verbose=False, learning_rate='constant'):
        self.alpha = alpha
        self.max_iter = max_iter
        self.verbose = verbose
        self.learning_rate = learning_rate 
        # can be 'constant' or'invscaling'
        self.thetas = []
        # Your code here (e.g. a list of loss for each epochs...)
    
    def fit(self, x_train, y_train):
        X_N = np.full((x_train.shape[0], x_train.shape[1] + 1), 1, float)
        X_N[:, 1:] = x_train
        self.thetas = np.zeros(x_train.shape[1])
        for i in enumerate(range(self.max_iter)):
            self.thetas = self.thetas - self.alpha * self.__gradient_(x_train, y_train, self.predict(x_train))
            if self.verbose and int(i[0]) % 100 == 0:
                print("epoch   " + str(i[0]) + "loss   " + str(self.__lossfc_(y_train, self.__sigmoid_(np.dot(x_train, self.thetas)), y_train.size)))
        return(self)
        
    def predict(self, x_train):
        # Your code here
        H = np.dot( x_train, self.thetas)
        a = self.__sigmoid_(H)
        b = a > 0.5
        return(b.astype(int))

    def score(self, x_train, y_train):
        # Your code here
        y_pred = self.predict(x_train)
        m = (y_pred == y_train).mean()
        return(m)

    def __lossfc_(self, y_true, y_pred, m, eps=1e-15):
        if m > 1:
            if y_true.size != m or y_pred.size != y_true.size:
                return(None)
            return(-1/m * (np.dot(y_true, np.log(y_pred + eps)) + np.dot((1 - y_true), np.log(1 - y_pred))))
        return(y_true * np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred))

    def __gradient_(self, x, y_true, y_pred):
        if isinstance(y_true, np.ndarray):
            if y_true.size != y_pred.size:
                print("Problem1")
                return(None)
            if y_true.size != x.shape[0]:
                print("Problem2")
                return(None)
        return(np.dot(y_pred - y_true, x))
    
    def __sigmoid_(self, x):
        if isinstance(x, list):
            for i in range(len(x)):
                x[i] = 1 / (1 + np.exp(-x[i]))
            return(x)
        else:
            return(1 / (1 + np.exp(-x)))


# We load and prepare our train and test dataset into x_train, y_train and x_test, y_test
df_train = pd.read_csv('train_dataset_clean.csv', delimiter=',',header=None, index_col=False)
x_train, y_train = np.array(df_train.iloc[:, 1:82]), df_train.iloc[:, 0]
df_test = pd.read_csv('test_dataset_clean.csv', delimiter=',', header=None,index_col=False)
x_test, y_test = np.array(df_test.iloc[:, 1:82]), df_test.iloc[:, 0]

# We set our model with our hyperparameters : alpha, max_iter, verbose andlearning_rate
model = LogisticRegressionBatchGd(alpha=0.0001, max_iter=1500, verbose=True,learning_rate='constant')

# We fit our model to our dataset and display the score for the train andtest datasets
model.fit(x_train, y_train)
print(f'Score on train dataset : {model.score(x_train, y_train)}')
y_pred = model.predict(x_test)
print(f'Score on test dataset : {(y_pred == y_test).mean()}')