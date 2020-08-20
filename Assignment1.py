import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file

X, y = load_svmlight_file('diabetes_scale')
X = np.array(X.todense())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

def train_data(W,T,X_train,y_train,learning_rate):
    X_train = sklearn.preprocessing.normalize(X_train)
    X_train = np.c_[np.ones(len(X_train)), X_train]
    for j in range(T):
        step = np.zeros([len(X_train[0])])
        for i in range(len(X_train)):
            if y_train[i] * np.dot(X_train[i], W) <= 0:
                step += y_train[i] * X_train[i]
        W = W + learning_rate * step
    return W

def predict(x,W):
    x = np.c_[np.ones(len(x)), x]
    g = np.dot(x,W)
    s = np.sign(g)
    return s    


W = np.zeros([len(X_train[0])+1])
W = train_data(W,100,X_train, y_train, 0.001)
y_predict = predict(X_test,W)

accuracy = np.count_nonzero(y_test == y_predict)/np.count_nonzero(y_test)
print(accuracy)
