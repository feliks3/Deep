import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score 
from sklearn.datasets import load_svmlight_file

class Perceptron:
    def init(self):
        print("init")
        
        
        
    def load(self):
        self.X, self.y = load_svmlight_file('diabetes_scale')
        self.X = np.array(self.X.todense())
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def pre_processing(self, randomInit, epoch, learning_rate):
        self.randomInit = randomInit
        if randomInit:
            self.W = np.random.randn(len(self.X_train[0])+1)
        else:
            self.W = np.zeros([len(self.X_train[0])+1])
    
        self.epoch = epoch
        self.learning_rate = learning_rate
    
    def train_data(self):
        self.X_train_b = np.c_[np.ones(len(self.X_train)), self.X_train]
        for _ in range(self.epoch):
            step = np.zeros([len(self.X_train_b[0])])
            for i in range(len(self.X_train_b)):
                if self.y_train[i] * np.dot(self.X_train_b[i], self.W) <= 0:
                    step += self.y_train[i] * self.X_train_b[i]
            self.W = self.W + self.learning_rate * step

    
    
    def predict(self, x):
        x = np.c_[np.ones(len(x)), x]
        self.g = np.dot(x,self.W)
        s = np.sign(self.g)
        return s

    
    def getAccuracy(self):

        self.train_data()

        self.y_predict_train = self.predict(self.X_train)
        self.accuracy_train = accuracy_score(self.y_train , self.y_predict_train)    

        self.y_predict_test = self.predict(self.X_test)
        self.accuracy_test = accuracy_score(self.y_test, self.y_predict_test)    
    
    def getRisk(self):
        self.loss = 0
        for i in range(len(self.X_train)):
            self.loss += np.maximum(0, -self.y_train[i] * np.dot(self.X_train_b[i], self.W))
        self.empirical_risk = self.loss/len(self.X_train)
        print(self.empirical_risk)
    

    def showResult(self):
        self.getAccuracy()
        print("accuracy_train" ,self.accuracy_train , "accuracy_test", self.accuracy_test, self.epoch, self.learning_rate, self.randomInit)
        self.getRisk()



for randomInit in [True, False]:
    for epoch in [10 ,50, 100, 500]:
        for learning_rate in [0.1, 0.01, 0.001, 0.0001]:
            perceptron = Perceptron()
            perceptron.init()
            perceptron.load()
            perceptron.pre_processing(randomInit, epoch, learning_rate)
            perceptron.train_data()
            perceptron.showResult()
            

