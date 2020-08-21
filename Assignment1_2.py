from keras import models 
from keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score 
from sklearn.datasets import load_svmlight_file

X, y = load_svmlight_file('diabetes_scale')
X = np.array(X.todense())
y = (y + 1)/2
train_images, test_images, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=42)

network = models.Sequential()
network.add(layers.Dense(8, activation='relu', input_shape=(8,))) 
network.add(layers.Dense(2, activation='softmax'))


network.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

network.fit(train_images, train_labels, epochs=100, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)


