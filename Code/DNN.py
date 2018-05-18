import pandas as pd 
import numpy as np 
import tensorflow as tf 
import keras as k 
from keras.models import Sequential
from keras.layers import Dense, Dropout

# fix random seed for reproducibility
# numpy.random.seed(7)

# load pima indians dataset
# dataset = np.loadtxt("Data.csv", delimiter=",")
# # split into input (X) and output (Y) variables
# X = dataset[:,0:8]
# Y = dataset[:,8]


# create model
model = Sequential()
model.add(Dense(31, input_dim=5, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(31, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])	



# Fit the model#
# model.fit(X, Y, epochs=1000, batch_size=50)
# model.save('DNN.h5')
# # evaluate the model
# scores = model.evaluate(X, Y)
# print(scores)
# print(model.metrics_names)