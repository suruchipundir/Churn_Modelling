#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 22:45:55 2019
Predicting customers who are more likely to leave the bank using ANN
@author: suruchi
"""
#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Churn_Modelling.csv')
X = data.iloc[:, 3:13].values
Y = data.iloc[:, -1].values
#Encoding into categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x1 = LabelEncoder()
X[:, 1] = labelencoder_x1.fit_transform(X[:, 1])
labelencoder_x2 = LabelEncoder()
X[:, 2] = labelencoder_x1.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
#Deleting 1st column here to avoid dummy variable trap
X = X[:, 1:]
#Splitting data into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.fit_transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
#Initializing ANN
classifier = Sequential()
#Adding input and 1st hidden layer such that there are 6 nodes and weights are uniform and small
#Using rectifier activation function for our hidden layer with dropout  
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
classifir.add(Dropout(p=0.1))
#Adding 2nd hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
classifir.add(Dropout(p=0.1))
#Adding output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
#Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=10, epochs=100)
#PRedicting on x_test
y_pred = classifier.predict(X_test)
y_pred = (y_pred>0.5)
from sklearn.metrics import confusion_matrix
cm  = confusion_matrix(y_test, y_pred)
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc_x.fit_transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction>0.5)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
mean = accuracies.mean()
var = accuracies.std()
#Tuning Parameters
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           n_jobs=-1,
                           cv=10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_