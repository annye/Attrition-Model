'''

Attrition Classifer

Author: Annye Braca

Date: 06/02/18

'''
from __future__ import print_function, division
import os
import pandas as pd 
import numpy as np 
import matplotlib  as plt 
import sys
import tensorflow as tf 
from sklearn import feature_extraction
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Conv1D
from keras import optimizers


def set_trace():
    """A Poor mans break point"""
    # without this in iPython debugger can generate strange characters.
    from IPython.core.debugger import Pdb
    Pdb().set_trace(sys._getframe().f_back)

def read_data():
    data = pd.read_csv('watson.csv')
    data.drop('Over18', axis=1, inplace=True)
    return data


def one_hot_dataframe(data, cols, replace=False):    
    vec = feature_extraction.DictVectorizer()

    def mkdict(row): return dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(
        data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return data


def get_train_test_split(df):
    # Get X and Y for train_test
    y = df['Attrition'].to_frame()
    target_map = {'Yes':1, 'No': 0}
    y['Target'] =y['Attrition'].map(target_map)
    y.drop('Attrition', axis=1, inplace=True)
    df.drop('Attrition', axis=1, inplace=True)
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)    
    return X_train, X_test, y_train, y_test


def run_classifier(X_train, X_test, y_train, y_test):
    # Defining Sequential model
    model = Sequential()
    model.add(Conv1D(1, 3, activation='relu', input_shape=(54,1)))
    model.add(Dense(2, activation='softmax'))

    # Summary to list al layers, output shapes and no. of params
    model.summary()

    model.compile(optimizer=optimizers.Adam(lr=0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train,
              epochs=30,
              batch_size=32,
              validation_data=(X_test, y_test),
              callbacks=[PlotLosses(figsize=(10,4)),
              tensorboard('logistic')])
    set_trace()

def main():
    data = read_data()
    df = one_hot_dataframe(data, ['Department',
                                  'BusinessTravel',
                                  'EducationField',
                                  'Gender',
                                  'JobRole',
                                  'MaritalStatus',
                                  'OverTime',
                                  'EmployeeCount',
                                  'EmployeeNumber'],
                              replace=True)
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    run_classifier(X_train, X_test, y_train, y_test)


    set_trace()

if __name__ == "__main__":
    main()
