# -*- coding: utf-8 -*-
import numpy as np

def time_train_test_split(X,y,test_rate=0.2,shuffle=True,random_state=1):
    length = X.shape[0]


    test_size = int(length * test_rate)
    train_size = length - test_size

    X_train = X.iloc[:train_size]
    y_train = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]

    if shuffle:
        np.random.seed(random_state)
        idx = np.arange(train_size)
        np.random.shuffle(idx)
        X_train = X_train.iloc[idx]
        y_train = y_train.iloc[idx]

    return X_train,y_train,X_test,y_test
