# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 16:50:31 2016

@author: DMalygin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as pre
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

X = pd.read_csv('./Datasets/parkinsons.data', index_col=0)
y = X['status']
X.drop('status', axis=1, inplace=True)

# split dataset into 30% test samples with rnd state 7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=7)

# transforming data
scalers = [pre.Normalizer, pre.MaxAbsScaler, pre.MinMaxScaler, 
           pre.KernelCenterer, pre.StandardScaler]
for scaler in scalers:
    scaler = scaler()
    scaler.fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    # training support vector classifier for training set
    svc = SVC()
    svc.fit(X_train_s, y_train)
    score = svc.score(X_test_s, y_test)
    print('Score for SVC with default parameters is {}. Features scaled with {}'.format(score, scaler.__class__))

# performing naive best parameter search
# C from 0.05 to 2 with 0.05 step
# gamma from 0.001 to 0.1 with 0.001 step

best_parameters = {'best_score': 0,
                   'C': 0.05,
                   'gamma': 0.001}
for scaler in scalers:
    scaler = scaler()
    scaler.fit(X_train)
    X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
    for n_components in range(4, 15):
#        pca = PCA(n_components=n_components)
#        pca.fit(X_train_s)
#        X_train_s_r, X_test_s_r = pca.transform(X_train_s), pca.transform(X_test_s)
        iso = Isomap(n_components=n_components)
        iso.fit(X_train_s)
        X_train_s_r, X_test_s_r = iso.transform(X_train_s), iso.transform(X_test_s)
        for C in np.arange(0.05, 2, 0.05):
            for gamma in np.arange(0.001, 0.1, 0.001):
                svc = SVC(C=C, gamma=gamma)
                svc.fit(X_train_s_r, y_train)
                score = svc.score(X_test_s_r, y_test)
                if score > best_parameters['best_score']:
                    best_parameters['best_score'] = score
                    best_parameters['C'] = C
                    best_parameters['gamma'] = gamma
# show best result by unpacking dict into formatted string
print('Highest score is {best_score} for C={C}, gamma={gamma}'.format(**best_parameters))