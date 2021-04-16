#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def Significant(train_data_path, type, col, label):
    raw_df = pd.read_csv(train_data_path)
    columns = col.split(",")
    features_df = raw_df[columns]
    train_data = StandardScaler().fit_transform(features_df.values)
    y = np.array(raw_df[label])
    alpha = 0.1
    max_iter = 1000
    l1_ratio = 0.5
    normalize = True
    tol = 0.0001
    selection_path = "out/select_linear.csv"

    if type == 'elasticNet':
        clf = linear_model.ElasticNet(alpha=alpha, copy_X=True, fit_intercept=True, l1_ratio=l1_ratio,
                                      max_iter=max_iter, normalize=normalize, positive=False, precompute=False,
                                      random_state=None, selection='cyclic', tol=tol, warm_start=False)
    elif type == 'lasso':
        clf = linear_model.Lasso(alpha=alpha, copy_X=True, fit_intercept=True, max_iter=max_iter,
                                 normalize=normalize, positive=False, precompute=False, random_state=None,
                                 selection='cyclic', tol=tol, warm_start=False)


    elif type == 'ridge':
        clf = linear_model.Ridge(alpha=alpha, copy_X=True, fit_intercept=True, max_iter=max_iter,
                                 normalize=normalize, random_state=None, solver='auto', tol=tol)
    else:
        print("warning!!!! the type is wrong!!")
        return

    clf.fit(train_data, y)

    print(clf.fit(train_data, y).coef_)

    coef = clf.fit(train_data, y).coef_

    select_df = pd.DataFrame(np.array(coef).reshape(1, -1), columns=columns)
    select_df.to_csv(selection_path, index=None, sep=",")

    y_pred = clf.fit(train_data, y).predict(train_data)
    print(f"The model {type} MSE:   ", mean_squared_error(y, y_pred))
