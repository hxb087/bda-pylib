#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author      : liujunfeng
# @Description : one class svm的离群值检测

import argparse
import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

import os
import sys
sys.path.append(os.path.join(os.getcwd(),"bda-pylib"))
from util.model_utils import ModelUtils


def main(args):
    np.random.seed(452346324)
    # load data
    columns = args.features.split(",")
    raw_df = pd.read_csv(args.train_data_path)
    data = raw_df[columns].values
    targets = raw_df[args.label].values
    x_train, x_test, y_train, y_test = train_test_split(data, targets, train_size=0.8)

    # fit the model
    clf = svm.OneClassSVM(nu=args.nu, kernel=args.kernel, gamma=args.gamma)
    clf.fit(x_train)
    y_pre = clf.predict(x_test)
    print(metrics.classification_report(y_test, y_pre, target_names=["outlier", "normValue"]))
    ModelUtils.save_model(columns, clf, args.model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='one class svm program training...')
    parser.add_argument('--train_data_path', type=str, default="", help='the path of training file')
    parser.add_argument('--nu', type=float, default=0.1, help="An upper bound on the fraction of training "
                                                              "errors and a lower bound of the fraction of support")
    parser.add_argument('--features', type=str, default="", help='choose columns from training file')
    parser.add_argument('--label', type=str, default="", help='choose label column from training file')
    parser.add_argument('--kernel', type=str, default="rbf", help='Specifies the kernel type to be used '
                                                                  'in the algorithm.')
    parser.add_argument('--gamma', type=float, default=0.1, help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.")
    parser.add_argument('--model_path', type=str, default="", help='the path of model file')

    args = parser.parse_args()
    main(args)
