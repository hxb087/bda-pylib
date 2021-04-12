#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author      : huxb
# @Description : lr 逻辑回归因为分布式的逻辑回归是适合文字识别用的，所以建立新的逻辑回归

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV, LinearRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler
import argparse

import os
import sys

sys.path.append(os.path.join(os.getcwd(), "bda-pylib"))
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
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)  ## 训练模型及归一化数据
    lr = LogisticRegressionCV(fit_intercept=True, Cs=np.logspace(-2, 2, 20), cv=2, penalty='l2', solver='lbfgs',
                              tol=0.01)
    lr.fit(x_train, y_train)
    x_test = ss.fit_transform(x_test)
    r = lr.score(x_test, y_test)
    print("R值（准确率）：", r)

    ModelUtils.save_model(columns, lr, args.model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='one class svm program training...')
    parser.add_argument('--train_data_path', type=str, default="", help='the path of training file')
    parser.add_argument('--features', type=str, default="", help='choose columns from training file')
    parser.add_argument('--label', type=str, default="", help='choose label column from training file')
    parser.add_argument('--model_path', type=str, default="", help='the path of model file')

    args = parser.parse_args()
    main(args)


#python bda-pylib/classification/lr_train.py --train_data_path  bda-pylib/data/breast-cancer-wisconsin.csv --features ClumpThickness,UniformityOfCellSize,UniformityOfCellShape,MarginalAdhesion,SingleEpithelialCellSize,BareNuclei,BlandChromatin,NormalNucleoli,Mitoses
# --label Class --model_path  bda-pylib/out/model/lr_Model