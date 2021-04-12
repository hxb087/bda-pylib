#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author      : liujunfeng
# @Description : 局部异常因子异常值检测训练算子

import argparse
import numpy as np
import pandas as pd
from sklearn import metrics

import os
import sys
sys.path.append(os.path.join(os.getcwd(),"bda-pylib"))
from util.model_utils import ModelUtils
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split


def main(args):
    np.random.seed(0)

    # load data
    columns = args.features.split(",")
    raw_df = pd.read_csv(args.train_data_path)
    data, targets = raw_df[columns], raw_df[args.label]

    x_train, x_test, y_train, y_test = train_test_split(data, targets, train_size=0.8)

    # fit the model for outlier detection (default)
    lof = LocalOutlierFactor(n_neighbors=args.n_neighbors, novelty=True, n_jobs=args.n_jobs).fit(x_train)

    y_pre = lof.predict(x_test)

    print(metrics.classification_report(y_test, y_pre, target_names=["outlier", "normValue"]))

    ModelUtils.save_model(columns, lof, args.model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LocalOutlierFactor program training...')
    parser.add_argument('--train_data_path', type=str, default="", help='the path of training file')
    parser.add_argument('--n_neighbors', type=int, default=20,
                        help="The actual number of neighbors used for :meth:`kneighbors` queries.")
    parser.add_argument('--features', type=str, default="", help='choose columns from test file')
    parser.add_argument('--label', type=str, default="", help='choose label column from training file')

    parser.add_argument('--n_jobs', type=int, default=-1, help='The number of parallel jobs to run')
    parser.add_argument('--model_path', type=str, default="", help='the path of model file')

    args = parser.parse_args()
    main(args)


#python bda-pylib/neighbors/lof_train.py --train_data_path  bda-pylib/data/lof_sample.csv --n_neighbors 20 --features V0,V1 -
#-label label --n_jobs -1 --model_path bda-pylib/out/model/lof_model
