#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn import linear_model
import argparse
from sklearn.metrics import mean_squared_error


def main(args):

    raw_df = pd.read_csv(args.train_data_path)
    columns = args.features.split(",")
    features_df = raw_df[columns]
    train_data = StandardScaler().fit_transform(features_df.values)
    y = np.array(raw_df[args.label])

    if args.type == 'elasticNet':
        clf = linear_model.ElasticNet(alpha=args.alpha, copy_X=True, fit_intercept=args.fitIntercept,l1_ratio=args.l1_ratio,
                            max_iter=args.max_iter, normalize=args.normalize, positive=False, precompute=False,
                            random_state=None, selection='cyclic', tol=args.tol, warm_start=False)
    elif args.type == 'lasso':
        clf = linear_model.Lasso(alpha=args.alpha, copy_X=True, fit_intercept=args.fitIntercept, max_iter=args.max_iter,
                                 normalize=args.normalize, positive=False, precompute=False, random_state=None,
                                 selection='cyclic', tol=args.tol, warm_start=False)
    elif args.type == 'radge':
        clf = linear_model.Ridge(alpha=args.alpha, copy_X=True, fit_intercept=args.fitIntercept, max_iter=args.max_iter,
                                 normalize=args.normalize, random_state=None, solver='auto', tol=args.tol)

    clf.fit(train_data, y)
    joblib.dump({"columns": columns, "model": clf}, args.model_path)
    y_pred = clf.fit(features_df.values, y).predict(features_df.values)

    print(f"The model {args.type} MSE:   ",mean_squared_error(y, y_pred))



if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='linear_model  program train...')
        parser.add_argument('--train_data_path', type=str, default="", help='the path of test file')
        parser.add_argument('--model_path', type=str, default="", help='the path of model file')
        parser.add_argument('--features', type=str, default="", help='choose columns from test file')
        parser.add_argument('--label', type=str, default="", help='the label')
        parser.add_argument('--alpha', type=float, default=1.0, help='the alpha')
        parser.add_argument('--fitIntercept', type=bool, default=True, help='the fitIntercept')
        parser.add_argument('--l1_ratio', type=float, default=0.5, help='the l1_ratio')
        parser.add_argument('--max_iter', type=int, default=5, help='the max_iter')
        parser.add_argument('--normalize', type=bool, default=False, help='the normalize')
        parser.add_argument('--tol', type=float, default=0.001, help='the tol')
        parser.add_argument('--type', type=str, default="lasso", help='lasso,radge,elasticNet')

        args = parser.parse_args()
        main(args)


#python linearTrain.py --train_data_path ../data/linear_data.csv --model_path D:\bda-pylib\out\model\linearModel --features V1,V2,V3,V4,V5 --label label --alpha 1.0 --fitIntercept True --l1_ratio 0.2  --max_iter 1000 --normalize False --tol 0.001 --type elasticNet
