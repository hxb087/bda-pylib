#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author      : huxb
# @Description : lr 逻辑回归因为分布式的逻辑回归是适合文字识别用的，所以建立新的逻辑回归
import argparse
import pandas as pd

import os
import sys
sys.path.append(os.path.join(os.getcwd(),"bda-pylib"))
from util.df_utils import DFUtils
from util.model_utils import ModelUtils
from sklearn.preprocessing import StandardScaler


def main(args):
    # load data
    columns, lr = ModelUtils.load_model(args.model_path)
    test_df = pd.read_csv(args.test_data_path)[columns]
    test_data = test_df.values

    # fit the model
    ss = StandardScaler()
    test_data= ss.fit_transform(test_data)  ## 训练模型及归一化数据

    # inference
    y_pre = lr.predict(test_data)
    DFUtils.save(test_df, y_pre, args.output_path)
    print("inference successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='one class svm program predict...')
    parser.add_argument('--test_data_path', type=str, default="", help='the path of test file')
    parser.add_argument('--model_path', type=str, default="", help='the path of model file')
    parser.add_argument('--output_path', type=str, default="", help='the path of predict file')

    args = parser.parse_args()
    main(args)

#python bda-pylib/classification/lr_predict.py --test_data_path  bda-pylib/data/breast-cancer-wisconsin.csv  --model_path  bda-pylib/out/model/lr_Model --output_path bda-pylib/out/lr_pre.csv
