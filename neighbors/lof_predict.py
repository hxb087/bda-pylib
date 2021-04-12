#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author      : liujunfeng
# @Description : 局部异常因子异常值检测预测算子

import argparse
import pandas as pd

import os
import sys
sys.path.append(os.path.join(os.getcwd(),"bda-pylib"))
from util.df_utils import DFUtils
from util.model_utils import ModelUtils


# TODO：预测模块相同逻辑的整合，BaseInference

def main(args):
    # load data
    columns, clf = ModelUtils.load_model(args.model_path)
    test_df = pd.read_csv(args.test_data_path)[columns]
    test_data = test_df.values

    # inference
    y_pre = clf.predict(test_data)
    DFUtils.save(test_df, y_pre, args.output_path)
    print("inference successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='one class svm program predict...')
    parser.add_argument('--test_data_path', type=str, default="", help='the path of test file')
    parser.add_argument('--model_path', type=str, default="", help='the path of model file')
    parser.add_argument('--output_path', type=str, default="", help='the path of predict file')

    args = parser.parse_args()
    main(args)
