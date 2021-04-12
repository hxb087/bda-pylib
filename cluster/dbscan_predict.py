#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author      : liujunfeng
# @Description : dbscan聚类算子预测

import joblib
import argparse
import pandas as pd


def main(args):
    # 加载数据和模型
    param_dict = joblib.load(args.model_path)
    test_df = pd.read_csv(args.test_data_path)
    columns, db_model = param_dict["columns"], param_dict["model"]
    pre_df = test_df[columns]
    pre = db_model.fit_predict(pre_df.values)

    pre_df["prediction"] = pre
    pre_df.to_csv(args.output_path, index=None, sep=",")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dbscan program predict...')
    parser.add_argument('--test_data_path', type=str, default="", help='the path of test file')
    parser.add_argument('--model_path', type=str, default="", help='the path of model file')
    parser.add_argument('--output_path', type=str, default="", help='the path of predict file')

    args = parser.parse_args()
    main(args)
