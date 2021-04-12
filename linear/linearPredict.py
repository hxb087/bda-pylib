#!/usr/bin/env python
# -*- coding: utf-8 -*-

import joblib
import argparse
import pandas as pd

def main(args):

    param_dict = joblib.load(args.model_path)
    test_df = pd.read_csv(args.test_data_path)

    columns, db_model = param_dict["columns"], param_dict["model"]
    pre_df = test_df[columns]
    pre = db_model.predict(pre_df.values)
    pre_df["prediction"] = pre
    print(pre_df.head(5))
    pre_df.to_csv(args.output_path, index=None, sep=",")


if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='linear_model program predict...')
        parser.add_argument('--test_data_path', type=str, default="", help='the path of test file')
        parser.add_argument('--model_path', type=str, default="", help='the path of model file')
        parser.add_argument('--output_path', type=str, default="", help='the path of predict file')

        args = parser.parse_args()
        main(args)

#python linearPredict.py --test_data_path  ../data/linear_data.csv --model_path  D:\bda-pylib\out\model\linearModel --output_path ../out/linear_pre.csv

