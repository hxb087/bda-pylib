#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author      : huxb
# @Description : 时间序列模型

import argparse
import numpy as np
import pandas as pd
import dill

import os
import sys
sys.path.append(os.path.join(os.getcwd(),"bda-pylib"))
# from fbprophet import Prophet


def main(args):
    np.random.seed(0)

    # load data

    df= pd.DataFrame()

    files = os.listdir(args.test_data_path)

    for fn in files:
        # print(fn)
        if fn.endswith(".csv"):
            filename = args.test_data_path + '/' + fn
            df1 = pd.read_csv(filename)[[args.dt, args.y]]
            # print(df1)
            # df=df.append(df1)
            df = pd.concat([df1, df], axis=0)
            # print(fn)

    print(df.shape)
    df.columns = ['ds', 'y']


    print(df.head(3))
    df.drop_duplicates(inplace=True)
    df.set_index('ds',inplace=True)
    df.sort_index(inplace=True)
    # df.to_csv("aa1.csv")
    df.reset_index(inplace=True)
    # print(df.shape)
    df.to_csv("aa.csv")



    f = open(args.model_path, 'rb')
    param_dict = dill.load(f)
    # df_hat = param_dict.predict(df)
    df_hat= param_dict.predict(df)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    f.close()
    # print(df_hat.shape)
    # df_hat.set_index('ds')
    # df_hat.to_csv("bb.csv")
    df_hat.to_csv(args.out_path, index=None, sep=",")


    # result = df_hat.join(df['y'])
    # # print(result)
    # result.to_csv(args.out_path, index=None, sep=",")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prophet model program training...')
    parser.add_argument('--test_data_path', type=str, default="", help='the path of training file')
    parser.add_argument('--dt', type=str, default="", help='choose dt columns from training file')
    parser.add_argument('--y', type=str, default="", help='choose y column from training file')
    parser.add_argument('--model_path', type=str, default="", help='the path of model file')
    parser.add_argument('--out_path', type=str, default="aa", help='the path of out file')

    args = parser.parse_args()
    main(args)

#python -W ignore bda-pylib/TimeSeriesAnalysis/prophet_predict.py --test_data_path bda-pylib/data/Wind_power_generator_test.csv --dt rectime --y averageWindSpeed  --model_path bda-pylib/out/model/prophet_model --out_path prophet_predict.csv

#python -W ignore bda-pylib/TimeSeriesAnalysis/prophet_predict.py --test_data_path bda-pylib/data/prophet_predict --dt rectime --y averageWindSpeed --model_path bda-pylib/out/model/prophet_model --out_path prophet_predict.csv