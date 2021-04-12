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
from fbprophet import Prophet



def main(args):
    np.random.seed(0)

    # load data
    df = pd.DataFrame()

    files = os.listdir(args.train_data_path)

    for fn in files:
        # print(fn)
        if fn.startswith("part-"):
            # print(fn)
            filename = args.train_data_path + '/' + fn
            df1 = pd.read_csv(filename)
            # df=df1
            df= pd.concat([df1, df], axis=0)
            # print(df1.head(2))
            # df = df.append(df1)
            # print(df.shape)
    df.drop_duplicates(inplace=True)
    df.set_index('id', inplace=True)
    df.sort_index(inplace=True)
    # df.to_csv("aa1.csv")
    df.reset_index(inplace=True)
    # print(df.shape)

    print(df.head(5))
    print(df.shape)

    df= df[[args.dt, args.y]]
    df.columns = ['ds', 'y']

    #model train
    m = Prophet(yearly_seasonality= False,weekly_seasonality=False,daily_seasonality=False)
    m.fit(df)
    # future = m.make_future_dataframe(periods=30, freq='10min')
    # print(m.predict(future))

#model save
    f = open(args.model_path, 'wb')
    dill.dump(m, f)
    f.close()

#model predict
    f = open(args.model_path, 'rb')
    param_dict = dill.load(f)

    predict_df = param_dict.predict(df)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    print(predict_df.tail())
    f.close()

    df.columns = ['dt', 'y']
    result = predict_df.join(df['y']).head(100)
    # print(result)
    result.to_csv(args.out_path, index=None, sep=",")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='prophet model program training...')
    parser.add_argument('--train_data_path', type=str, default="", help='the path of training file')
    parser.add_argument('--dt', type=str, default="", help='choose dt columns from training file')
    parser.add_argument('--y', type=str, default="", help='choose y column from training file')
    # parser.add_argument('--periods', type=int, default=30, help='the periods of predict data')
    # parser.add_argument('--freq', type=str, default="10min", help='the freq of data')
    parser.add_argument('--model_path', type=str, default="", help='the path of model file')
    parser.add_argument('--out_path', type=str, default="prophet_train.csv", help='the path of model file')


    args = parser.parse_args()
    main(args)

#python -W ignore bda-pylib/TimeSeriesAnalysis/prophet_train.py --train_data_path bda-pylib/data/prophet_predict --dt rectime  --y averageWindSpeed --model_path bda-pylib/out/model/prophet_model --out_path prophet_train.csv
