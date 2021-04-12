#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        : 2020/10/16 14:20
# @Author      : liujunfeng
# @Software    : PyCharm
# @Description : kmo 检验统计量
# 增加输入输出文件


import numpy as np
import math as math
import pandas as pd
import argparse


def corr(data):
    return np.corrcoef(data)


def kmo(dataset_corr):
    corr_inv = np.linalg.inv(dataset_corr)
    nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
    data = np.ones((nrow_inv_corr, ncol_inv_corr))
    for i in range(0, nrow_inv_corr, 1):
        for j in range(i, ncol_inv_corr, 1):
            data[i, j] = -(corr_inv[i, j]) / (math.sqrt(corr_inv[i, i] * corr_inv[j, j]))
            data[j, i] = data[i, j]
    dataset_corr = np.asarray(dataset_corr)
    kmo_num = np.sum(np.square(dataset_corr)) - np.sum(np.square(np.diagonal(data)))
    kmo_denom = kmo_num + np.sum(np.square(data)) - np.sum(np.square(np.diagonal(data)))
    kmo_value = kmo_num / kmo_denom
    return kmo_value


def main(args):
    # dataset = np.array([[3, 5, 1, 4, 1],
    #                     [4, 4, 3, 5, 3],
    #                     [3, 4, 4, 4, 4],
    #                     [3, 3, 5, 2, 1],
    #                     [3, 4, 5, 4, 3]])

    cols=args.cols.split(",")
    df = pd.read_csv(args.input)
    if(len(cols) != len(df)):
        print("Waring!!!!!   the length of csv is not equal as cols ")

    dataset = df.head(len(cols))[cols].values

    dataset_corr = corr(dataset)
    kmo_value = kmo(dataset_corr)
    print("kmo统计值：{}".format(kmo(dataset_corr)))
    df= pd.DataFrame({'kmo_value':[kmo_value]})
    # print(df)
    df.to_csv(args.output,index=False)



if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', required=True, help='input: input file name.')
        parser.add_argument('--cols', required=True, help='input: columns value.')
        parser.add_argument('--output', type=str, default="aa.csv", help='the path of test file')

        args = parser.parse_args()
        main(args)

#python bda-pylib/statistics/kmo_analysis.py --input bda-pylib/data/linear_data.csv --cols V1,V2,V3,V4,V5