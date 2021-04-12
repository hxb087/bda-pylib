#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time        : 2020/10/16 14:26
# @Author      : liujunfeng
# @Software    : PyCharm
# @Description : bartlett平均方差检验

import numpy as np
from scipy.stats import bartlett
import pandas as pd
import argparse


def corr(data):
    return np.corrcoef(data)


def main(args):
    # dataset = np.array([[3, 5, 1, 4, 1],
    #                     [4, 4, 3, 5, 3],
    #                     [3, 4, 4, 4, 4],
    #                     [3, 3, 5, 2, 1],
    #                     [3, 4, 5, 4, 3]])

    cols = args.cols.split(",")

    dataset = pd.read_csv(args.input)[cols].values

    dataset_corr = corr(dataset)

    stat, p = bartlett(*dataset_corr)
    # stat, p = bartlett(dataset_corr[0],dataset_corr[2])


    # stat, p = bartlett(dataset_corr[0], dataset_corr[1], dataset_corr[2], dataset_corr[3], dataset_corr[4])
    print("巴特利特检验概率P值：{}".format(p))

    df = pd.DataFrame({'bartlett_value': [p]})
    # print(df)
    df.to_csv(args.output, index=False)

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', required=True, help='input: input file name.')
        parser.add_argument('--cols', required=True, help='input: columns value.')
        parser.add_argument('--output', type=str, default="aa.csv", help='the path of test file')
        args = parser.parse_args()
        main(args)

#python bda-pylib/statistics/bartlett_analysis.py --input bda-pylib/data/linear_data.csv --cols V3,V4,V5,label  --output bda-pylib/out/bartlett.csv
