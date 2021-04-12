# 线性回归
import ast

import numpy as np  # 快速操作结构数组的工具
from sklearn.linear_model import LinearRegression  # 线性回归

import matplotlib.pyplot as plt
import argparse
import pandas as pd


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    # 样本数据集，第一列为x，第二列为y，在x和y之间建立回归模型
    df = pd.read_csv(args.input)

    # 生成X和y矩阵
    features= args.feature.split(",")
    X = np.array(df[features])
    y = np.array(df[args.label]).reshape(-1, 1)
    # print (X,y)
    norm = str2bool(args.normalize)
    fit_inter = str2bool(args.fitIntercept)

    # ========线性回归========
    model = LinearRegression(copy_X=True, fit_intercept=fit_inter, n_jobs=args.n_jobs, normalize=norm)
    model.fit(X, y)  # 线性回归建模
    print('系数矩阵:\n', model.coef_)
    print('系数矩阵:\n', model.intercept_)
    print('线性回归模型:\n', model)
    print(fit_inter, norm)
    # 使用模型预测
    predicted = model.predict(X)

    # # 绘制散点图 参数：x横轴 y纵轴
    # plt.figure(facecolor='white', figsize=(10, 5))
    # plt.scatter(X, y, marker='x')
    # plt.plot(X, predicted, c='r')
    #
    # title = 'The  Linear Graph of One Dimension'
    # # 绘制x轴和y轴坐标
    # plt.xlabel(args.feature)
    # plt.ylabel(args.label)
    # plt.title(title)
    # plt.grid()
    # # 显示图形
    # plt.savefig(args.data_png, format='png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='input: input file name.')
    parser.add_argument('--feature', type=str, default="", help='choose columns from test file')
    parser.add_argument('--label', type=str, default="", help='choose label column from training file')
    parser.add_argument('--fitIntercept', type=str, default=True, help='the fitIntercept')
    parser.add_argument('--normalize', type=str, help='the fitIntercept')
    parser.add_argument('--n_jobs', type=int, default=1, help='the n_jobs')
    parser.add_argument('--output', default='linear_data.csv', help='out file name.')
    # parser.add_argument('--data_png', type=str, default="../out/png/linear_plot.png", help='plot of png')

    args = parser.parse_args()
    main(args)
#python linearRegression1Dimension.py --input ../data/linear_test.csv --feature V1 --label label --n_jobs 2 --fitIntercept False --normalize True
