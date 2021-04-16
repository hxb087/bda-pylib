#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')



def main(args):
    np.random.seed(9876789)

    df = pd.read_csv(args.train_data_path)

    feature = args.feature.split(",")
    s1 = ' + '.join(feature)
    s2 = args.label
    s = s2 + " ~ " + s1

    if args.type == "ols":
        results = smf.ols(s, data=df).fit(use_t=True)
    elif args.type == "gls":
        results = smf.gls(s, data=df).fit(use_t=True)
    elif args.type == "glsar":
        results = smf.glsar(s, data=df).fit(use_t=True)
    elif args.type == "wls":
        results = smf.wls(s, data=df).fit(use_t=True)
    else:
        print("No this type!!!")
        exit(0)

    print("**********************************************************************************\n")
    alpha = args.alpha
    # print(results.summary())

    data_t = {"coef": results.params, "std err": results.bse, "t": results.tvalues, "P>|t|": results.pvalues,
              "[" + str(alpha / 2.0): results.conf_int(alpha)[0],
              str(1 - alpha / 2.0) + "]": results.conf_int(alpha)[1]}

    sdata_df = pd.DataFrame(data_t)
    print(sdata_df)
    sdata_df.to_csv(args.output2)

    from statsmodels.stats.stattools import (
        jarque_bera, omni_normtest, durbin_watson)

    jb, jbpv, skew, kurtosis = jarque_bera(results.wresid)
    omni, omnipv = omni_normtest(results.wresid)


    title = ["Model", "R-squared", "Adj. R-squared", "F-statistic", "Prob (F-statistic)", "Log-Likelihood", "AIC",
             "BIC",
             "Omnibus", "Prob(Omnibus)", "Skew", "Kurtosis", "Durbin-Watson", "Jarque-Bera (JB)", "Prob(JB)",
             "Cond. No."]

    value = [results.model.__class__.__name__, results.rsquared, results.rsquared_adj, results.fvalue,
             results.f_pvalue, results.llf, results.aic, results.bic, omni, omnipv, skew, kurtosis,
             durbin_watson(results.wresid), jb, jbpv, results.diagn['condno']]

    datadf = {"title": np.array(title), "value": np.array(value)}

    select_df = pd.DataFrame(datadf)
    print(select_df)
    select_df.to_csv(args.output1)

    # 画1D或者3D图形
    predicted = results.predict(df)
    import matplotlib.pyplot as plt
    if len(feature) == 1:
        x = np.array(df[feature]).reshape(-1, 1)
        y = np.array(df[s2]).reshape(-1, 1)
        plt.figure(facecolor='white', figsize=(10, 5))
        plt.scatter(x, y, marker='x')
        plt.plot(x, predicted, c='r')

        title = 'The  Linear Graph of One Dimension'
        # 绘制x轴和y轴坐标
        plt.xlabel(feature[0])
        plt.ylabel(s2)
        plt.title(title)
        plt.grid()
        plt.savefig(args.data_png, format='png')

    elif len(feature) == 2:
        from mpl_toolkits.mplot3d import Axes3D
        ax1 = plt.axes(projection='3d')

        x = np.array(df[feature[0]]).reshape(-1, 1)
        y = np.array(df[feature[1]]).reshape(-1, 1)
        z = np.array(df[s2]).reshape(-1, 1)
        ax1.scatter3D(x, y, z, cmap='Blues')  # 绘制散点图
        ax1.plot3D(x, y, predicted, 'gray')  # 绘制空间曲线
        ax1.set_xlabel(feature[0])
        ax1.set_ylabel(feature[1])
        ax1.set_zlabel(s2)
        plt.savefig(args.data_png, format='png')
    else:
        print("The number of feature is big than 2 ,no plot!")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='linear_model  program train...')
    parser.add_argument('--train_data_path', type=str, default="", help='the path of test file')
    parser.add_argument('--feature', type=str, default="V1,V2,V3", help='choose columns from test file')
    parser.add_argument('--label', type=str, default="label", help='choose label column from training file')
    parser.add_argument('--alpha', type=float, default=0.05, help='the alpha')
    parser.add_argument('--type', type=str, default="ols", help='ols,gls,wls,glsar')
    parser.add_argument('--output1', default='linear_1.csv', help='out file name.')
    parser.add_argument('--output2', default='linear_2.csv', help='out file name.')
    parser.add_argument('--data_png', default='data_png.png', help='out file name.')

    args = parser.parse_args()
    main(args)

# python linear_ols.py --train_data_path ../data/linear_data.csv  --feature V1 --label label --alpha 0.04 --type gls --output1 ../out/linear_estimate.csv --output2 ../out/linear_coef.csv --data_png ../out/png/liner_new.png
