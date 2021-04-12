#!/usr/bin/env python
# -*- coding: utf-8 -*-



import argparse
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

def main(args):

    np.random.seed(9876789)
    df = pd.read_csv(args.train_data_path)
    # df['treated'] = df['treated'].map(lambda a: 1 if (a == "NJ") else 0).astype(float)
    df['treated'] = df['treated'].astype(float)
    df['t'] = df['t'].astype(float)
    df['bk'] = df['bk'].astype(float)
    df['did12'] = df['t'] * df['treated']
    df['did13'] = df['t'] * df['bk']
    df['did23'] = df['treated'] * df['bk']
    df['ddd'] = df['t'] * df['treated'] * df['bk']
    results = smf.ols('fte ~ t + treated + bk + did12 + did13 + did23 + ddd', data=df).fit()
    print("**********************************************************************************\n")
    print(results.summary())
    alpha = 0.05

    data_t = {"coef": results.params, "std err": results.bse, "t": results.tvalues, "P>|t|": results.pvalues,
              "[" + str(alpha / 2.0): results.conf_int(alpha)[0],
              str(1 - alpha / 2.0) + "]": results.conf_int(alpha)[1]}

    sdata_df = pd.DataFrame(data_t)
    print(sdata_df)
    sdata_df.to_csv(args.output)


    return


if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='linear_model  program train...')
        parser.add_argument('--train_data_path', type=str, default="", help='the path of test file')
        parser.add_argument('--output', type=str, default="aa.csv", help='the path of test file')


        args = parser.parse_args()
        main(args)

#python  DDDModel.py --train_data_path ../../data/CardKrueger1994.csv --output ../../out/ddd.csv