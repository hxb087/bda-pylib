# -*- coding: utf-8 -*-
#! /usr/bin/python

import sys
import argparse
# import numpy as np
# from scipy import stats, linalg

if __name__ == "__main__":
    import pandas as pd
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='pearson', help='pearson,kendall,spearman')
    parser.add_argument('--input', required=True, help='input: input file name.')
    parser.add_argument('--col', required=True, help='input: columns name.')
    parser.add_argument('--output', required=True, default='corr.csv',help='out file name.')
    args = parser.parse_args()
    method =  args.method
    in_path = args.input
    output = args.output
    col=args.col.strip().split(',')
    
    df= pd.read_csv(in_path)
    df=df[col].astype("float64")

    print(df.corr(method=method).head(20))
    # 'pearson', 'kendall', 'spearman'
    df.corr(method=method).to_csv(output,encoding='utf_8_sig')
    

    #python statistics/correlation.py --method pearson --input D:\testdata\242\corr\win.csv --col fixed_acidity,volatile_acidity --output out/correlation.csv
