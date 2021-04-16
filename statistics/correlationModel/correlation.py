# -*- coding: utf-8 -*-
#! /usr/bin/python

import pandas as pd

def correlation(col,input,method='pearson',output="out/restult.csv"):
    col = col.strip().split(',')

    df = pd.read_csv(input)
    df = df[col].astype("float64")

    print(df.corr(method).head(20))
    # 'pearson', 'kendall', 'spearman'
    df.corr(method).to_csv(output, encoding='utf_8_sig')

