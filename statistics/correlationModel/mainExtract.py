# -*- coding: utf-8 -*-
#! /usr/bin/python

import os.path

rootdir = "E:/bda-pylib/statistics/correlationModel"
os.chdir(rootdir)

from correlation import correlation


if __name__=='__main__':
    col = "V0,V1,V2,V3"
    input = "data/iris.csv"
    method = 'pearson' # 'pearson', 'kendall', 'spearman'
    correlation(col,input,method)
