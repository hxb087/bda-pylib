# -*- coding: utf-8 -*-
#! /usr/bin/python

import os.path
import numpy as np

rootdir = "E:/bda-pylib/AssociationAnalysis/AprioriModel"
os.chdir(rootdir)

from Apriori import aprioriModel


if __name__=='__main__':
    data =  np.loadtxt("data/ShoppingData.txt", dtype=str)
    min_supp = 0.5
    min_conf = 0.2
    aprioriModel(data,min_supp,min_conf)

#Support. This says how popular an itemset is, as measured by the proportion of transactions in which an itemset appears.
#Confidence. This says how likely item Y is purchased when item X is purchased, expressed as {X -> Y}.