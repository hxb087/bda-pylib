#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author      : liujunfeng
# @Description : dbscan聚类算子训练

import numpy as np

import pandas as pd
import joblib
import argparse
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


def main(args):
    # read data
    raw_df = pd.read_csv(args.train_data_path)
    columns = args.features.split(",")
    features_df = raw_df[columns]
    train_data = StandardScaler().fit_transform(features_df.values)

    # Compute DBSCAN
    db = DBSCAN(eps=args.eps, min_samples=args.min_samples, n_jobs=args.n_jobs).fit(train_data)

    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('聚类中心点数量: %d' % n_clusters_)
    print('聚类噪音数据数量: %d' % n_noise_)
    # print("轮廓系数: %0.3f" % metrics.silhouette_score(train_data, labels))

    joblib.dump({"columns": columns, "model": db}, args.model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dbscan program training...')
    parser.add_argument('--train_data_path', type=str, default="", help='the path of training file')
    parser.add_argument('--eps', type=float, default=0.3,
                        help="The maximum distance between two samples for one to be considered as "
                             "in the neighborhood of the other")
    parser.add_argument('--features', type=str, default="", help='choose columns from test file')
    parser.add_argument('--min_samples', type=int, default=5, help='the number of neighborhood samples')
    parser.add_argument('--n_jobs', type=int, default=-1, help='The number of parallel jobs to run')
    parser.add_argument('--model_path', type=str, default="", help='the path of model file')

    args = parser.parse_args()
    main(args)
#python dbscan_train.py --train_data_path ../data/dbscan_sample.csv --eps 0.5 --features V0,V1 --min_samples
 # 5 --n_jobs -1 --model_path ../out/model/dbscan_model
