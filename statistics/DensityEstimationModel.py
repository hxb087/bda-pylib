# encoding: UTF-8
from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd
import argparse

def main(args):
    df = pd.read_csv(args.train_data_path)

    columns = args.features.split(",")
    features_df = df[columns]
    X = features_df.values

    kde = KernelDensity(kernel=args.kernel, bandwidth=0.2).fit(X)
    core=    kde.score_samples(X)
    print(f"the {args.kernel} of KernelDensity: ", core)

    datadf = {"core": core }

    core_df = pd.DataFrame(datadf)
    print(core_df)

    core_df.to_csv(args.output, index=None, sep=",")


    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dbscan program training...')
    parser.add_argument('--train_data_path', type=str, default="", help='the path of training file')
    parser.add_argument('--features', type=str, default="", help='choose columns from test file')
    parser.add_argument('--kernel', type=str, default="", help='the path of model file')
    parser.add_argument('--output', default='linear_core.csv', help='out file name.')

    args = parser.parse_args()
    main(args)


#python bda-pylib/statistics/DensityEstimationModel.py --train_data_path bda-pylib/data/linear_data.csv --features V4,V5 --kernel cosine --output bda-pylib/out/core.csv