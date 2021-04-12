import pycasso
import pandas as pd
import argparse
import numpy as np


def main(args):
    raw_df = pd.read_csv(args.train_data_path)
    columns = args.features.split(",")
    features_values = raw_df[columns].values
    label = args.label

    y = raw_df[label].values

    # predict_df = pd.read_csv(args.predict_data_path)
    # predict_values = predict_df[columns].values

    s = pycasso.Solver(features_values, y, penalty=args.penalty)
    s.train()
    beta = s.coef()['beta'][-1]
    print("beta :",  beta)

    intercept = s.coef()['intercept'][-1]
    print("intercept :", intercept)

    result = s.predict(features_values)
    print("result :",    result )

    select_df = pd.DataFrame(np.array(beta).reshape(1, -1), columns=columns)
    select_df.to_csv(args.selection_path, index=None, sep=",")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='linear_model  program train...')
    parser.add_argument('--train_data_path', type=str, default="", help='the path of test file')
    parser.add_argument('--selection_path', type=str, default="", help='the path of test file')
    parser.add_argument('--features', type=str, default="", help='choose columns from test file')
    parser.add_argument('--label', type=str, default="", help='the label')
    parser.add_argument('--penalty', type=str, default="l1", help='the penalty "l1", "mcp" and "scad" ')

    args = parser.parse_args()
    main(args)

#python pycassoModel.py --train_data_path data/linear_data.csv --selection_path out/linear_select.csv --features V1,V2,V3,V4,V5 --label label --penalty l1