import pycasso
import pandas as pd
import argparse


def main(args):
    raw_df = pd.read_csv(args.train_data_path)
    columns = args.features.split(",")
    features_values = raw_df[columns].values
    label = args.label

    y = raw_df[label].values

    predict_df = pd.read_csv(args.predict_data_path)
    predict_values = predict_df[columns].values

    s = pycasso.Solver(features_values, y, penalty=args.penalty)
    s.train()
    beta = s.coef()['beta'][-1]
    print("beta :",  beta)

    intercept = s.coef()['intercept'][-1]
    print("intercept :", intercept)

    result = s.predict(predict_values)
    print("result :",    result )

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='linear_model  program train...')
    parser.add_argument('--train_data_path', type=str, default="", help='the path of test file')
    parser.add_argument('--predict_data_path', type=str, default="", help='the path of test file')
    parser.add_argument('--features', type=str, default="", help='choose columns from test file')
    parser.add_argument('--label', type=str, default="", help='the label')
    parser.add_argument('--penalty', type=str, default="l1", help='the penalty "l1", "mcp" and "scad" ')

    args = parser.parse_args()
    main(args)

#python pycassoModel.py --train_data_path data/linear_data.csv --predict_data_path data/linear_data.csv --features V1,V2,V3,V4,V5 --label label --penalty l1