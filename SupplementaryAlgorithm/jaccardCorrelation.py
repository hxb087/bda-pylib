import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

def main(args):

    df = pd.read_csv(args.input,index_col=None,encoding='utf-8', parse_dates=True)
    print(df)

    b = df[args.col1].values
    c = df[args.col2].values

    X = np.vstack([b, c])
    d2 = pdist(X, 'jaccard')
    # print(d2)

    finadf = {"value": d2}
    finadf = pd.DataFrame(finadf)

    finadf.to_csv(args.output, encoding='utf_8_sig')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=False, help='input: input file name.')
    parser.add_argument('--col1', required=False, help='input: columns date.')
    parser.add_argument('--col2', required=False, help='input: columns value.')
    parser.add_argument('--output', required=False, default='predict_data.csv', help='out file name.')
    args = parser.parse_args()
    main(args)


    #python jaccardCorrelation.py --input ../data/jaccardCorrelation.csv --col1 label --col2 V0 --output ../out/aa.csv
