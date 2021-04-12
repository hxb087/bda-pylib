import Levenshtein
import argparse
import numpy as np
import pandas as pd



def main(args):

    df = pd.read_csv(args.input,index_col=None,encoding='utf-8', parse_dates=True)

    str1 = df[args.col1][0]
    str2 = df[args.col2][0]

    Levenshtein.distance(str1,str2)

    print(Levenshtein.distance(str1,str2))

    distance_value =  Levenshtein.distance(str1,str2)
    finadf = {"value": np.array([distance_value])}
    finadf = pd.DataFrame(finadf)

    finadf.to_csv(args.output, encoding='utf_8_sig')




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=True, help='input: input file name.')
    parser.add_argument('--col1', required=True, help='input: columns date.')
    parser.add_argument('--col2', required=True, help='input: columns value.')
    parser.add_argument('--output', required=True, default='predict_data.csv', help='out file name.')
    args = parser.parse_args()
    main(args)

#python editDistance.py --input ../data/editDistance_test.csv --col1 V0 --col2 V1 --output ../out/aa.csv