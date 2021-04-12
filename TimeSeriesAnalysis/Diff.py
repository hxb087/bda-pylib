import pandas as pd
import warnings
import argparse
warnings.filterwarnings("ignore")





if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=True, help='input: input file name.')
    parser.add_argument('--col1', required=True, help='input: columns value.')
    parser.add_argument('--output', required=True, default='predict_data.csv', help='out file name.')
    parser.add_argument('--diffNum', required=True, default=1, type=int, help='out file name.')

    args = parser.parse_args()
    in_path = args.input
    output = args.output
    col1 = args.col1
    diffNum= args.diffNum

    df = pd.read_csv(in_path, encoding='utf-8', index_col=0,parse_dates=True)
    data = df[col1]
    data_diff = data.diff(diffNum)
    print(data_diff.head(10))

    finadf = {
        "diffValue": data_diff}
    finadf = pd.DataFrame(finadf)
    print(finadf.head(5))
    finadf.to_csv(output, encoding='utf_8_sig')


#python -W ignore Diff.py --input ../data/ChinaBank.csv --col1 Close --out ../out/data_diff.csv --diffNum 2
