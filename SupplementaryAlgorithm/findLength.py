import argparse
import numpy as np
import pandas as pd


def main(args):
    df = pd.read_csv(args.input, index_col=None, encoding='utf-8', parse_dates=True)
    b = df[args.col1][0]
    c = df[args.col2][0]
    # b = "acbcccdaaada"
    # c = "acbcccdaaad"
    print(findLength(b, c))
    findLength_value = findLength(b, c)
    finadf = {"findLength": np.array([findLength_value])}
    finadf = pd.DataFrame(finadf)
    finadf.to_csv(args.output, encoding='utf_8_sig')


def findLength(A, B):
    m, n = len(A), len(B)
    ans = 0
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if A[i - 1] == B[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                ans = max(ans, dp[i][j])
    return ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=True, help='input: input file name.')
    parser.add_argument('--col1', required=True, help='input: columns date.')
    parser.add_argument('--col2', required=True, help='input: columns value.')
    parser.add_argument('--output', required=True, default='predict_data.csv', help='out file name.')
    args = parser.parse_args()
    main(args)

# python findLength.py --input ../data/editDistance_test.csv --col1 V0 --col2 V1 --output ../out/aa.csv
