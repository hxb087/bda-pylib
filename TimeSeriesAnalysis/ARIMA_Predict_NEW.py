import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import argparse
warnings.filterwarnings("ignore")
import dill
import os


def main(args):
    np.random.seed(0)

    import pandas as pd

    df = pd.DataFrame()
    files = os.listdir(args.test_data_path)

    for fn in files:
        if fn.startswith("part-"):
            filename = args.test_data_path + '/' + fn
            df1 = pd.read_csv(filename)
            df = pd.concat([df1, df], axis=0)
    df.drop_duplicates(inplace=True)
    df.set_index('id', inplace=True)
    df.sort_index(inplace=True)
    df.reset_index(inplace=True)


    df = df.set_index(args.col1)
    data = df[args.col2]

    f = open(args.model_path, 'rb')
    param_dict = dill.load(f)
    pred = param_dict.predict(start=1, end=len(data)+10)
    result_fina = np.array(pred[0:-10]) + (np.array(data.shift(1)))

    ds = np.cumsum(pred[-10:]) + result_fina[-1:]
    fina = np.concatenate((result_fina, np.array(ds)), axis=0)

    import pandas as pd
    finadf = {
        "dt":np.array(data.index)[1:],"yhat": fina[1:-10]}
    finadf = pd.DataFrame(finadf)
    print(finadf.head(5))
    finadf.to_csv(args.out_path, encoding='utf_8_sig')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_data_path', type=str, default="", help='the path of training file')
    parser.add_argument('--col1', required=True, help='input: columns date.')
    parser.add_argument('--col2', required=True, help='input: columns value.')
    parser.add_argument('--model_path', type=str, default="", help='the path of model file')
    parser.add_argument('--out_path', type=str, default="aa", help='the path of out file')

    args = parser.parse_args()

    main(args)

#python -W ignore ARIMA_Predict_NEW.py --test_data_path ../data/prophet_predict --col1 rectime --col2 averageWindSpeed --model_path ../out/ARIMA_model