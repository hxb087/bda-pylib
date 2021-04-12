import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
import argparse

warnings.filterwarnings("ignore")


def plot(data, data_png):
    plt.figure(facecolor='white', figsize=(10, 5))
    plt.plot(data.index, data)
    plt.savefig(data_png, format='png')


def plotDiff(data_diff, data_png):
    plt.figure(facecolor='white', figsize=(10, 5))
    plt.plot(data_diff)
    plt.savefig(data_png, format='png')


def stabilityTest(data_diff, data_png):  # 单位根检验，确定数据为平稳时间序列

    # from statsmodels.tsa.stattools import adfuller
    # print(adfuller(data_diff))
    # from statsmodels.stats.diagnostic import acorr_ljungbox
    # print(acorr_ljungbox(data_diff, lags = 10))# 第一个数：统计值； 第二个数：p值

    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, axes = plt.subplots(2, 1)
    plot_acf(data_diff, ax=axes[0])
    plot_pacf(data_diff, ax=axes[1])
    plt.tight_layout()
    plt.savefig(data_png, format='png')

    return


def arimaModel(data, p, d, q, n, output, data_png):
    # import statsmodels.tsa.stattools as st
    # model = st.arma_order_select_ic(data_diff, max_ar=3, max_ma=3, ic=['aic', 'bic', 'hqic'])
    # p,q=model.bic_min_order #返回一个元组，分别为p值和q值

    # 当使用data原始数据时，拟合ARIMA模型
    from statsmodels.tsa.arima_model import ARIMA
    model = ARIMA(data, order=(p, d, q))
    result = model.fit()
    # pred = result.predict(start=1, end =len(data) + n ) # 从训练集第0个开始预测(start=1表示从第0个开始)，预测完整个训练集后，还需要向后预测10个
    pred = result.forecast(n)[0]

    # 将预测的平稳值还原为非平稳序列
    #     result_fina = np.array(pred1) + (np.array(data.shift(d)[d:]))
    # ds = np.cumsum(pred) + result_fina[-1:]
    fina = np.concatenate((data.values, pred), axis=0)

    import pandas as pd
    finadf = {"value": fina}
    finadf = pd.DataFrame(finadf)
    finadf.to_csv(output, encoding='utf_8_sig')

    import matplotlib.pyplot as plt
    plt.figure(facecolor='white', figsize=(10, 5))
    plt.plot(fina, label='Pre value')
    plt.plot(np.array(data), label='True value')
    plt.grid(True)
    plt.axis('tight')  # 坐标轴适应数据量 axis 设置坐标轴
    plt.legend()
    plt.savefig(data_png, format='png')
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=True, help='input: input file name.')
    parser.add_argument('--col1', required=True, help='input: columns date.')
    parser.add_argument('--col2', required=True, help='input: columns value.')
    parser.add_argument('--output', required=True, default='predict_data.csv', help='out file name.')
    parser.add_argument('--method', default='1', help='method: Can be 1, 2 ,3 or 4, '
                                                      '1 for figure of time series data, '
                                                      '2 for figure of autocorrelation and partial autocorelation '
                                                      '3 for diff of data  '
                                                      '4 train the ARIMA model and predict future points given begin time and end time.')
    parser.add_argument('--d', help='arguement of the ARIMA model: the degree of differencing')
    parser.add_argument('--p',
                        help='arguement of the ARIMA model: the order (number of time lags) of the autoregressive model')
    parser.add_argument('--q', help='arguement of the ARIMA model: the order of the moving-average model')

    parser.add_argument('--data_png', type=str, default="../out/png/plot.png", help='plot of png ')

    parser.add_argument('--n', help='number of prediction')

    args = parser.parse_args()
    in_path = args.input
    output = args.output
    col1 = args.col1
    col2 = args.col2
    method = args.method

    d = int(args.d)
    p = int(args.p)
    q = int(args.q)
    n = int(args.n)

    df = pd.read_csv(in_path, encoding='utf-8', index_col=0, parse_dates=True)
    df = df.set_index(col1)
    data = df[col2]
    data_diff = data.diff()
    data_diff = data_diff.dropna()

    if method == '1':
        plot(data, args.data_png)
    elif method == '2':
        plotDiff(data_diff, args.data_png)
    elif method == '3':
        stabilityTest(data_diff, args.data_png)
    elif method == '4':
        arimaModel(data, p, d, q, n, output, args.data_png)


# python -W ignore ARIMA.py --input ../data/ChinaBank.csv --col1 Date --col2 Close --out ../out/predict_data.csv
# python -W ignore ARIMA.py --input ../data/ChinaBank.csv --col1 Date --col2 Close --out ../out/predict_data.csv --d 1  --p 2 --q 3  --method 4 --n 20