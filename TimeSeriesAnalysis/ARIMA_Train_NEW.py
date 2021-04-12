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

#add  BIC for p,q


def plot(data,data_png):
    plt.figure(facecolor='white', figsize=(10, 5))
    plt.plot(data.index, data)
    plt.savefig(data_png, format='png')

def plotDiff(data_diff,data_png):
    plt.figure(facecolor='white', figsize=(10, 5))
    plt.plot(data_diff)
    plt.savefig(data_png, format='png')


def stabilityTest(data_diff,data_png): # 单位根检验，确定数据为平稳时间序列


    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    fig, axes = plt.subplots(2, 1)
    plot_acf(data_diff, ax=axes[0])
    plot_pacf(data_diff,  ax=axes[1])
    plt.tight_layout()
    plt.savefig(data_png, format='png')

    return


def arimaModel(data,select_ic,p,d,q,output,data_png,model_path):


    if(select_ic):
        print("aa",select_ic)
        import statsmodels.tsa.stattools as st
        model = st.arma_order_select_ic(data_diff, max_ar=3, max_ma=3, ic=['aic', 'bic', 'hqic'])
        p,q=model.bic_min_order #返回一个元组，分别为p值和q值
        print("the best p,q:", p,q)
# 当使用data原始数据时，拟合ARIMA模型
    from statsmodels.tsa.arima_model import ARIMA
    model = ARIMA(data, order=(p,d,q))
    result = model.fit()
    print("1....")

    f = open(model_path, 'wb')
    dill.dump(result, f)
    f.close()


    f = open(model_path, 'rb')
    param_dict = dill.load(f)
    pred = param_dict.predict(start=1, end=len(data) + 10)
    print(pred)

    # pred = result.predict(start=1, end =len(data) + 10 ) # 从训练集第0个开始预测(start=1表示从第0个开始)，预测完整个训练集后，还需要向后预测10个

# 将预测的平稳值还原为非平稳序列
    result_fina = np.array(pred[0:-10]) + (np.array(data.shift(1)))

    ds = np.cumsum(pred[-10:]) + result_fina[-1:]
    fina = np.concatenate((result_fina, np.array(ds)), axis=0)

    import pandas as pd
    finadf = {
        "dt":np.array(data.index)[1:],"y":np.array(data)[1:],"yhat": fina[1:-10]}
    finadf = pd.DataFrame(finadf)
    print(finadf.head(5))
    finadf.to_csv(output,encoding='utf_8_sig')



    import matplotlib.pyplot as plt
    plt.plot(fina[-110:], label='Pre value')
    plt.plot(np.array(data)[-100:], label='True value')
    plt.grid(True)  ##增加格点
    plt.axis('tight')  # 坐标轴适应数据量 axis 设置坐标轴
    plt.legend()
    plt.savefig(data_png, format='png')

    return

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=True, help='input: input file name.')
    parser.add_argument('--col1', required=True, help='input: columns date.')
    parser.add_argument('--col2', required=True, help='input: columns value.')
    parser.add_argument('--output',  default='predict_data.csv', help='out file name.')
    parser.add_argument('--method', default='1', help='method: Can be 1, 2 ,3 or 4, '
                        '1 for figure of time series data, '
                        '2 for figure of autocorrelation and partial autocorelation '
                        '3 for diff of data  '
                        '4 train the ARIMA model and predict future points given begin time and end time.')

    parser.add_argument('--select_ic',type=bool, default=False, help='plot of png ')
    parser.add_argument('--d', type=str, default='1',  help='arguement of the ARIMA model: the degree of differencing')
    parser.add_argument('--p',type=str,default='1',
                        help='arguement of the ARIMA model: the order (number of time lags) of the autoregressive model')
    parser.add_argument('--q',type=str,default='1', help='arguement of the ARIMA model: the order of the moving-average model')

    parser.add_argument('--model', type=str, default="../out/ARIMA_model", help='plot of png ')
    parser.add_argument('--data_png', type=str, default="../out/png/plot.png", help='plot of png ')

    args = parser.parse_args()
    in_path = args.input
    output = args.output
    col1 = args.col1
    col2 = args.col2
    method = args.method
    select_ic=args.select_ic
    model_path=args.model

    d = int(args.d)
    p = int(args.p)
    q = int(args.q)

#read data

    df = pd.DataFrame()
    files = os.listdir(in_path)

    for fn in files:
        # print(fn)
        if fn.startswith("part-"):
            # print(fn)
            filename = in_path + '/' + fn
            df1 = pd.read_csv(filename)
            # df=df1
            df = pd.concat([df1, df], axis=0)
            # print(df1.head(2))
            # df = df.append(df1)
            # print(df.shape)
    df.drop_duplicates(inplace=True)
    df.set_index('id', inplace=True)
    df.sort_index(inplace=True)
    # df.to_csv("aa1.csv")
    df.reset_index(inplace=True)
    # print(df.shape)

    print(df.head(5))
    print(df.shape)


    # df = pd.read_csv(in_path, encoding='utf-8', index_col=0,parse_dates=True)
    print(col1)
    df = df.set_index(col1)
    data = df[col2]
    data_diff = data.diff()
    data_diff = data_diff.dropna()

    if method == '1':
        plot(data,args.data_png)
    elif method == '2':
        plotDiff(data_diff, args.data_png)
    elif method == '3':
        stabilityTest(data_diff, args.data_png)
    elif method == '4':
        arimaModel(data,select_ic, p,d,q,output,args.data_png,model_path)




#python -W ignore ARIMA.py --input ../data/ChinaBank.csv --col1 Date --col2 Close --out ../out/predict_data.csv

#python -W ignore ARIMA_Train_NEW.py --input ../data/prophet_predict --col1 rectime --col2 averageWindSpeed --output ../out/bb.csv --method 4 --model ../out/ARIMA_model
