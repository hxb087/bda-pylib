import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import numpy as np


def plot_results(predicted_data, true_data,data_png):
    fig = plt.figure(facecolor='white', figsize=(10, 5))
    ax = fig.add_subplot(111)
    ax.plot(predicted_data, label='Prediction')
    plt.plot(true_data, label='True Data')
    plt.legend()
    # plt.show()
    # plt.draw()
    # plt.pause(3)# 间隔的秒数：6s
    # plt.close(fig)
    plt.savefig(data_png, format='png')

def arma_predict(dataset,col1,p,q,n,type,output,data_png):
    data = list(dataset[col1])
    from statsmodels.tsa.arima_model import ARMA
    # """
    #   import statsmodels.tsa.stattools as st
    #   order = st.arma_order_select_ic(data,max_ar=2,max_ma=2,ic=['aic', 'bic', 'hqic'])
    # """
    # print(ifUse)
    # if ifUse:
    #     p,q=p,q
    #     print("With the p,q as you put:  ",p,q)
    # else:
    #     (p, q) = (sm.tsa.arma_order_select_ic(data, max_ar=3, max_ma=3, ic='aic')['aic_min_order'])
    #     print("With the p,q as AUTO of AIC:  ", p, q)

    if type == "AR":
        model = ARMA(data, order=(p, 0))
        result_arma = model.fit(disp=-1, method='css')
    elif type == "MA":
        model = ARMA(data, order=(0, q))
        result_arma = model.fit(disp=-1, method='css')
    elif type == "ARMA":
        model = ARMA(data, order=(p, q))
        result_arma = model.fit(disp=-1, method='css-mle')
    print("Used model is :", type)

    # predict = result_arma.predict(1, len(data)+1000)
    pred = result_arma.forecast(n)[0]
    print(pred)
    predict = np.concatenate((data, pred), axis=0)

    # result_arma


    import pandas as pd
    finadf = {
        "value": predict}
    finadf = pd.DataFrame(finadf)
    print(finadf.tail(15))
    finadf.to_csv(output, encoding='utf_8_sig')

    plot_results(predict[-300-n:], data[-300:],data_png)

    return predict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='input: input file name.')
    parser.add_argument('--col1', required=True, help='input: columns value.')
    parser.add_argument('--p', required=True, type=int,help='p value.')
    parser.add_argument('--q', required=True, type=int,help='q value.')
    parser.add_argument('--type', required=True, help='model')
    # parser.add_argument('--ifUse', default=False,type=bool, help='if use p q')
    parser.add_argument('--output', required=True, default='predict_data.csv', help='out file name.')
    parser.add_argument('--data_png', type=str, default="../out/png/plot.png", help='plot of png ')
    parser.add_argument('--n', required=True, type=int,help='number of prediction')

    args = parser.parse_args()

    in_path = args.input
    output = args.output
    col1 = args.col1
    p = args.p
    q = args.q
    n = args.n
    type = args.type
    data_png= args.data_png


    df = pd.read_csv(in_path,index_col=0, parse_dates=True)
    arma_predict(df,col1=col1,p=p,q=q,n=n,type=type,output=output,data_png= data_png)


#python -W ignore ARMA.py --input ../data/daily-minimum-temperatures-in-me.csv --col1 temperatures --p 1
# --q 2 --type MA --ifUse False --output ../out/pre.csv

#python -W ignore ARMA.py --input ../data/daily-minimum-temperatures-in-me.csv --col1 temperatures --p 5 --q 6 --type ARMA  --output ../out/pre.csv --n 30