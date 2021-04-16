import os.path
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

rootdir = "E:/bda-pylib/TimeSeriesAnalysis/ARMA"
os.chdir(rootdir)

from ARMA import arma_predict


if __name__=='__main__':
    in_path = "data/daily-minimum-temperatures-in-me.csv"
    col1 = "temperatures"
    p=5
    q=6
    n=30
    type="MA"     #the choose from "AR,MA,ARMA"

    df = pd.read_csv(in_path, index_col=0, parse_dates=True)
    arma_predict(df, col1=col1, p=p, q=q, n=n, type=type)