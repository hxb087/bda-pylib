import numpy as np
import argparse
import pandas as pd
from dtw import *

def main(args):
    ## A noisy sine wave as query
#     idx = np.linspace(0,6.28,num=100)
#     query = np.sin(idx) + np.random.uniform(size=100)/10.0
#
# ## A cosine is for template; sin and cos are offset by 25 samples
#     template = np.cos(idx)


    df = pd.read_csv(args.input, encoding='utf-8', parse_dates=True)
    # print(df)

    query=np.array(df[args.col1])
    template=np.array(df[args.col2])


## Find the best match with the canonical recursion formula

    alignment = dtw(query, template, keep_internals=True)

## Display the warping curve, i.e. the alignment curve
    alignment.plot(type="threeway")



## Align and plot with the Rabiner-Juang type VI-c unsmoothed recursion
    dtw(query, template, keep_internals=True,step_pattern=rabinerJuangStepPattern(6, "c")).plot(type="twoway",offset=-2)

## See the recursion relation, as formula and diagram
    print(rabinerJuangStepPattern(6,"c"))
    rabinerJuangStepPattern(6,"c").plot()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='input: input file name.')
    parser.add_argument('--col1', required=True, help='input: columns date.')
    parser.add_argument('--col2', required=True, help='input: columns value.')
    parser.add_argument('--output', required=True, default='predict_data.csv', help='out file name.')
    args = parser.parse_args()

    main(args)