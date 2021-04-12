# -*- coding: utf-8 -*-
# ! /usr/bin/python


import argparse
from scipy import stats
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)



if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='Tukey', help='Sidak,Bonferroni,SNK,Duncan,holm-sidak,'
                                'holm,simes-hochberg,hommel,fdr_bh,fdr_by,fdr_tsbh,fdr_tsbky,Tukey')
    parser.add_argument('--input', required=True, help='input: input file name.')
    parser.add_argument('--col1', required=True, help='input: columns group name.')
    parser.add_argument('--col2', required=True, help='input: columns data name.')
    parser.add_argument('--output', required=True, default='MultiComparison.csv', help='out file name.')

    args = parser.parse_args()
    method = args.method
    in_path = args.input
    # output = args.output
    col1 = args.col1
    col2 = args.col2

    df = pd.read_csv(in_path)
    df[col2] = df[col2].astype("float64")
    # print(df.head(3))

    multiComp = MultiComparison(df[col2], df[col1])

    if method == 'Tukey':
        print(multiComp.tukeyhsd().summary())
        result = multiComp.tukeyhsd().summary()
        resultdf = pd.DataFrame(result)
        resultdf.to_csv(args.output,header=None,index=False)


    else:
        print(multiComp.allpairtest(stats.ttest_rel, method=method)[-1])
        result = multiComp.allpairtest(stats.ttest_rel, method=method)[-1]
        resultdf= pd.DataFrame(result)
        resultdf.to_csv(args.output,index=False)






#python MultiComparison.py --method "Tukey" --input ../../data/sample.csv --col1 Treatment --col2 "StressReduction"  --output ../../out/MultiComparison.csv