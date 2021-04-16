# -*- coding: utf-8 -*-
'''
@Time   :   2020/3/1315:46
@Auth   :   zhou
@File   :   test_bilstmcrf_wordsegment_app.py
'''

import time
import argparse
import numpy as np
import pandas as pd
import re
import warnings
warnings.filterwarnings("ignore")

from bilstm_crf import call_bilstm_crf

segmenter = call_bilstm_crf()

def chinese_wordsegment(infile, col='sentence', append=True):
    data = pd.read_csv(infile, sep=',')
    # data['wseg'] = data[col].apply(lambda x: decode_text([x]))
    outCol = col
    if append:
        outCol = col + '_append'

    try:
        data['text_list'] = data[col].apply(lambda x: cut_sent(x))
        data[outCol] = data['text_list'].apply(lambda x: segmenter.decode_text(x))

        del data['text_list']
    except IOError as e:
        print(e)

    return data

def df_save(data, save_file='out/result.csv'):
    data.to_csv(save_file, encoding='utf_8_sig', index=None, header=True)
    return

def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--input_file', help='input data, (csv)', default='test_data.csv', type=str)
#     parser.add_argument('--selCol', help='text', default='sentence', type=str)
#     parser.add_argument('--append', help='是否追加一列结果数据', default='True', type=bool)
#     parser.add_argument('--output_file', help='output file', default='result.csv', type=str)
#
#     args = parser.parse_args()
#
#     data = chinese_wordsegment(args.input_file, args.selCol, args.append)
#     df_save(data, save_file=args.output_file)