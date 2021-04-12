#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author      : liujunfeng
# @Description : 用于dataframe的工具方法

import pandas as pd


class DFUtils(object):
    __default_prediction_name = "prediction"

    def __init__(self):
        pass

    @classmethod
    def save(cls, origin_df, values, save_path, col_name=__default_prediction_name):
        """
        函数主要用于模型预测后保存csv，原始测试df加上预测列值
        :param origin_df: 原始df
        :param values: 新增列 列值
        :param save_path: df保存路径
        :param col_name: 新增列列名，默认为: __default_prediction_name
        :return:
        """
        origin_df[col_name] = values
        origin_df.to_csv(save_path, index=None, sep=",")
