#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author      : liujunfeng
# @Description : 用于模型的工具方法

import joblib


class ModelUtils(object):
    __col_key = "columns"
    __model_key = "model"

    def __init__(self):
        pass

    @classmethod
    def save_model(cls, columns, model_object, save_path):
        """函数用于存储选中的特阵列和模型文件
        :param columns: 特征列
        :param model_object: 模型对象
        :param save_path: 存储路径
        :return:
        """
        joblib.dump({cls.__col_key: columns, cls.__model_key: model_object}, save_path)

    @classmethod
    def load_model(cls, load_path):
        """给定模型路径，加载模型和可选的特征列"""
        param = joblib.load(load_path)
        return param[cls.__col_key], param[cls.__model_key]
