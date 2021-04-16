# -*- coding: utf-8 -*-
# ==============================
# @Time   :   2020/7/1 11:20
# @Auth   :   zhou
# @File   :   demo
# ==============================

from bilstm_crf import call_bilstm_crf
segmenter = call_bilstm_crf()


sentence = '采用的数据集包含三类标签：背景、肝脏、肝肿瘤。肿瘤附着在肝脏上，体积很小。所以，直接按照三类来进行训练，会导致肿瘤分割效果较差，这个可以通过实验结果验证。所以对此类问题我们一般都是先分割或检测肝脏得到肝脏ROI，然后在此ROI内完成肿瘤分割。'

data = []
data.append(sentence)
output = segmenter.decode_text(data)

print(output)