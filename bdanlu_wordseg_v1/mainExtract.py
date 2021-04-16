import os.path
import warnings
warnings.filterwarnings("ignore")

rootdir = "E:/bda-pylib/bdanlu_wordseg_v1"
os.chdir(rootdir)

from test_bilstmcrf_ws_app import chinese_wordsegment,df_save


if __name__=='__main__':
    input_file = "data/test_data.csv"
    selCol = 'sentence'

    data = chinese_wordsegment(input_file, selCol)
    df_save(data)


