import os.path

rootdir = "E:/bda-pylib/SignificantVariableSelectionMethod/Elastic"
os.chdir(rootdir)

from linear_Significant import Significant


if __name__=='__main__':
    type = "lasso"
    #elasticNet,lasso,ridge
    train_data_path = "data/linear_data.csv"
    col="V1,V2,V3,V4,V5"
    label="label"
    Significant(train_data_path, type, col, label)
