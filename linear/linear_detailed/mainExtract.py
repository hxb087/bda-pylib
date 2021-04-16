import os.path

rootdir = "E:/bda-pylib/linear/linear_detailed"
os.chdir(rootdir)

from linear_detail import linear_new


if __name__=='__main__':
    type = "ols"
    input = "data/linear_test2D.csv"
    linear_new(type, input)
