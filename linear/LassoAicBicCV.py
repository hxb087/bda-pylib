
# print(__doc__)


import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
import pandas as pd
import argparse




EPSILON = 1e-4

#AIC、BIC model
def plot_AIC_BIC(X,y,png):
    model_bic = LassoLarsIC(criterion='bic')
    t1 = time.time()
    model_bic.fit(X, y)
    t_bic = time.time() - t1
    alpha_bic_ = model_bic.alpha_

    print("Best BIC alpha : %.4f" %alpha_bic_)

    model_aic = LassoLarsIC(criterion='aic')
    model_aic.fit(X, y)
    alpha_aic_ = model_aic.alpha_
    print("Best BIC alpha : %.4f" % alpha_aic_)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    # plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    plt.figure(1,figsize=(12, 8))
    plot_ic_criterion(model_aic, 'AIC', 'b')
    plot_ic_criterion(model_bic, 'BIC', 'r')
    plt.legend()
    plt.title('Information-criterion for model selection (training time %.3fs)' % t_bic)
    # plt.show()
    plt.savefig(png, format='png')
    plt.close('all')



def plot_ic_criterion(model, name, color):
    criterion_ = model.criterion_
    plt.semilogx(model.alphas_ + EPSILON, criterion_, '--', color=color,
                 linewidth=3, label='%s criterion' % name)
    plt.axvline(model.alpha_ + EPSILON, color=color, linewidth=3,
                label='alpha: %s estimate' % name)
    plt.xlabel(r'$\alpha$')
    plt.ylabel('criterion')

#LassoCV
def plot_LassoCV(X,y,cv,png):

    t1 = time.time()
    model = LassoLarsCV(cv=cv).fit(X, y)
    t_lasso_lars_cv = time.time() - t1

    plt.figure(1,figsize=(12, 8))
    plt.semilogx(model.cv_alphas_ + EPSILON, model.mse_path_, ':')
    plt.semilogx(model.cv_alphas_ + EPSILON, model.mse_path_.mean(axis=-1), 'k',
                 label='Average across the folds', linewidth=2)
    plt.axvline(model.alpha_, linestyle='--', color='k',
                label='alpha CV')
    plt.legend()

    plt.xlabel(r'$\alpha$')
    plt.ylabel('Mean square error')
    plt.title('Mean square error on each fold: Lars (train time: %.2fs)'
              % t_lasso_lars_cv)
    plt.axis('tight')
    # plt.show()
    plt.savefig(png, format='png')
    plt.close('all')
    print("Best CV  alpha : %.4f" %model.alpha_)




def main(args):
    df = pd.read_csv(args.train_data_path).dropna()  #'../data/sample_linear_regression_data.csv'
    columns = args.features.split(",")
    X = df[columns].values
    y = df[args.label].values
    png=args.pt_png
    if args.type =="CV":
        print("the method is Lasso CV")
        plot_LassoCV(X, y, args.cv,png)
    else:
        print("the method is Lasso AIC and BIC")
        plot_AIC_BIC(X, y,png)




if __name__ == '__main__':
        parser = argparse.ArgumentParser(description='Lasso_model  program train...')
        parser.add_argument('--train_data_path', type=str, default="", help='the path of test file')
        parser.add_argument('--pt_png', type=str, default="", help='the path of model pt png')
        parser.add_argument('--features', type=str, default="", help='choose columns from test file')
        parser.add_argument('--label', type=str, default="", help='the label')
        parser.add_argument('--type', type=str, default="CV", help='the AIC_BIC or CV')
        parser.add_argument('--cv', type=int, default=10, help='the alpha')

        args = parser.parse_args()
        main(args)

######  with  D:  PATH use
##  python  /bda-pylib/linear/LassoAicBicCV.py --train_data_path /bda-pylib/data/sample_linear_regression_data.csv --pt_png /bda-pylib/out/png/Lasso.png --features V1,V2,V3,V4,V5,V6,V7,V8 --label label --type AIC  --cv 10
