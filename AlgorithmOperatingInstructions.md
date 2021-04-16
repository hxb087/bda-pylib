## python 算子操作：

### 1、相关度分析:

#### 命令行操作:
python bda-pylib/statistics/correlation.py --method pearson --input D:\testdata\242\corr\win.csv --col fixed_acidity,volatile_acidity --output bda-pylib/out/correlation.csv

#### 参数说明：

- method: 相关方式可选择 pearson,kendall,spearman
- input：文件输入，csv文件格式
- col：特征列
- output：文件输出，csv文件格式

### 2、PrefixSpan:

#### 命令行操作:
python bda-pylib/statistics/prefixspan/prefixspan-cli frequent 2 test.txt --file bda-pylib/statistics/prefixspan/test.txt --optype frequentaaa --threshold 2 --minlen 2 --maxlen 10

#### 参数说明：

- file：模型训练数据，CSV文件格式，模型输入文件。  
- optype：模型结果输出，CSV文件格式，数据结果文件输出。
- threshold： 数据项集出现频率阈值
- minlen：最短的数据项数
- maxlen：最长的数据项数

### 3、线性回归详细

#### 命令行操作:

python bda-pylib/linear/linear_ols.py --train_data_path bda-pylib/data/linear_data.csv  --feature V1 --label label --alpha 0.04 --type gls --output1 bda-pylib/out/linear_estimate.csv --output2 bda-pylib/out/linear_coef.csv --data_png bda-pylib/out/png/liner_new.png

#### 参数说明：

- train_data_path: 模型训练数据，CSV文件格式，模型输入文件。
- feature: 用于拟合的特征项
- label: 用于拟合的标签项
- alpha: t检验的置信水平
- type： ols（简单最小二乘法）、gls（广义最小二乘）、wls（加权最小二乘法）、glsar(广义最小二乘ar)
- output1：模型结果评估输出，CSV文件格式，数据结果的评估文件。
- output2：模型结果系数评估，CSV文件格式，数据系数的评估。
- data_png：模型结果拟合图片，png文件格式，一元或者二元线性拟合的图。


### 4、逻辑回归

#### 命令行操作:
python bda-pylib/classification/lr_train.py --train_data_path  bda-pylib/data/breast-cancer-wisconsin.csv 
--features ClumpThickness,UniformityOfCellSize,UniformityOfCellShape,MarginalAdhesion
 --label Class --model_path  bda-pylib/out/model/lr_Model

#### 参数说明：
- train_data_path：模型训练数据，CSV文件格式，模型输入文件。
- features：训练数据的标签列。
- label：特征列。
- model_path：模型结果输出，二进制文件 训练好的模型文件。


### 5、逻辑回归预测

#### 命令行操作:
python bda-pylib/classification/lr_predict.py --test_data_path  bda-pylib/data/breast-cancer-wisconsin.csv  --model_path  bda-pylib/out/model/lr_Model --output_path bda-pylib/out/lr_pre.csv

#### 参数说明：
- test_data_path 模型测试数据，CSV文件格式，模型输入文件。
- model_path 二进制文件 训练好的模型文件。
- output_path 模型结果输出，CSV文件格式，数据结果文件输出。

### 6、核密度估计

#### 命令行操作:
python bda-pylib/statistics/DensityEstimationModel.py --train_data_path bda-pylib/data/linear_data.csv --features V4,V5 --kernel cosine --output bda-pylib/out/core.csv


#### 参数说明：
- train_data_path 模型训练数据，CSV文件格式，模型输入文件。
- features 特征选择
- kernel 核密度估计的核函数，可选择gaussian、tophat、epanechnikov、exponential、linea、cosine几种核函数。
- output 模型结果输出，CSV文件格式，数据结果文件输出。


### 7、巴特利特球度检验

#### 命令行操作:
python bda-pylib/statistics/bartlett_analysis.py --input bda-pylib/data/linear_data.csv --cols V3,V4,V5,label  --output bda-pylib/out/bartlett.csv

#### 参数说明：
- input 模型训练数据，CSV文件格式，模型输入文件。
- cols  特征选择。
- output 模型结果输出，CSV文件格式，数据结果文件输出。

### 8、KMO检验

#### 命令行操作:
python bda-pylib/statistics/kmo_analysis.py --input bda-pylib/data/linear_data.csv --cols V1,V2,V3,V4,V5 --output aa.csv

#### 参数说明：
- input 模型训练数据，CSV文件格式，模型输入文件。
- output 模型结果输出，CSV文件格式，数据结果文件输出。
- cols 特征选择

### 9、DBSCAN

#### 命令行操作:
python bda-pylib/cluster/dbscan_train.py --train_data_path bda-pylib/data/dbscan_sample.csv --eps 0.5 --features V0,V1 --min_samples 5 --n_jobs -1 --model_path bda-pylib/out/model/dbscan_model


#### 参数说明：
- train_data_path 模型训练数据，CSV文件格式，模型输入文件。
- eps 对于一个样本，两个样本之间的最大距离
- features 特征选择
- min_samples 最小的样本数
- n_jobs 同时运行的线程数
- model_path 二进制文件 训练好的模型文件

### 10、DBSCAN预测

#### 命令行操作:

python bda-pylib/cluster/dbscan_predict.py --test_data_path bda-pylib/data/dbscan_sample.csv --model_path bda-pylib/out/model/dbscan_model  --output_path bda-pylib/out/dbscan_sample.csv

#### 参数说明：
- test_data_path 模型训练数据，CSV文件格式 待分析测试数据。
- model_path  二进制文件 输入模型文件 
- output_path CSV文件格式 模型的结果文件。


### 11、One-Class-SVM

#### 命令行操作:
python bda-pylib/classification/one_class_svm_train.py --train_data_path bda-pylib/data/one_class_svm_sample.csv --nu 0.1 --features V0,V1 --label label --kernel rbf --gamma 0.1 --model_path bda-pylib/out/model/one_class_svm_model

#### 参数说明：
- train_data_path 模型训练数据，CSV文件格式，模型输入文件。
- nu 训练的最大误差
- features 特征列
- label 标签列
- kernel 核函数  可选择包括 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
- gamma 核函数的系数
- model_path 二进制文件 训练好的模型文件

### 12、One-Class-SVM预测

#### 命令行操作:

python bda-pylib/classification/one_class_svm_predict.py --test_data_path bda-pylib/data/one_class_svm_sample.csv  --model_path bda-pylib/out/model/one_class_svm_model --output_path bda-pylib/out/one_class_svm.csv

#### 参数说明：
- test_data_path 模型训练数据，CSV文件格式，模型输入文件。
- model_path 二进制文件 输入模型文件 
- output_path 模型结果输出，CSV文件格式 模型的结果文件。

### 13、 LOF

#### 命令行操作:
python bda-pylib/neighbors/lof_train.py --train_data_path  bda-pylib/data/lof_sample.csv --n_neighbors 20 --features V0,V1 --label label --n_jobs -1 --model_path bda-pylib/out/model/lof_model

#### 参数说明：
- train_data_path 模型训练数据，CSV文件格式，模型输入文件。
- n_neighbors 用于查询的邻居数目，默认值20
- features 特征列
- label 标签列
- n_jobs 同时运行的线程数
- model_path 二进制文件 训练好的模型文件

### 14、LOF预测

#### 命令行操作:
python bda-pylib/neighbors/lof_predict.py --test_data_path  bda-pylib/data/lof_sample.csv  --model_path bda-pylib/out/model/lof_model --output_path bda-pylib/
out/lof_data.csv

#### 参数说明：
- test_data_path 模型测试数据，CSV文件格式，模型输入文件。
- model_path 二进制文件 训练好的模型文件
- output_path CSV文件格式 模型的结果文件  

### 15、双重差分

#### 命令行操作:
python bda-pylib/statistics/DifferentialAnalysis/DIDModel.py --train_data_path  bda-pylib/data/CardKrueger1994.csv  --output bda-pylib/out/did.csv


#### 参数说明：
- train_data_path 模型训练数据，CSV文件格式，模型输入文件。
- output CSV文件格式 模型的结果文件

### 16、三重差分

#### 命令行操作:
python bda-pylib/statistics/DifferentialAnalysis/DDDModel.py --train_data_path  bda-pylib/data/CardKrueger1994.csv  --output bda-pylib/out/ddd.csv

#### 参数说明：
- train_data_path 模型训练数据，CSV文件格式，模型输入文件。 
- output CSV文件格式 模型的结果文件

### 17、变点检测CUSUM

#### 命令行操作:
python  bda-pylib/ChangePointDetection/cusumdect.py --train_data_path bda-pylib/data/bar5rb8888.csv --feature close --data_png bda-pylib/out/png/cusum.png --o
utput bda-pylib/out/cusum.csv

#### 参数说明：
- train_data_path 模型训练数据，CSV文件格式，模型输入文件。
- feature 特征列
- data_png 变点检测的图
- output 模型结果数据，CSV文件格式，模型的数据结果

### 18、弹性网族回归

#### 命令行操作:
python bda-pylib/linear/linear_Significant.py --train_data_path bda-pylib/data/linear_data.csv --selection_path bda-pylib/out/select_linear.csv --features V1,V2,V3,V4,V5 --label label --alpha 0.1 --fitIntercept True --l1_ratio 0.5  --max_iter 1000 --normalize False --tol 0.0001 --type elasticNet

#### 参数说明：
- train_data_path 模型训练数据，CSV文件格式，模型输入文件。 
- selection_path 模型拟合系数。
- features 特征列  
- label 标签列 
- alpha 惩罚项常数 
- fitIntercept 拟合时是否有截距项
- l1_ratio ElasticNet混合参数，其值为``0 <= l1_ratio <= 1''
- max_iter 最大迭代数
- normalize 是否归一化拟合
- tol 优化的容忍程度，小于这个值时停止迭代
- type 线性模型选择，可以选择Lasso，ElasticNet，Ridge


### 20、非凸惩罚函数回归

#### 命令行操作:
python bda-pylib/statistics/nonconvexPenalty/pycassoModel.py --train_data_path data/linear_data.csv --selection_path out/linear_select.csv --features V1,V2,V3
,V4,V5 --label label --penalty l1 
注：需要在linux下运行，不支持Window

#### 参数说明：
- train_data_path 模型训练数据，CSV文件格式，模型输入文件。
- selection_path CSV文件格式，模型结果文件。
- features 特征列 
- label 标签列   
- penalty 惩罚函数，可以选择L1，mcp，scad

### 21、重复测量方差分析

#### 命令行操作:

python bda-pylib/statistics/repeatedMeasurementAnalysisOfVariance/MultiComparison.py --method Tukey --input bda-pylib/data/sample.csv --col1 Treatment --col2
 "StressReduction"  --output bda-pylib/out/MultiComparison.csv

#### 参数说明：
- method 可选的分析方法Sidak、Bonferroni （one-step correction）Tukey、holm-sidak、holm、simes-hochberg、hommel、fdr_bh、fdr_by、fdr_tsbh、fdr_tsbky等 
- input CSV文件格式 待分析测试数据
- col1 组名特征，重复测量方差分析的组名信息列 
- col2 数值特征，重复测量方差分析的数值信息列   
- output CSV文件格式 模型的结果文件 

### 22、ARMA

#### 命令行操作:
python -W ignore bda-pylib/TimeSeriesAnalysis/ARMA.py --input bda-pylib/data/daily-minimum-temperatures-in-me.csv --col1 temperatures --p 2 --q 2 --type ARMA
 --output bda-pylib/out/pre.csv --n 30 --data_png bda-pylib/out/png/plot.png

#### 参数说明：
- input 模型训练数据，CSV文件格式，模型输入文件。
- col1 特征列
- p 自回归系数
- q 移动平均系数   
- type 模型选择：可选择AR模型，MA模型或者ARMA模型
- output CSV文件格式，模型结果文件
- data_png PNG文件格式，模型图片结果文件
- n 预测周期


### 23、差分运算

#### 命令行操作:
python   bda-pylib/TimeSeriesAnalysis/Diff.py --input bda-pylib/data/ChinaBank.csv --col1 Close --out bda-pylib/out/data_diff.csv --diffNum 2


#### 参数说明：
- input CSV文件格式 待分析测试数据
- col1 数据的特征列
- output CSV文件格式 模型的结果文件
- diffNum 差分的阶数

#### 24、ARIMA

#### 命令行操作:
python -W ignore bda-pylib/TimeSeriesAnalysis/ARIMA.py --input bda-pylib/data/ChinaBank.csv --col1 Date --col2 Close --out bda-pylib/out/predict_data.csv --d
1  --p 2 --q 3  --method 4 --n 20  --data_png bda-pylib/out/png/plot.png


#### 参数说明：
- input CSV文件格式  模型测试数据
- col1 数据的时间列
- col2 数据的特征列
- output CSV文件格式，模型结果文件
- method 模型选择可以选择1（时序图）、2（自相关和偏自相关系数）,3（差分图）、4(ARIMA训练和预测)。
- d 差分系数
- p 自相关系数
- q 偏自相关系数
- data_png png文件格式，模型图片文件
- n 预测周期