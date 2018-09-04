# coding:utf-8
"""
    File Name : _csdn_xgb_lgb
    Author :    Guanpx
    data :      18-1-31
"""

import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
import numpy as np
from time import time

a = time()


def miss(ans, pre):
    sum = 0.0
    for i in range(len(pre)):
        sum += np.square(pre[i] - ans[i])

    return (sum / float(len(pre) * 2))


train_file = pd.read_csv("../data/d_train_20180102.csv", encoding='gbk')
test_file = pd.read_csv("../data/d_test_A_20180102.csv", encoding='gbk')

# print('train shape', train_file.shape)
# print('test shape', test_file.shape)
# print (train_file.head())


# 男1 女0
train_file['性别'] = train_file['性别'].map({'男': 1, '女': 0, '??': 0})
test_file["性别"] = test_file['性别'].map({'男': 1, '女': 0, '??': 0})

# 去掉乙肝五项 id 体检日期
train_file.drop(['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体', 'id', '体检日期'], axis=1, inplace=True)
test_file.drop(['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体', 'id', '体检日期'], axis=1, inplace=True)
# 白球比例是指肝功能白球比例（白球比）=白蛋白/球蛋白=白蛋白/（总蛋白-白蛋白）
# 单核细胞 白细胞冲突
train_file.drop(['单核细胞%', '白球比例', '白蛋白', '*总蛋白'], axis=1, inplace=True)
test_file.drop(['单核细胞%', '白球比例', '白蛋白', '*总蛋白'], axis=1, inplace=True)

# 合并相似项目 酶
train_file['酶'] = train_file['*天门冬氨酸氨基转换酶'] + train_file["*丙氨酸氨基转换酶"] \
                  + train_file["*碱性磷酸酶"] + train_file["*r-谷氨酰基转换酶"]
test_file['酶'] = test_file["*天门冬氨酸氨基转换酶"] + test_file["*丙氨酸氨基转换酶"] \
                 + test_file["*碱性磷酸酶"] + test_file["*r-谷氨酰基转换酶"]

# 血细胞计数 * 平均参数
train_file['红细胞总血红蛋白量'] = train_file['红细胞计数'] * train_file['红细胞平均血红蛋白量']
test_file['红细胞总血红蛋白量'] = test_file['红细胞计数'] * test_file['红细胞平均血红蛋白量']

train_file['红细胞总血红蛋白浓度'] = train_file['红细胞计数'] * train_file['红细胞平均血红蛋白浓度']
test_file['红细胞总血红蛋白浓度'] = test_file['红细胞计数'] * test_file['红细胞平均血红蛋白浓度']

train_file['红细胞总体积'] = train_file['红细胞计数'] * train_file['红细胞平均体积']
test_file['红细胞总体积'] = test_file['红细胞计数'] * test_file['红细胞平均体积']

train_file['血小板总体积'] = train_file['血小板计数'] * train_file['血小板平均体积']
test_file['血小板总体积'] = test_file['血小板计数'] * test_file['血小板平均体积']
# cv 0.7912094213627026

# 肾指标
train_file['肾'] = train_file['尿酸'] + train_file['尿素'] + train_file['肌酐']
test_file['肾'] = test_file['尿酸'] + test_file['尿素'] + test_file['肌酐']

# 嗜酸细胞
train_file["嗜酸细胞"] = train_file['白细胞计数'] * train_file["嗜酸细胞%"]
test_file["嗜酸细胞"] = test_file['白细胞计数'] * test_file["嗜酸细胞%"]

# 删除嗜碱细胞 与嗜酸细胞矛盾
train_file.drop(['嗜碱细胞%'], axis=1, inplace=True)
test_file.drop(['嗜碱细胞%'], axis=1, inplace=True)

# 删除年龄血糖异常值
# train_file = train_file[train_file['年龄'] >= 16]
train_file = train_file[train_file['血糖'] <= 18]
tmp = train_file['血糖']
train_file.drop(['血糖'], axis=1, inplace=True)
train_file['血糖'] = tmp

# print (test_file.head())
# print (train_file.head())
print("test_file shape :", test_file.shape)
print("train_file shape", train_file.shape)

train_label = train_file.values[:, -1]
train_data = train_file.values[:, :-1]

test_data = test_file.values
test_label = pd.read_csv("../data/answer.csv", encoding="utf-8").values

# xgb
xgb_para = {
    'booster': 'gbtree',
    'objective': 'count:poisson',
    'eval_metric': 'rmse',
    'eta': 0.03,  #
    'silent': 1,
    'max_depth': 4,  # 4 best
    'subsample': 0.9,  # 0.9 best
    "miss": -999,
    "colsample_bytree": 0.7,
    "lambda": 1.5  # 1.5 best
}

train_DMdata = xgb.DMatrix(train_data, train_label)
watchlist = [(train_DMdata, 'train')]
num_boost_round = 1000
early_stopping_rounds = 50

n_splits = 3  # 0.7918201324482904
kf = KFold(n_splits=n_splits)
sum = 0.0
for train, test in kf.split(train_data):
    x = train_data[train]
    y = train_label[train]
    x_test = train_data[test]
    y_test = train_label[test]
    x = xgb.DMatrix(x, y)
    x_test = xgb.DMatrix(x_test)
    model = xgb.train(xgb_para, x,
                      num_boost_round=num_boost_round,
                      evals=watchlist,
                      early_stopping_rounds=early_stopping_rounds)
    sum += miss(model.predict(x_test), y_test)

model = xgb.train(xgb_para, train_DMdata,
                  num_boost_round=num_boost_round,
                  evals=watchlist,
                  early_stopping_rounds=early_stopping_rounds)
test_DMdata = xgb.DMatrix(test_data)
pre = model.predict(test_DMdata)

print("cv", sum / float(n_splits))
print("real", miss(pre, test_label))
print("time", time() - a)

b = time()

lgb_para = {
    'learning_rate': 0.03,  # 1500 50
    'boosting_type': 'goss',
    'objective': 'poisson',
    'metric': 'rmse',  # L2-root
    'max_depth': 4,
    'sub_feature': 0.9,
    'colsample_bytree': 0.9,
    'verbose': -1  # <0 打印最少
}

num_boost_round = 1500
early_stopping_rounds = 50
train_ds = lgb.Dataset(train_data[:4000, :], train_label[:4000])
train_test_ds = lgb.Dataset(train_data[4000:, :], train_label[4000:])
test_ds = lgb.Dataset(test_data)

n_splits = 3
kf = KFold(n_splits=n_splits)
sum = 0.0
for train, test in kf.split(train_data):
    x = train_data[train]
    y = train_label[train]
    x_test = train_data[test]
    y_test = train_label[test]
    print("lens is:", len(x))
    x_test_train = lgb.Dataset(x[2800:, :], y[2800:])
    x = lgb.Dataset(x[:2800, :], y[:2800])

    model = lgb.train(lgb_para, x, valid_sets=x_test_train,
                      verbose_eval=100,
                      num_boost_round=num_boost_round,
                      early_stopping_rounds=50)
    sum += miss(model.predict(x_test), y_test)

model = lgb.train(lgb_para, train_ds, valid_sets=train_test_ds,
                  early_stopping_rounds=early_stopping_rounds,
                  num_boost_round=num_boost_round,
                  verbose_eval=100)

pre = model.predict(test_data)

print("\ncv :", sum / float(n_splits))
print("real :", miss(test_label, pre))
print("time", time() - b)
