# coding:utf-8
"""
    File Name : xgboost
    Author :    Guanpx
    data :      18-1-22
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold  # kflod
import matplotlib


# 计算得分
def miss(pre, label):
    m = len(pre)
    sum = 0.0
    for i in range(0, m):
        sum += (pre[i] - label[i]) * (pre[i] - label[i])
    return sum / float(2 * m) + 0.01


# xgboost训练
def xgboost_train(train_DMdata, test_DMdata, num_boost_round, early):
    watchlist = [(train_DMdata, 'train')]
    para = {
        'booster': 'gbtree',
        'objective': 'count:poisson',
        'eval_metric': 'rmse',
        'eta': 0.3,  # alpha
        'silent': 1,
        'max_depth': 3,
        'subsample': 0.9,  # 0.9样本 防止过拟合
        "miss": -999,
        "lambda": 1.5
    }
    paras = {
        'booster': 'gbtree',
        # 'booster':'gblinear',
        # 'objective': 'reg:linear',
        'objective': 'count:poisson',
        'eval_metric': 'rmse',
        'eta': 0.03,  # alpha
        'silent': 1,
        'max_depth': 3,
        'subsample': 0.9,  # 0.9样本 防止过拟合
        "miss": -999,
        "colsample_bytree": 0.7
    }

    model = xgb.train(paras, train_DMdata, num_boost_round=num_boost_round,
                      evals=watchlist, early_stopping_rounds=early)

    pre = model.predict(test_DMdata)
    return pre


# cv测试
def cv_miss(n_splits, train_file, num, early):
    kf = KFold(n_splits=n_splits)
    loop = 0
    mean_miss = 0.0
    for train_index, test_index in kf.split(train_file.values):
        loop += 1
        train_data = train_file.values[train_index][:, 1:-1]
        train_label = train_file.values[train_index][:, -1]
        train_DMdata = xgb.DMatrix(train_data, train_label)

        test_data = train_file.values[test_index][:, 1:-1]
        test_label = train_file.values[test_index][:, -1]
        test_DMdata = xgb.DMatrix(test_data, test_label)

        mean_miss += miss(test_label, xgboost_train(train_DMdata, test_DMdata, num, early))

    return mean_miss / float(loop)


def main():
    test_file = pd.read_csv("../data/testA.csv")
    train_file = pd.read_csv("../data/train.csv")

    # 处理为DMatrix
    # train_file = train_file[train_file["label"] <= 18]
    # train_file = train_file.drop(["3", "9", "12", "19", "21", "29", "31"], axis=1)
    # test_file = test_file.drop(["3", "9", "12", "19", "21", "29", "31"], axis=1)

    train_data = train_file.values[:, 1:-1]
    label = train_file.values[:, -1]
    test_data = test_file.values[:, 1:]
    train_DMdata = xgb.DMatrix(train_data, label)
    test_DMdata = xgb.DMatrix(test_data)

    # pre = xgboost_train(train_DMdata, test_DMdata, 1000, 50)
    print
    cv_miss(3, train_file, 1000, 50)
    # ans = pd.read_csv("../data/answer.csv").values
    # print miss(ans, pre)


if __name__ == '__main__':
    main()
