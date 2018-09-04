# coding:utf-8
"""
    File Name : _lightgbm_main
    Author :    Guanpx
    data :      18-1-29
"""

import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import KFold


# 计算得分
def miss(pre, label):
    m = len(pre)
    sum = 0.0
    for i in range(0, m):
        sum += (pre[i] - label[i]) * (pre[i] - label[i])
    return sum / float(2 * m)


# lgb训练保存模型
def lgb_train(train_ds, test_train_ds, num_round, early):
    para = {
        'learning_rate': 0.01,  # 3000 150
        'boosting_type': 'goss',
        'objective': 'poisson',
        'metric': 'l2',  # L2
        # 'metric': 'mae', # L1
        'sub_feature': 0.9,
        "max_depth": 4,
        'colsample_bytree': 0.8,
        'verbose': -1
    }

    model = lgb.train(para, train_ds,
                      num_boost_round=num_round,
                      early_stopping_rounds=early,
                      valid_sets=test_train_ds,
                      verbose_eval=100,
                      )
    model.save_model("./Save/model.txt",
                     num_iteration=model.best_iteration)


# 读出模型
def restore(test_ds):
    model = lgb.Booster(model_file="./Save/model.txt")
    ans = model.predict(test_ds, num_iteration=model.best_iteration)
    return ans


# cv测试
def cv_miss(n_splits, train_file, num, early):
    kf = KFold(n_splits=n_splits)
    loop = 0
    mean_miss = 0.0
    for train_index, test_index in kf.split(train_file.values):
        loop += 1
        train_data = train_file.values[train_index][:, 1:-1]
        train_label = train_file.values[train_index][:, -1]

        train_ds = lgb.Dataset(train_data, train_label)
        test_data = train_file.values[test_index][:, 1:-1]
        test_label = train_file.values[test_index][:, -1]
        test_train_ds = lgb.Dataset(test_data, test_label)

        lgb_train(train_ds, test_train_ds, num, early)
        mean_miss += miss(test_label, restore(test_data))

    return mean_miss / float(loop)


if __name__ == '__main__':
    test_file = pd.read_csv("../data/testA.csv")
    train_file = pd.read_csv("../data/train.csv")
    test_label = pd.read_csv("../data/answer.csv")

    # cv 测试
    print
    cv_miss(5, train_file, 1000, 50)

    # 删除18以下的 ???
    train_file = train_file[train_file["label"] <= 20]
    #
    # train_data = train_file.values[:, 1:-1]
    # train_label = train_file.values[:, -1]
    # test_data = test_file.values[:, 1:]
    # test_label = test_label.values
    #
    # train_Ds_1 = lgb.Dataset(train_data[:4000, :], train_label[:4000])
    # train_Ds_2 = lgb.Dataset(train_data[4000:4600, :], train_label[4000:4600])
    # test_Ds = lgb.Dataset(test_data)

    # lgb_train(train_Ds_1, train_Ds_2, 1000, 50)  # 0.82607332
