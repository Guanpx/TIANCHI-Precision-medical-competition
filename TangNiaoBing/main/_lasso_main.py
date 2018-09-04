# coding:utf-8

"""
    Fail
    File Name : _csdn_github
    Author :    Guanpx
    data :      18-1-31
"""

import pandas as pd

from sklearn.model_selection import KFold
import numpy as np
from time import time
from sklearn.linear_model import Lasso, LassoCV

a = time()


def miss(ans, pre):
    sum = 0.0
    for i in range(len(pre)):
        sum += np.square(pre[i] - ans[i])

    return (sum / float(len(pre) * 2))


# test_a 已经处理好的数据 除了空值
test_file = pd.read_csv("../data/test_a.csv", encoding="utf-8")
train_file = pd.read_csv("../data/train_a.csv", encoding="utf-8")
for i in range(1, train_file.shape[-1] - 1):
    median = train_file[str(i)].median()
    train_file = train_file.fillna(value={str(i): median})

for i in range(1, test_file.shape[-1]):
    median = test_file[str(i)].median()
    test_file = test_file.fillna(value={str(i): median})

test_file = test_file.fillna(0)
train_file = train_file.fillna(0)
print(test_file.head())
print(train_file.head())
print("test_file shape :", test_file.shape)
print("train_file shape :", train_file.shape)

train_label = train_file.values[:, -1]
train_data = train_file.values[:, :-1]

test_data = test_file.values
test_label = pd.read_csv("../data/answer.csv", encoding="utf-8").values

print(train_label)

n_splits = 5
kf = KFold(n_splits=n_splits)
sum = 0.0
for train, test in kf.split(train_data):
    x = train_data[train]
    y = train_label[train]
    x_test = train_data[test]
    y_test = train_label[test]
    lassocv = LassoCV()
    lassocv.fit(x, y)
    print(lassocv.alpha_)
    model = Lasso(lassocv.alpha_)
    model.fit(x, y)
    sum += miss(model.predict(x_test), y_test)

model = Lasso(0.06)
model.fit(train_data, train_label)
pre = model.predict(test_data)

print("cv", sum / float(n_splits))
print("real", miss(pre, test_label))
print("time", time() - a)

"""
cv 0.814837870298924
real [0.89833792]
time 1.6675097942352295
"""
