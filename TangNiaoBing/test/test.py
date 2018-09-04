# coding:utf-8
"""
    File Name : test
    Author :    Guanpx
    data :      18-1-20
"""
import matplotlib.pylab as plt
from time import time
import pandas as pd
import numpy as np


def point_pic(x, y, title):
    plt.plot(x, y, "*")
    plt.title(title)
    plt.show()


def all_print_point_pic(vec):
    label = vec[:, -1]  # 标签
    for i in range(1, 40):
        row = vec[:, i]  # 特征
        print
        row


def miss(pre, label):
    m = len(pre)
    sum = 0.0
    for i in range(0, m):
        sum += (pre[i] - label[i]) * (pre[i] - label[i])
    return sum / float(2 * m)


if __name__ == '__main__':
    a = time()
    # pre = pd.read_csv("../main/ans_mean_poisson.csv").values
    ans = pd.read_csv("../data/answer.csv").values
    pre = pd.read_csv("../test/result_1.csv").values
    print
    miss(pre, ans)

    print
    "Time:", time() - a
