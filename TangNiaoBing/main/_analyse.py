# coding:utf-8
"""
    File Name : analyse
    Author :    Guanpx
    data :      18-1-19
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from time import time
import seaborn as sns


def bar_pic(xname, y, title):
    # plt.legend() #sample
    x = [i for i in range(1, len(y) + 1)]
    plt.bar(x, y, 0.1)
    plt.xticks(x, xname)
    plt.title(title)
    plt.savefig(title + ".png")
    plt.show()


def point_pic(x, y, title):
    plt.plot(x, y, ".")
    plt.title(title + "-label")
    plt.ylabel(title)
    plt.xlabel("label")
    plt.show()


# 打印特征标签圆点图
def all_print_point_pic(vec):
    label = vec[:, -1]  # 标签
    for i in range(1, 40):
        row = vec[:, i]  # 特征
        point_pic(label, row, str(i))


# 打印describe柱型图
def all_print_bar_pic(train_file):
    analyse_data = train_file.describe()
    col = np.array(analyse_data.columns[1:])
    ind = np.array(analyse_data.index)
    num = 0
    for name in ind:
        print(name)
        line = analyse_data.values[num, 1:]
        bar_pic(col, line, name)
        num += 1


# 删除乙肝五项
def del_5(train_file, test_file):
    for i in (18, 19, 20, 21, 22):
        del test_file[str(i)]
        del train_file[str(i)]

    test_file.to_csv("test_numlabel_del5.csv", index=False)
    train_file.to_csv("train_numlabel_del5.csv", index=False)


# lambda x : x+1
# 中值填充空值
def fill_median(train_file, test_file):
    for i in range(2, train_file.shape[-1] - 1):
        median = train_file[str(i)].median()
        train_file = train_file.fillna(value={str(i): median})

    for i in range(2, test_file.shape[-1]):
        median = test_file[str(i)].median()
        test_file = test_file.fillna(value={str(i): median})

    test_file.to_csv("test_median.csv", index=False)
    train_file.to_csv("train_median.csv", index=False)


if __name__ == '__main__':
    a = time()
    test_file = pd.read_csv("../data/testA.csv")
    train_file = pd.read_csv("../data/train.csv")

    # train_file = train_file.drop(["3","9","12","19","21","29","31"], axis=1)

    # sns.heatmap(train_file.corr(), fmt=".1f", annot=True)
    # sns.plt.show()

    print("Time:", time() - a)

# 肝功能障碍
