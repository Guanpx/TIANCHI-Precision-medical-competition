# coding:utf-8
"""
    File Name : cnn_main
    Author :    Guanpx
    data :      18-1-22
"""

import tensorflow as tf
from sklearn import preprocessing
import pandas as pd
import numpy as np
from time import time
from sklearn.model_selection import KFold


def miss(pre, label):
    m = len(pre)
    sum = 0.0
    for i in range(0, m):
        sum += (pre[i] - label[i]) * (pre[i] - label[i])
    return sum / float(2 * m)


# 权重
def Weight(shape):
    weight = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(weight)  # var


# 偏置
def Bias(shape):
    bias = tf.constant(0.1, shape=shape)
    return tf.Variable(bias)


# 卷积
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# 池化,没用到
def max_pool(conv):
    return tf.nn.max_pool(conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# cnn模型函数
def CNN_model(x_train, y_train, x_test, lens, train_loop_num,
              learning_rate, save_model_name, Optimizer=tf.train.AdamOptimizer):
    # 占位符
    x = tf.placeholder(tf.float32, shape=[None, 36], name="input_x")
    y = tf.placeholder(tf.float32, shape=[None, 1], name="input_y")

    # reshape
    x_data = tf.reshape(x, shape=[-1, 6, 6, 1])

    # 卷积1
    W_conv1 = Weight(shape=[2, 2, 1, 32])
    Bias_conv1 = Bias(shape=[32])
    conv1 = tf.nn.relu(conv2d(x_data, W_conv1) + Bias_conv1)

    # 卷积2
    W_conv2 = Weight(shape=[2, 2, 32, 64])
    Bias_conv2 = Bias(shape=[64])
    conv2 = tf.nn.relu(conv2d(conv1, W_conv2) + Bias_conv2)

    # 全链接
    W_fc = Weight(shape=[6 * 6 * 64, 512])
    Bias_fc = Bias(shape=[512])
    conv2_reshape = tf.reshape(conv2, shape=[-1, 6 * 6 * 64])
    fc = tf.nn.relu(tf.matmul(conv2_reshape, W_fc) + Bias_fc)

    # dropout层
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    fc_drop_out = tf.nn.dropout(fc, keep_prob)

    # 输出层
    W_out = Weight(shape=[512, 1])
    Bias_out = Bias(shape=[1])
    y_pre = tf.matmul(fc_drop_out, W_out) + Bias_out  # 注意

    # saver保存模型 注意add_to_collection
    saver = tf.train.Saver()
    tf.add_to_collection("y_pre", y_pre)

    # loss优化层
    loss = tf.reduce_mean(tf.square(y - y_pre))
    train_step = Optimizer(learning_rate).minimize(loss)

    # 计算分数
    miss = tf.div(tf.reduce_sum(tf.square(y - y_pre)), 2 * lens)

    # Session 初始化
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    # 训练
    for i in range(train_loop_num):
        sess.run(train_step, feed_dict={x: x_train, y: y_train, keep_prob: 0.5})
        # print sess.run(miss, feed_dict={x: x_train, y: y_train, keep_prob: 1})
        print
        sess.run(loss, feed_dict={x: x_train, y: y_train, keep_prob: 1})
        # 保存模型
        if train_loop_num - i == 1:
            saver.save(sess, save_model_name)


# cv测试
def cv_cnn_model(n_splits, label, data, save_model_name):
    sum = 0.0
    kf = KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(data):
        train_data = data[train_index]
        train_label = label[train_index]
        test_data = data[test_index]
        test_label = label[test_index]
        ans = restore_(save_model_name, test_data)
        sum += miss(test_label, ans)
    print
    float(sum) / float(n_splits)


# 读出模型
def restore_(save_model_name, test_data):
    with tf.Session() as sess:
        new_saver = tf.train.import_meta_graph(save_model_name + '.meta')  # 注意 .meta
        new_saver.restore(sess, save_model_name)
        y_pre = tf.get_collection("y_pre")  # 与to_collection对应
        graph = tf.get_default_graph()  # 从默认图中读出占位符
        input_x = graph.get_operation_by_name('input_x').outputs[0]
        keep_prob = graph.get_operation_by_name('keep_prob').outputs[0]

        y = sess.run(y_pre, feed_dict={input_x: test_data, keep_prob: 1})[0]
        return y


if __name__ == '__main__':
    test_file = pd.read_csv("test_mean.csv")
    train_file = pd.read_csv("train_mean.csv")

    train_data = train_file.values[:, 1:-1]
    train_data = np.column_stack([train_data, train_data[:, 5:7]])
    label = train_file.values[:, -1]
    test_data = test_file.values[:, 1:]
    test_data = np.column_stack([test_data, test_data[:, 5:7]])

    label = label.reshape((len(label), 1))
    standard = preprocessing.StandardScaler()
    train_data = standard.fit_transform(train_data)
    test_data = standard.fit_transform(test_data)

    a = time()
    save_name = "Save/Adam_0.01_1000_512_standard"
    # CNN_model(train_data,label,test_data,len(label),2000,0.01,save_name)
    cv_cnn_model(5, label, train_data, save_model_name=save_name)

    # ans = restore_(save_name, test_data)
    # ans = map(lambda x : float(x), ans)
    # ans_file = pd.Series(map(lambda x: "{:.3f}".format(x), ans))  # 保留三位小数
    # ans_file.to_csv("ans_Adam0.01_2000_512_0.5_standard.csv", index=False)

    print
    "time", time() - a
