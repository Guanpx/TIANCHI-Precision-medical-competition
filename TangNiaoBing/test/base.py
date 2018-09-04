# coding:utf-8
"""
    File Name : base
    Author :    Guanpx
    data :      18-1-29
"""
import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error

data_path = '../data/'

train = pd.read_csv('d_train_20180102.csv', encoding='gb2312')
test = pd.read_csv('d_test_A_20180102.csv', encoding='gb2312')


def make_feat(train, test):
    train_id = train.id.values.copy()
    test_id = test.id.values.copy()
    train = train[train['年龄'] >= 16]
    train = train[train['血糖'] <= 18]

    data = pd.concat([train, test])

    data['性别'] = data['性别'].map({'男': 1, '女': 0, '??': 1})

    # data['体检日期'] = (pd.to_datetime(data['体检日期']) - parse('2017-10-09')).dt.days
    # del data['体检日期']
    data.drop(['乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体', '体检日期'], axis=1, inplace=True)

    # data.fillna(data.median(axis=0))

    train_feat = data[data.id.isin(train_id)]
    test_feat = data[data.id.isin(test_id)]

    del train_feat['id']
    del test_feat['id']
    return train_feat, test_feat


train_feat, test_feat = make_feat(train, test)

predictors = [f for f in test_feat.columns if f not in ['血糖']]


def evalerror(pred, df):
    label = df.get_label().values.copy()
    score = mean_squared_error(label, pred) * 0.5
    return ('mse', score, False)


# 后验修正
def xx(x):
    max_x = max(x) * 0.95
    qut_x = max(x) * 0.9
    min_x = max(x) * 0.25

    # for i,t in enumerate(x):
    #     if x[i] >= max_x:
    #         x[i] = x[i] * 1.5
    #     elif (x[i] < max_x) & (x[i] >= qut_x):
    #         x[i] = x[i] * 1.25
    #     elif x[i] <= min_x:
    #         x[i] = x[i] * 0.75
    #     else:
    # x[i] = x[i]
    return x


print('开始训练...')
params = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'sub_feature': 0.9,
    'num_leaves': 16,
    'colsample_bytree': 0.9,
    # 'feature_fraction': 0.9,
    # 'min_data': 100,
    # 'min_hessian': 1,
    'verbose': -1,
}

print('开始CV 5折训练...')
scores = []
t0 = time.time()
train_preds = np.zeros(train_feat.shape[0])
test_preds = np.zeros((test_feat.shape[0], 5))

kf = KFold(len(train_feat), n_folds=5, shuffle=True, random_state=520)
for i, (train_index, test_index) in enumerate(kf):
    print('第{}次训练...'.format(i))
    train_feat1 = train_feat.iloc[train_index]
    # print(train_feat1)
    train_feat2 = train_feat.iloc[test_index]
    lgb_train1 = lgb.Dataset(train_feat1[predictors], train_feat1['血糖'], categorical_feature=['性别'])
    lgb_train2 = lgb.Dataset(train_feat2[predictors], train_feat2['血糖'])
    gbm = lgb.train(params,
                    lgb_train1,
                    num_boost_round=3000,
                    valid_sets=lgb_train2,
                    verbose_eval=100,
                    feval=evalerror,
                    early_stopping_rounds=150)
    feat_imp = pd.Series(gbm.feature_importance(), index=predictors).sort_values(ascending=False)
    train_preds[test_index] += xx(gbm.predict(train_feat2[predictors]))
    test_preds[:, i] = xx(gbm.predict(test_feat[predictors]))

print('线下得分：    {}'.format(mean_squared_error(train_feat['血糖'], train_preds) * 0.5))
print('CV训练用时{}秒'.format(time.time() - t0))
print(train_preds)
submission = pd.DataFrame({'pred': test_preds.mean(axis=1)})
# submission.loc[submission.pred > 10,'pred'] = 20
print(submission.describe())
submission.to_csv(r'sub{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d')), header=None,
                  index=False, float_format='%.4f')
