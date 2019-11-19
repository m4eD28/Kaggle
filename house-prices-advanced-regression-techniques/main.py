"""House prise"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
import missingno as msno
import xgboost as xgb

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.dtypes)

for i in range(train.shape[1]):
    if train.iloc[:,i].dtypes == object:
        lbl = LabelEncoder()
        lbl.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))
        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))
        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))

print(train.dtypes)

# print(msno.matrix(df=train, figsize=(20, 14), color=(0.5, 0, 0)))
# plt.show()

train_ID = train['Id']
test_ID = test['Id']

y_train = train['SalePrice']
X_train = train.drop(['Id', 'SalePrice'], axis=1)
X_test = test.drop('Id', axis=1)

Xmat = pd.concat([X_train, X_test])
Xmat = Xmat.drop(['LotFrontage', 'MasVnrArea', 'GarageYrBlt'], axis=1)
Xmat = Xmat.fillna(Xmat.median())

# print(msno.matrix(df=Xmat, figsize=(20, 14), color=(0.5, 0, 0)))
# plt.show() 
Xmat['TotalSF'] = Xmat['TotalBsmtSF'] + Xmat['1stFlrSF'] + Xmat['2ndFlrSF']
# ax = sns.distplot(y_train)
# plt.show()

y_train = np.log(y_train)
# ax = sns.distplot(y_train)
# plt.show()

X_train = Xmat.iloc[:train.shape[0], :]
X_test = Xmat.iloc[train.shape[0]:, :]

rf = RandomForestRegressor(n_estimators=80, max_features='auto')
rf.fit(X_train, y_train)
print('Training done using Random Forest')

ranking = np.argsort(-rf.feature_importances_)
f, ax = plt.subplots(figsize=(11, 9))
# sns.barplot(x=rf.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')
# ax.set_xlabel("feature importance")
# plt.tight_layout()
# plt.show()

X_train = X_train.iloc[:, ranking[:30]]
X_test = X_test.iloc[:, ranking[:30]]
X_train["Interaction"] = X_train["TotalSF"]*X_train["OverallQual"]
X_test["Interaction"] = X_test["TotalSF"]*X_train["OverallQual"]

fig = plt.figure(figsize=(12, 7))
# for i in np.arange(30):
#     ax = fig.add_subplot(5, 6, i+1)
#     sns.regplot(x=X_train.iloc[:,i], y=y_train)

# plt.tight_layout()
# plt.show()

Xmat = X_train
Xmat['SalePrice'] = y_train
Xmat = Xmat.drop(Xmat[(Xmat['TotalSF']>5) & (Xmat['SalePrice'] < 12.5)].index)
Xmat = Xmat.drop(Xmat[(Xmat['GrLivArea'] > 5) & (Xmat['SalePrice'] < 13)].index)

y_train = Xmat['SalePrice']
X_train = Xmat.drop(['SalePrice'], axis=1)


print("Parameter optimization")
xgb_model = xgb.XGBRegressor()
reg_xgb = GridSearchCV(xgb_model,
                   {'max_depth': [2,4,6],
                    'n_estimators': [50,100,200]}, verbose=1)
reg_xgb.fit(X_train, y_train)
print(reg_xgb.best_score_)
print(reg_xgb.best_params_)


