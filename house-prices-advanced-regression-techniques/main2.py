import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['WhatIsData'] = 'Train'
test['WhatIsData'] = 'Test'
test['SalePrice'] = 10**9 + 7
alldata = pd.concat([train, test], axis=0).reset_index(drop=True)
print('The size of train is : ' + str(train.shape))
print('The size of testis : ' + str(test.shape))

print(train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False))
test.isnull().sum()[test.isnull().sum()>0].sort_values(ascending=False)
