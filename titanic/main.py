"""
Titanic
"""

import pandas as pd
import numpy as np
from sklearn import tree

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")


def missing_value_table(df):
    """
    欠損値を探す
    """
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum() / len(df)
    missing_value_table = pd.concat([null_val, percent], axis=1)
    missing_value_table_ren_columns = missing_value_table.rename(
        columns={0: '欠損数', 1: '%'}
        )
    return missing_value_table_ren_columns


# 前処理
train["Age"] = train["Age"].fillna(train["Age"].mean())
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])
train.loc[train["Sex"] == "male", "Sex"] = 0
train.loc[train["Sex"] == "female", "Sex"] = 1
# train.replace("male", 0).replace("female", 1)
train["Embarked"] = train["Embarked"].map({"S": 0, "C": 1, "Q": 2})
# train["Embarked"][train["Embarked"] == "S"] = 0

test["Age"] = test["Age"].fillna(test["Age"].mean())
test["Sex"] = test["Sex"].map({"male": 0, "female": 1})
test["Embarked"] = test["Embarked"].map({"S": 0, "C": 1, "Q": 2})
test["Fare"] = test["Fare"].fillna(test["Fare"].mean())


# print(train.head())
# print(test.head())

# print(train.describe())
# print(test.describe())

# print(missing_value_table(train))
# print(missing_value_table(test))


# 決定木
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

my_prediction = my_tree_one.predict(test_features)

# print(my_prediction.shape)
# print(my_prediction)
PassengerId = np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(my_prediction, PassengerId, columns=["Survived"])
my_solution.to_csv("my_tree_one.csv", index_label=["PassengerId"])

features_two = train[["Pclass", "Age", "Sex", "Fare", "SibSp","Parch", "Embarked"]].values

max_depth = 10
min_samples_split = 5
my_tree_two = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
my_tree_two = my_tree_two.fit(features_two, target)

test_features_2 = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

my_prediction_tree_two = my_tree_two.predict(test_features_2)
PassengerId = np.array(test["PassengerId"]).astype(int)
my_solution_tree_two = pd.DataFrame(my_prediction_tree_two, PassengerId, columns=["Survived"])
my_solution_tree_two.to_csv("my_tree_two.csv", index_label=["PassengerId"])
