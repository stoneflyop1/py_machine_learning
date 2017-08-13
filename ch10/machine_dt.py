import pandas as pd

# data from https://archive.ics.uci.edu/ml/datasets/Computer+Hardware
df = pd.read_csv('../data/machine.data', header=None)
df.columns = [
    'VENDOR', 'MODEL', 'MYCT', 'MMIN', 'MMAX',
    'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP'
]

import numpy as np

X = df[['PRP']].values
y = df['ERP'].values

import matplotlib.pyplot as plt

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')

from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=3)
tree.fit(X, y)
sort_idx = X.flatten().argsort()
lin_regplot(X[sort_idx], y[sort_idx], tree)
plt.xlabel('[PRP]')
plt.ylabel('[ERP]')
plt.show()

# Random Forests
from sklearn.model_selection import train_test_split
X = df[['CACH', 'CHMIN', 'CHMAX', 'PRP']].values
y = df['ERP'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(
    n_estimators=500, criterion="mse", random_state=1, n_jobs=2
)
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % (
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_test, y_test_pred)
))
from sklearn.metrics import r2_score
print(
    'R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)
    )
)
plt.scatter(
    y_train_pred, y_train_pred - y_train, c='black',
    marker='o', s=35, alpha=0.5, label='Training data'
)
plt.scatter(
    y_test_pred, y_test_pred - y_test, c='lightgreen',
    marker='s', s=35, alpha=0.7, label='Test data'
)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=0, xmax=1000, lw=2, color='red')
plt.xlim([0, 1000])
plt.show()