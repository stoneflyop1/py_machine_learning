
import pandas as pd

# data from https://archive.ics.uci.edu/ml/datasets/Computer+Hardware
df = pd.read_csv('../data/machine.data', header=None)
df.columns = [
    'VENDOR', 'MODEL', 'MYCT', 'MMIN', 'MMAX',
    'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP'
]

# print(df.head())

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context="notebook")
cols = ['CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']

# sns.pairplot(df[cols], size=2.0)
# plt.show()

# 恢复默认的matplotlib的样式设置
# sns.reset_orig()

import numpy as np
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(
    cm, cbar=True, annot=True, square=True, fmt=".2f",
    annot_kws={'size':15}, yticklabels=cols, xticklabels=cols
)
plt.show()

class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta *X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:] + self.w_[0])

    def predict(self, X):
        return self.net_input(X)

X = df[['PRP']].values
y = df['ERP'].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)

lr = LinearRegressionGD()
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')

lin_regplot(X_std, y_std, lr)
plt.xlabel('PRP (standardized)')
plt.ylabel('ERP (standardized)')
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(
    LinearRegression(), max_trials=100, min_samples=50, random_state=0,
    residual_metric=lambda x: np.sum(np.abs(x), axis=1), residual_threshold=50.0
)
ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X = np.arange(0,1200,50)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(X[inlier_mask], y[inlier_mask], c='blue', marker='o', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], c='lightgreen', marker='s', label='Outliers')
plt.plot(line_X, line_y_ransac, color='red')
plt.xlabel('Average PRP')
plt.ylabel('ERP')
plt.legend(loc='upper left')
plt.show()
print('Slope: %.3f' % ransac.estimator_.coef_[0])
print('Intercept: %.3f' % ransac.estimator_.intercept_)


from sklearn.model_selection import train_test_split
X = df[['CACH', 'CHMIN', 'CHMAX', 'PRP']].values
y = df['ERP'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)
slr = LinearRegression()
slr.fit(X_train, y_train)
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

plt.scatter(
    y_train_pred, y_train_pred - y_train,
    c='blue', marker='o', label='Training data'
)
plt.scatter(
    y_test_pred, y_test_pred - y_test,
    c='lightgreen', marker='s', label='Test data'
)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=0, xmax=1000, lw=2, color='red')
plt.xlim([0, 1000])
plt.show()

from sklearn.metrics import mean_squared_error
print('MSE train: %.3f, test: %.3f' % (
    mean_squared_error(y_train, y_train_pred),
    mean_squared_error(y_test, y_test_pred)
))

from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f' % (
    r2_score(y_train, y_train_pred),
    r2_score(y_test, y_test_pred)
))