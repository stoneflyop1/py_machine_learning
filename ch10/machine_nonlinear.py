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

from sklearn.linear_model import LinearRegression
regr = LinearRegression()

# create polynomial features
from sklearn.preprocessing import PolynomialFeatures
quadratic = PolynomialFeatures(degree=2)
cubic = PolynomialFeatures(degree=3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

from sklearn.metrics import r2_score
# linear fit
X_fit = np.arange(X.min(), X.max(), 1)[:, np.newaxis]
regr = regr.fit(X, y)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y, regr.predict(X))

# quadratic fit
regr = regr.fit(X_quad, y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y, regr.predict(X_quad))

# cubic fit
regr = regr.fit(X_cubic, y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y, regr.predict(X_cubic))

# plot results
import matplotlib.pyplot as plt
plt.scatter(X, y, label='training points', color='lightgray')
plt.plot(
    X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$' % linear_r2,
    color='blue', lw=2, linestyle=':'
)
plt.plot(
    X_fit, y_quad_fit, label='quadratic (d=2), $R^2=%.2f$' % quadratic_r2,
    color='red', lw=2, linestyle='-'
)
plt.plot(
    X_fit, y_cubic_fit, label='cubic (d=3), $R^2=%.2f$' % cubic_r2,
    color='green', lw=2, linestyle='--'
)
plt.xlabel('% [PRP]')
plt.ylabel('[ERP]')
plt.legend(loc='upper left')
plt.show()

# transform features
X_log = np.log(X)
y_sqrt = np.sqrt(y)

# fit features
X_fit = np.arange(X_log.min()-1, X_log.max()+1, 1)[:, np.newaxis]
regr = regr.fit(X_log, y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt, regr.predict(X_log))

# plot results
plt.scatter(X_log, y_sqrt, label='training points', color='lightgray')
plt.plot(
    X_fit, y_lin_fit, label='linear (d=1), $R^2=%.2f$' % linear_r2,
    color='blue', lw=2
)
plt.xlabel('log(PRP)')
plt.ylabel('$\sqrt{ERP}$')
plt.legend(loc='lower left')
plt.show()