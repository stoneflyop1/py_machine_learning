import pandas as pd
import numpy as np
from wine_data import get_wine_data

X_train_std, X_test_std, y_train, y_test = get_wine_data()

d = X_train_std.shape[1] # feature size
class_count = len(np.unique(y_train))

np.set_printoptions(precision=4)
mean_vecs = []
for label in range(1,class_count+1):
    mean_vecs.append(np.mean(X_train_std[y_train==label], axis=0))
    print('MV %s: %s\n' %(label, mean_vecs[label-1]))

S_W = np.zeros((d,d))
for label, mv in zip(range(1,class_count+1), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train==label].T)
    # class_scatter = np.zeros((d,d))
    # for row in X_train_std[y_train==label]:
    #     row, mv = row.reshape(d, 1), mv.reshape(d, 1)
    #     class_scatter += (row-mv).dot((row-mv).T)
    S_W += class_scatter
print('Within-class scatter matrix: %sx%s' % (S_W.shape[0], S_W.shape[1]))
# print('Class label distribution: %s' % np.bincount(y_train)[1:])

mean_overall = np.mean(X_train_std, axis=0)
S_B = np.zeros((d,d))
for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train==i+1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)
    mean_overall = mean_overall.reshape(d,1)
S_B += n * (mean_vec - mean_overall).dot((mean_vec-mean_overall).T)
print('Between-class scatter matrix: %sx%s' % (S_B.shape[0], S_B.shape[1]))