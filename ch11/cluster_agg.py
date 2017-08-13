import pandas as pd
import numpy as np
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_'+str(i) for i in range(0,5)]
X = np.random.random_sample([len(labels),len(variables)])*10
df = pd.DataFrame(X, columns=variables, index=labels)
#print(df)

from scipy.spatial.distance import pdist, squareform
row_dist = pd.DataFrame(
    squareform(pdist(df, metric='euclidean')), columns=labels, index=labels
)
print(row_dist)

from scipy.cluster.hierarchy import linkage
## Incorrect approach
#row_clusters = linkage(row_dist, method='complete', metric='euclidean')
## Correct approach
#row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
# Correct approach
row_clusters = linkage(df.values, method='complete', metric='euclidean')

# df = pd.DataFrame(
#     row_clusters,
#     columns=['row label 1', 'row label2', 'distance', 'no. of items in clust.'],
#     index=['cluster %d' %(i+1) for i in range(row_clusters.shape[0])]
# )
# print(df)

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
# row_dendr = dendrogram(
#     row_clusters, labels=labels
# )
# plt.tight_layout()
# plt.ylabel('Euclidean distance')
# plt.show()

# add to heat map
fig = plt.figure(figsize=(8,8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='left')

df_rowclust = df.ix[row_dendr['leaves'][::-1]]

axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust, interpolation='nearest', cmap='hot_r')

axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values(): i.set_visible(False)
fig.colorbar(cax)
axm.set_xticklabels([''] + list(df_rowclust.columns))
axm.set_yticklabels([''] + list(df_rowclust.index))
plt.show()

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2,
                                affinity="euclidean",
                                linkage='complete')
labels = ac.fit_predict(X)
print('Cluster labels: %s' % labels)