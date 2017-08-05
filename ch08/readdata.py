import pandas as pd
import os

labels = {'pos':1, 'neg': 0}
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = '../data/aclImdb/%s/%s' % (s, l)
        for f in os.listdir(path):
            with open(os.path.join(path, f), 'r') as infile:
                txt = infile.read()
                df = df.append([[txt, labels[l]]], ignore_index=True)
df.columns = ['review', 'sentiment']


import numpy as np
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('../data/movie_data.csv', index=False)

df = pd.read_csv('../data/movie_data.csv')
print(df.head(3))