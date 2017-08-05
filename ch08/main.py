import pandas as pd
df = pd.read_csv('../data/movie_data.csv')

import cleandata
df['review'] = df['review'].apply(cleandata.preprocessor)

# grid search, 非常耗时
#import gridlearn
#gridlearn.learn(df)

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

import tokendata

vect = HashingVectorizer(
    decode_error='ignore', n_features=(2 ** 21), 
    preprocessor=None, tokenizer=tokendata.tokenizer
)
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)
import ooclearn
doc_stream = ooclearn.stream_docs(path='../data/movie_data.csv')

import pyprind # 进度条
pbar = pyprind.ProgBar(45)
import numpy as np
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = ooclearn.get_minibatch(doc_stream, size=1000)
    if not X_train: break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

X_test, y_test = ooclearn.get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))