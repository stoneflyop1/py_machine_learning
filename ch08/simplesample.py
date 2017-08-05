import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining and the weather is sweet'])
bag = count.fit_transform(docs)
print(count.vocabulary_)
print(bag.toarray())

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer(norm='l2') # normalized data
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

