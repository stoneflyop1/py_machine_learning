import pandas as pd
df = pd.read_csv('../data/movie_data.csv')

import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # remove html markups
    emotions = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emotions).replace('-', '')
    return text

df['review'] = df['review'].apply(preprocessor)


from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

def tokenizer(text):
    return text.split()

vect = HashingVectorizer(
    decode_error='ignore', n_features=(2 ** 21), 
    preprocessor=None, tokenizer=tokenizer
)
clf = SGDClassifier(loss='log', random_state=1, n_iter=1)

from nltk.corpus import stopwords
stop = stopwords.words('english')

def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv) # skip header
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

doc_stream = stream_docs(path='../data/movie_data.csv')

import pyprind # 进度条
pbar = pyprind.ProgBar(45)
import numpy as np
classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train: break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))