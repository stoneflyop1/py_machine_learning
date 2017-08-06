from preparedata import *

import pickle
import os
dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)
stopwords_file = os.path.join(dest, 'stopwords.pkl')
if not os.path.exists(stopwords_file):
    pickle.dump(stop, open(stopwords_file, 'wb'), protocol=4)
clf_file = os.path.join(dest, 'classifier.pkl')
if not os.path.exists(clf_file):
    pickle.dump(clf, open(clf_file, 'wb'), protocol=4)