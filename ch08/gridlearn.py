from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import tokendata

def learn(df):
    X_train = df.loc[:25000, 'review'].values
    y_train = df.loc[:25000, 'sentiment'].values
    X_test = df.loc[25000:, 'review'].values
    y_test = df.loc[25000:, 'sentiment'].values

    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)

    param_grid = [
        {
            'vect__ngram_range': [(1,1)],
            'vect__stop_words': [tokendata.stop, None],
            'vect__tokenizer': [tokendata.tokenizer, tokendata.tokenizer_porter],
            'clf__penalty': ['l1', 'l2'],
            'clf__C': [1.0, 10.0, 100.0]
        },
        {
            'vect__ngram_range': [(1,1)],
            'vect__stop_words': [tokendata.stop, None],
            'vect__tokenizer': [tokendata.tokenizer, tokendata.tokenizer_porter],
            'vect__use_idf': [False],
            'vect__norm': [None],
            'clf__penalty': ['l1', 'l2'],
            'clf__C': [1.0, 10.0, 100.0]
        }
    ]
    lr_tfidf = Pipeline(
        [('vect', tfidf), ('clf', LogisticRegression(random_state=0))]
    )
    gs_lr_tfidf = GridSearchCV(
        lr_tfidf, param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1
    )
    gs_lr_tfidf.fit(X_train, y_train)
    print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
    print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
    clf = gs_lr_tfidf.best_estimator_
    print('Test Accuracy: %.3f' % clf.score(X_test, y_test))