import pandas as pd


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def get_wine_data():
    df_wine = pd.read_csv('../data/wine.data', header=None)
    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                test_size=0.3, random_state=0)
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    return (X_train, X_train_std,  X_test_std, y_train, y_test )
