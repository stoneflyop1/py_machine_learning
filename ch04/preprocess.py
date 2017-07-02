import pandas as pd
from io import StringIO
import numpy as np
csv_data = ''' green,M,10.1,class1
            red,L,13.5,class2
            blue,XL,15.3,class1
            yellow,,5.3,class3'''
# df = pd.read_csv(StringIO(csv_data))
df = pd.DataFrame([
    ['green', 'M',  10.1, 'class1'],
    ['red',   'L',  13.5, 'class2'],
    ['blue',  'XL', 15.3, 'class1']
])
df.columns = ['color', 'size', 'price', 'classlabel']

# 类标签映射到整数

class_mapping = {label:idx for idx, label in enumerate(np.unique(df['classlabel']))}
print('class_mapping:', class_mapping)
print(df['classlabel']) # Name: classlabel, dtype: object
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df['classlabel']) # Name: classlabel, dtype: int64
print('after encode classlabel:\r\n', df)
print(df['size'])
# use astype fix TypeError: '>' not supported between instances of 'float' and 'str'
print(np.unique(df['size'].astype(np.dtype(str), errors='ignore')))
print(df)
# 有序特性映射到整数数据
size_mapping = {
    'XL': 3,
    'L': 2,
    'M': 1
}
df['size'] = df['size'].map(size_mapping)
print('after mapping ordinal feature size:\r\n', df)
print(df)
inv_size_mapping = {v: k for k, v in size_mapping.items()}
print(inv_size_mapping)
# print('color', df['color'].dtype == np.dtype(object))
#### 合并两个ndarray
#print(np.concatenate((np.unique(df['classlabel']), np.unique(df['color']))))

from sklearn.preprocessing import LabelEncoder
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:,0])
print(X)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
print(ohe.fit_transform(X).toarray())

print(pd.get_dummies(df[['price', 'color', 'size']]))

# Split Wine dataset https://archive.ics.uci.edu/ml/datasets/Wine/wine.data
print('Start deal with wine.data: https://archive.ics.uci.edu/ml/datasets/Wine/wine.data')
df_wine = pd.read_csv('wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol',
                    'Malic acid', 'Ash',
                    'Alcaliniity of ash', 'Magnesium',
                    'Total phenols', 'Flavanoids',
                    'Nonflavanoid phenols', 'Proanthocyanins',
                    'Color intensity', 'Hue',
                    'OD280/OD315 of diluted wines', 'Proline']
print('Class labels', np.unique(df_wine['Class label']))
print(df_wine.head())
from sklearn.model_selection import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

print('############## Normalization using MinMaxScaler')
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)
print(X_train_norm)
print(X_test_norm)
print('############## Standardization using StandardScaler')
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
print(X_train_std)
print(X_test_std)