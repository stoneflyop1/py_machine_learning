import pandas as pd
from io import StringIO

csv_data = '''A,B,C,D
            1.0,2.0,3.0,4.0
            5.0,6.0,,8.0
            10.0,11.0,12.0,'''
## for python2
# csv_data = unicode(csv_data)
df = pd.read_csv(StringIO(csv_data))
print('#'*60)
print('# show data with missing values')
print(df)
print('# isnull: ', df.isnull().sum()) # convert to boolean false values using dataframe isnull method
print(df.values) # get numpy array from dataframe values
print('#'*60)
print('# drop NaN row or col')
print('########## drop row:\r\n', df.dropna()) # axis=0
print('########## drop col:\r\n', df.dropna(axis=1))
print('########## drop all col is NaN:\r\n', df.dropna(how='all'))
print('########## drop with threshold:\r\n', df.dropna(thresh=4))
print('########## drop specific cols:\r\n', df.dropna(subset=['C']))
print('#'*60)
print('# mean impute missing values')
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values="NaN", strategy="mean", axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data)
