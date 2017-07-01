import pandas as pd
from io import StringIO
import numpy as np
csv_data = ''' green,M,10.1,class1
            red,L,13.5,class2
            blue,XL,15.3,class1
            yellow,,5.3,class3'''
df = pd.read_csv(StringIO(csv_data))
# df = pd.DataFrame([
#     ['green', 'M',  10.1, 'class1'],
#     ['red',   'L',  13.5, 'class2'],
#     ['blue',  'XL', 15.3, 'class1'],
#     ['yellow', None, 15.3, 'class3']
# ])
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
