# 生成好的训练数据集 —— 数据前处理

- 删除或补全缺失的数据(missing values from the dataset)
- 为机器学习算法获取成型的分类数据(get categorical data into shape)
- 为模型建造选择相关特性(relevant features)


## 缺失的数据

实际中得到的数据经常会有数据不全的情况，比如为空字符串，NaN等。大部分的计算工具对这种情况都可能会产生无法预计的结果，如果我们仅仅是忽略这些数据的话。因为，如何处理缺失的数据对进一步的数据分析是至关重要的。

可以使用pandas库查询数据的缺失情况，具体见：[miss.py](miss.py)。

### 清理丢失数据的样本(行)或特性(列)

可以使用pandas的dataframe的dropna方法。

### 输入缺失的数据

一般地，可以使用数据插值方式补全数据。常用的插值技术是：均值替换法(mean imputation)。其方法是把整列数据取平均来补全缺失的值。除了整列平均外，还有中值(median)以及出现最多(most_frequent)两种策略，其中`most_frequent`常用在补全分类特性值的情况。

## 处理分类数据(categorical data)

特性可以分为命名特性(nominal)以及有序特性(ordinal)。

代码示例见：[preprocess.py](preprocess.py)

### 映射有序特性(Mapping ordinal features)

我们通常把有序特性的数据映射到整数类型数据上，若衣服的大小号虽然用字符串标识，但在使用时，我们一般按照大小号的顺序映射到整型数值。

### 对类别标签进行编码(Encoding class labels)

很多机器学习库都要求分类标签被编码为整数。尽管scikit-learn中的很多分类器都会内部转换类标签为整数，我们也建议自己提供此映射，或者使用scikit-learn中的`LabelEncoder`。

```python
>>> import pandas as pd
>>> df = pd.DataFrame([
    ['green', 'M',  10.1, 'class1'],
    ['red',   'L',  13.5, 'class2'],
    ['blue',  'XL', 15.3, 'class1']
])
df.columns = ['color', 'size', 'price', 'classlabel']
>>> from sklearn.preprocessing import LabelEncoder
>>> class_le = LabelEncoder()
>>> y = class_le.fit_transform(df['classlabel'].values)
>>> y
array([0, 1, 0])
```