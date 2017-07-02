# 生成好的训练数据集 —— 数据前处理

- 删除或补全缺失的数据(missing values from the dataset)
- 为机器学习算法获取成型的分类数据(get categorical data into shape)
- 为模型建造选择相关特征(relevant features)


## 缺失的数据

实际中得到的数据经常会有数据不全的情况，比如为空字符串，NaN等。大部分的计算工具对这种情况都可能会产生无法预计的结果，如果我们仅仅是忽略这些数据的话。因为，如何处理缺失的数据对进一步的数据分析是至关重要的。

可以使用pandas库查询数据的缺失情况，具体见：[miss.py](miss.py)。

### 清理丢失数据的样本(行)或特性(列)

可以使用pandas的dataframe的dropna方法。

### 输入缺失的数据

一般地，可以使用数据插值方式补全数据。常用的插值技术是：均值替换法(mean imputation)。其方法是把整列数据取平均来补全缺失的值。除了整列平均外，还有中值(median)以及出现最多(most_frequent)两种策略，其中`most_frequent`常用在补全分类特性值的情况。

## 处理分类数据(categorical data)

特性可以分为命名特征(nominal)以及有序特征(ordinal)。有序特征的值如果是字符串，则可以通过自定义映射变为数值。

代码示例见：[preprocess.py](preprocess.py)

### 映射有序特征(Mapping ordinal features)

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

### 对命名特征(nominal feature)实行one-hot编码

命名特征也可以使用上文提到的`LabelEncoder`进行到数值的映射。不过这样映射的值会形成天然的有序关系，而原本的命名特征其实没有这种关系。一种变通的方法(workaround)就是使用one-hot编码。

one-hot编码的想法是：对于命名特征的每个唯一值都创建一个虚拟的特征(dummy feature)。比如：对于颜色特征的blue,green,red。可以创建三个分别为blue,green,red的虚拟特征。

### 分离数据集为训练集和测试集(Partitioning a dataset in traing and test sets)

使用开源的数据集[Wine](https://archive.ics.uci.edu/ml/datasets/Wine)。本仓库中的wine.data以及wine.names就来自于此开源数据集。

一般地，训练和测试用的数据集比例为：60:40, 70:30，或80:20；主要看数据集的大小，若数据集非常大，使用90:10，甚至是99:1都是可能的。

### 数据的正则化(Bringing features onto the same scale)

特征缩放(feature scaling)是一个关键的预处理步骤，但很容易被忽略。只有极个别的算法，比如：决策树、随机森林，不需要进行特征缩放；大部分的其他机器学习算法以及优化算法在进行特征缩放后都会有很大的提升。

- 正则化(__normalization__)，把数据缩放到[0,1]区间。可以使用`sklearn.preprocessing.MinMaxScaler`。
- 标准化(__standardization__)，转换为均值(mean)为0，标准差(standard deviation)为1的特征。可以使用`sklearn.preprocessing.StandardScaler`

如下是0~5作为样本数据集输入的标准化和正则化的数值比较：

| 输入 | 标准化 | 正则化 |
| ---- | ---- | ---- |
| 0.0 | -1.336306 | 0.0 |
| 1.0 | -0.801784 | 0.2 |
| 2.0 | -0.267261 | 0.4 |
| 3.0 | 0.267261 | 0.6 |
| 4.0 | 0.801784 | 0.8 |
| 5.0 | 1.336306 | 1.0 |

注：正则化一般用来限制特性值中的值范围，标准化更实用。

## 使用有意义的数据优化模型

### 选择有意义的特征

如果训练的模型在训练数据上的表现远远好于测试数据上的表现，则表明模型过拟合(`overfitting`)了。为了降低过拟合的程度，一些常用方法如下：

- 收集更多的训练数据
- 针对复杂性，通过规范化(regularization)引入阈值(penalty)
- 选择一个较少参数的简单模型
- 对数据进行降维处理