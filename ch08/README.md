# 使用机器学习进行情感分析(Sentiment Analysis)

- 清理和准备文本数据
- 从文本文档中构造特征向量(feature vectors)
- 训练一个机器学习模型来对电影评论进行评分(+/-)
- 使用out-of-core学习处理大型的文本数据集

## 获取数据集

数据集来自于：[Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

代码见：[readdata.py](readdata.py)。

## bag-of-words模型

bag-of-words模型使得我们可以使用数值的特征向量表示文本。

1. 创建一个唯一标识的词汇集合。
1. 根据词(word)在文档中出现的次数构造一个特征向量。

注：特征向量一般比较稀疏(sparse)。

## 变换词为特征向量

总的代码示例见：[simplesample.py](simplesample.py)

使用sklearn中的CountVectorizer根据词频抽取文本的特征。

```python
>>> import numpy as np
>>> from sklearn.feature_extraction.text import CountVectorizer
>>> count = CountVectorizer()
>>> docs = np.array([
... 'The sun is shining',
... 'The weather is sweet',
... 'The sun is shining and the weather is sweet'])
>>> bag = count.fit_transform(docs)
>>> count.vocabulary_
{'the': 5, 'sun': 3, 'is': 1, 'shining': 2, 'weather': 6, 'sweet': 4, 'and': 0}
>>> print(bag.toarray())
[[0 1 1 1 0 1 0]
 [0 1 0 0 1 1 1]
 [1 2 1 1 1 2 1]]
```

- (raw) term frequencies: tf(t, d)：t表示term，d表示document
- term frequency-inverse document frequency：tf-idf(t,d)
- inverse document frequency：idf(t, d)

$$ (tf-idf)(t,d) = tf(t,d) x idf(t,d) $$

$$ idf(t,d) = log\frac{n_d}{1+df(d,t)} $$

其中 $n_d$表示文档的总数，$df(d,t)$表示含有t的文档数量。分母上的数字1是可选的，它使得所有训练样本的term都是非0值；而取对数(log)则使得很低的文档频率不会导致很大的权重。

```python
>>> from sklearn.feature_extraction.text import TfidfTransformer
>>> tfidf = TfidfTransformer(norm='l2')
>>> np.set_printoptions(precision=2)
>>> print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
[[ 0.    0.43  0.56  0.56  0.    0.43  0.  ]
 [ 0.    0.43  0.    0.    0.56  0.43  0.56]
 [ 0.4   0.48  0.31  0.31  0.31  0.48  0.31]]
```

sklearn中实现的idf(t,d)为：

$$ idf(t,d) = log\frac{1+n_d}{1+df(d,t)} $$

$$ (tf-idf)(t,d) = tf(t,d) x (idf(t,d)+1) $$