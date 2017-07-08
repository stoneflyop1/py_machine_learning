# 通过降维压缩数据(Compressing Data via Dimensionality Reduction)

- 对于非监督的数据压缩的主成分分析法(Pricipal component analysis, PCA)
- 作为最大化类别区分的监督降维技术的线性判别分析(Linear Discriminant Analysis, LDA)
- 通过内核主成分分析(kernel principal component analysis)的非线性降维

## 主成分分析

目标是在高维数据中找到最大化方差的方位，并把数据投射到一个新的、具有相等或更少维度的子空间上。

样本空间维度为d，新的降维后的特征子空间维度为k。构造一个dxk维的转换矩阵W，把d维的样本向量x投射到新的k维向量z。

1. 把d维数据集标准化
1. 构造协方差矩阵(covariance matrix)
1. 分解协方差矩阵为特征向量(eigenvectors)和特征值(eigenvalues)
1. 选择k个最大特征值对应的特征向量作为主成分(principal components)
1. 使用主要的k个特征向量构造一个投射矩阵W
1. 使用W变换d维的输入数据集到k维的新的特征子空间

> 尽管numpy.linalg.eig函数被设计为分解非对称的方阵，在某些情况下他会得到复数特征值。一个相关的函数numpy.linalg.eigh被实现为分解厄米特矩阵(Hermetian matix)，它对于对称阵在数值上更稳定。比如：对于协方差矩阵，它可以保证都返回实数特征值。

scikit-learn中有PCA模块。

```python
from sklearn.decomposition import PCA
```
