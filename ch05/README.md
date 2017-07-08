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

## 线性判别分析(LDA)

PCA是通过找到正交的特征并且保证方差最大化的方式进行降维。LDA也类似，它是要找一个特征子空间使得类别尽可能的区分开。它们之间的主要区别是：LDA是一种监督算法。

LDA的假设：

- 数据是正态分布的(normally distriubuted)
- 分类具有相同的协方差矩阵
- 特征在统计意义上相互独立

注：即使某些假设不满足，LDA可能依然表现很好。

步骤：

1. 标准化d维数据集
1. 对每一个类别，计算d维的平均向量
1. 构造`between-class`散点矩阵(scatter matrix) $S_B$ ，以及`within-class`散点矩阵 $S_w$
1. 计算 ${S_w}^{-1}S_B$ 的特征向量和特征值
1. 选择k个特征构造 $d \times k$ 维的变换矩阵W；特征向量作为W的列
1. 根据变换矩阵W，投射样本到新的特征子空间

### 计算散点矩阵

针对不同类别结果对样本进行平均。如下公式中的i表示第i个类别。

平均向量公式：

$$ m_i = \frac{1}{n_i} \sum\limits_{x \in D_i}^{c} x_m $$

within-class散点矩阵公式：

$$ S_W = \sum\limits_{i=1}^{c} S_i $$
$$ S_i = \sum\limits_{x \in D_i}^{c} {(x-m_i)(x-m_i)^T} $$

考虑到类别的值可能不是均匀分布的，需要对散点矩阵进行缩放(我们发现，缩放后的散点矩阵其实就是协方差矩阵)：

$$ \Sigma_i = \frac{1}{N_i}S_W = \frac{1}{N_i} \sum\limits_{x \in D_i}^{c} (x-m_i)(x-m_i)^T $$

between-class散点矩阵公式(m为整个的均值，包括所有类别的样本)：

$$ S_B = \sum\limits_{i=1}^{c} {N_i (m_i-m)(m_i-m)^T}  $$