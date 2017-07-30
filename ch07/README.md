# 组合不同的模型学习(Combining Different Models for Ensemble Learning)

- 基于多数表决(majority voting)的预测
- 通过随机重复组合训练集来降低过拟合
- 从错误中学习的弱分类器(weak learner)生成强大的模型

## 组合学习(Learning with ensembles)

组合方法(ensemble methods)是组合不同的分类器为一个元分类器，一个一般情况下都具有更好性能的分类器。本章主要关注使用多数表决原理的组合方法，对于多类的情况，可以扩展为plurality voting。

为什么组合方法会工作的更好呢？

以二分分类(binary classification)任务为例，假设所有的n个基础分类器都有相同的错误率(error rate) $\epsilon$。此外，假设分类器是相互独立的，且错误率是不相关的。那么我们可以得到出错的概率(二项分布的概率质量函数)如下：

$$ P(y \geq k) = \sum \limits_{k}^{n}{\left( \begin{array}{c} n \\ k \end{array} \right)\epsilon^k (1-\epsilon)^{n-k}}=\epsilon_{ensemble} $$

对比见代码示例[enerror.py](enerror.py)的结果。

## 实现一个简单的多数表决分类器

实现了一个可以带权重的分类器，表决支持类标记(classlabel)以及概率(probability)，代码见：[mvclassifier.py](mvclassifier.py)。

## Bagging - 从自举样本(bootstrap samples)构建分类器组合

Bagging也被称为自举汇聚法(bootstrap aggregating)，与有放回抽样类似。

随机森林方法可以认为是Bagging的一个特例。

示例代码见：[bagging.py](bagging.py)。

## 通过自适应增强(adaptive boosting)改进(leverage)弱分类器

原始的boosting算法：

1. 从训练样本中随机选取一个训练子集(无放回)d1来训练一个弱分类器C1
1. 随机选择第二个训练子集d2，并且把上一次分类错误的样本的50%加入到此训练集并训练第二个弱分类器C2
1. 再次选择一个训练集d3，包含C1和C2错误分类的样本，来训练弱分类器C3
1. 通过多数表决组合弱分类器C1、C2和C3

boosting算法从理论上可以降低偏差和方差，与bagging模型比较的话。但在实践中，经常会出现高方差的情况，即：容易过拟合。

与原始的boosting算法不同，自适应的boosting算法使用完整的训练集训练弱分类器，只不过每次训练弱分类器时，会对训练样本重新设置权重，通过不断的对上一次的分类情况进行改进。