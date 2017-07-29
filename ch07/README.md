# 组合不同的模型学习(Combining Different Models for Ensemble Learning)

- 基于多数表决(majority voting)的预测
- 通过随机重复组合训练集来降低过拟合
- 从错误中学习的弱学习者(weak learner)生成强大的模型

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