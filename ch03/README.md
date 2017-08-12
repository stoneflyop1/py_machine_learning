# Scikit-learn中的分类器导览

- 介绍流行的分类算法概念
- 使用scikit-learn的机器学习库
- 如何选择机器学习算法

## 选择一个分类算法

没有一个分类器在所有可能的情况下都是最好的选择(no single classifier works best across all possible scenarios)。实际应用中，都需要比较不同算法的性能以针对特定问题选择最好的模型；特征和样本数量、数据集的噪声，类别是否线性可分等都会影响分类器的选择。

机器学习算法训练的一般步骤：

1. 选择特征集合
1. 选择一个性能的度量标准(performance metric)
1. 选择分类器以及优化算法
1. 评估模型的性能
1. 微调(tuning)算法

### Logistic regression vs SVM

在实际的分类任务中，线性逻辑斯蒂回归和线性SVM通过会得到类似的结果。逻辑斯蒂回归尝试最大化训练数据的条件概率(conditional likelihoods of training data)。SVM主要关心位于决策边界(support vectors)附近的点。另一方面，逻辑斯蒂回归的模型更简单，因此更容易实现；此外，它也更容易更新模型，这对于流数据有很大的吸引力。

## 使用kernel SVM解决非线性问题

kernel方法的基本想法是：把线性不可分的数据使用一个对于特征的非线性的组合投影到一个高维的空间，从而使得高维数据线性可分。