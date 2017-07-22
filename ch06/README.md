# 模型评估和超参数调整的学习最佳实践(Learning Best Practices for Model Evaluation and Hyperparameter Tuning)

- 获取模型性能的无偏估计
- 诊断机器学习算法的常见问题(common problems)
- 调整(fine-tune)机器学习模型
- 使用不同的性能度量(performance metrics)评估预测模型

## 使用管道形成工作流流水线(Streamlining workflows with pipelines)

可以用scilearn-kit中的Pipeline来组合工作流。示例代码见：[pipeline.py](pipeline.py)

1. 标准化或正则化(scale)             ---- transform
1. 降维(Dimensional Reduction)     ---- transform
1. 使用算法估计(Predictive Model)    ---- estimate

## 使用k方面的交叉验证(k-fold cross-validation)得到模型性能

若模型太简单，容易欠拟合(underfitting)，高偏差(high bias)；若模型太过复杂，容易过拟合(overfitting)，高方差(high variance)。

### holdout方法

把数据划分为训练数据、验证数据、测试数据。训练数据用来训练不同的模型，验证数据用来进行模型选择(model selection)，测试数据用来减少估计偏差。
此方法的缺点是对模型划分敏感。

### k方面交叉验证

把数据分成k份，其中k-1份用来训练模型，第k份用做测试数据。k越大，用来训练的数据越多，对于很大的数据集，k可以取小一些。

对k方面交叉验证方法的一个改进是分层k方面交叉验证(stratified k-fold cross-validation)，一般可以得到更好的偏差和方差估计，尤其是对于类别不成比例的情况。因为在每个方面，它都会保持类别的比例。

## 通过学习曲线和验证曲线调试算法

- 通过学习曲线诊断过拟合(高方差)或欠拟合(高偏差)问题。
- 通过验证曲线给出学习算法的常见问题。

代码示例见：[curves.py](curves.py)。

注：scikit-learn中的`learing_curve`和`validation_curve`中交叉验证默认使用的都是`stratified k-fold cross-validation`。

### 学习曲线(learing curves)

高偏差(High Bias)说明训练和交叉验证的精确度太低，模型欠拟合。一般的改进做法是：增加模型的参数数量，比如 收集或构造其他特征，或减少规则化(regularization)的维度(如：SVM或LR分类器)。

高方差(High Variance)说明训练和交叉验证之间有很大的精度偏差，模型过拟合。改进做法是：收集更多的训练数据或减少模型的复杂度，通过特征选择或特征提取减少特征数量也很有帮助。

学习曲线的横轴是训练数据样本的数量，纵轴是测试数据的精确度。

### 验证曲线(validation curves)

与学习曲线类似，验证曲线也可以很容易的发现过拟合和欠拟合现象。与学习曲线不同，验证曲线的横轴是模型的参数，比如：LR中的反规范化参数C(inverse regularization parameter C)。

## 采用网格搜索(grid search)微调模型

机器学习中有两种类型的参数，一种参数用来从训练数据中学习，比如：LR中的权重，还有一种参数用来做独立的优化。后一种参数是调整参数(tuning parameters)，也叫做超参数(hyperparameters)，比如：LR中的规范化参数(regulation parameter)，或者决策树中的深度参数(depth parameter)

网格搜索是一种超参数优化技术，通过优化超参数值的组合，进一步优化模型的性能。

网格搜索还可以执行嵌套的交叉验证(nested cross-validation)，可以很方便的选择性能更好的算法。

示例代码见：[grid.py](grid.py)