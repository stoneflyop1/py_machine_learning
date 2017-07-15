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

