# 使用回归分析(regression analysis)预测连续目标变量

- 探索和可视化数据集
- 不同方法实现线性回归模型
- 训练外围鲁棒的回归模型
- 评估回归模型以及诊断常见问题
- 针对非线性数据拟合回归模型

## 线性回归模型

$$ y = w_0 + {w_1}{x_1} + ... + {w_m}{x_m} = {w^T}x $$

其中 y为目标变量， $w_0$是目标变量上的截距(intercept)，$x_1 ... x_m$为解释变量(explanatory varialbes)。

### 使用RANSAC拟合一个鲁棒的回归模型

RANdom SAmple Consensus(RANSAC)算法是用数据的一个子集，叫做内部数据(inliers)来拟合回归模型，扔掉了外围的数据(outliers)。

1. 选择随机数目的样本作为内部数据拟合模型
1. 根据拟合的模型测试所有其他数据点，若可以作为内部数据则添加到内部数据中
1. 使用所有内部数据重新拟合模型
1. 评估内部数据对拟合模型的误差
1. 若拟合模型的性能位于用户定义的阈值内或达到了固定的迭代次数，终止程序；否则回到第一步

### 评估线性回归模型的性能

- 划分样本为训练数据和测试数据
- Mean Squared Error(MSE)
- Standardized version of MSE ($R^2$)

### 回归中使用正规化方法(regularized methods)

正规化(regularization)是用来处理过拟合的方法，它会添加一些额外的信息，从而缩小了模型参数范围来诱发一个针对复杂性的补偿(induce a penalty against complexity)。常用的正规化线性回归分析有：

- 脊回归(Ridge Regression)，一个L2补偿模型(L2 penalized model)，直接在cost function上加上权重的L2函数(带有超参数$\lambda$)

    ```python
    from sklearn.linear_model import Ridge
    ridge = Ridge(alpha=1.0)
    ```
- 最小绝对收缩和选择算子(Least Absolute Shrinkage and Selection Operator, LASSO), L1

    ```python
    from sklearn.linear_model import Lasso
    lasso = Lasso(alpha=1.0)
    ```
- 弹性网(Elastic Net), L1 and L2

    ```python
    from sklearn.linear_model import ElasticNet
    elnet = ElasticNet(alpha=1.0, l1_ratio=0.5)
    ```

## 多项式回归

$$ y = {w_0}+{w_1}x+{w_2}{x^2}{x^2}+...+{w_d}x^d $$

简单的示例参见：[poly.py](poly.py)

machine的示例见：[machine_nonlinear.py](machine_nonlinear.py)