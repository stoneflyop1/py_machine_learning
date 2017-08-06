# 机器学习训练分类问题

## 人工神经元(Artificial neurons)，感知器

- 输入值向量x
- 激活函数(activation function)：若输入大于某个阈值$\theta$，则激活，返回1；否则返回-1。(Heaviside step function)
- 净输入(net input)：$z_i={w_i}*{x_i}$；若令$x_0=1$和$w_0=-\theta$

净输入的向量形式： $z=w^{T}x$，其中w为权重。

1. 初始化权重向量为0或很小的随机数字向量
1. 对每个训练样本，执行如下步骤：
    1. 计算输出值 $\overline{y}$
    1. 更新权重(根据误差进行反馈)

$$ w_j := w_j + \Delta{w_j} $$
$$ \Delta{w_j} = \eta ( y^{(i)}-{\overline{y}}^{(i)} ) x_j^{(i)} $$

其中$y^{(i)}$是真实的类标签值，${\overline{y}}^{(i)}$是预测的类标签值。

感知器的示例代码见：[perceptron.py](perceptron.py)

注意：感知器算法只能用于线性可分的情况，即：具有线性决策边界的情况。若线性不可分，则若没有给出最大的迭代次数(epochs)，权重更新部分将不会停止。

## 自适应线性神经元(Adaptive linear neurons, Adaline)

- 激活函数使用线性函数，使用净输入的值函数。
- 增加一个Quantizer函数用来最终输出。在Quantizer之前更新权重。

更新权重时，需要选取一个成本函数，目标是使得成本最小，一般可以使用梯度下降法。成本函数可以使用后平方误差和(Sum of Squared Errors, SSE)函数。

算法中出现的学习率参数$\eta$以及迭代次数参数(n_iter)被称为超参数(hyperparameter)。后面我们会看到，超参数也可以通过一些方式做调优。

根据所有样本更新权重的示例见：[adaline.py](adaline.py)。
梯度下降法可以引入随机元素，此时，我们可以仅根据当前样本更新权重。示例见：[adalinesgd.py](adalinesgd.py)。学习率也会使用一个递减的自适应学习率。

某些情况下，组合批量梯度下降(小样本情况)和随机梯度下降是更好的做法。