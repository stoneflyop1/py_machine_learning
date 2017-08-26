# 训练人工神经网络进行图像识别(Image Recognition)

- 了解多层神经网络的概念
- 训练神经网络进行图片分类
- 实现强大的后向传播算法(backpropagation algorithm)
- 调试神经网络的实现

## 多层神经网络结构简介

本章节会介绍如何把多个单一神经元连接为一个多层的前馈神经网络(multi-layer feedforward neural network)，也叫做多层感知器(multi-layer perceptron, MLP)。

> 误差梯度在层数不断增加时会变得非常之小。这个梯度消失(vanishing gradient)问题使得模型学习更加具有挑战性。因此，一些专门的算法被开发出来预训练(pretrain)深度神经网络结构，算法叫做深度学习(deep learning)。

## 通过前向传播(forward propagation)激活一个神经网络

1. 从输入层(input layer)开始，把训练数据的模式(patterns)通过网络前向传播来生成一个输出
1. 基于网络的输出，计算使得成本函数(cost function)最小化的误差
1. 反向传播误差，在网络中对权重求导，并更新模型

注：神经网络中的激活函数(activation)一般使用sigmoid函数，而不是线性函数。

## 手写数字分类问题

数据获取：[MNIST](http://yann.lecun.com/exdb/mnist/)。

手写图片的像素：28x28 = 784