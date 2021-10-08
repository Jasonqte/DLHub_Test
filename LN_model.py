import matplotlib_inline
import torch
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import  backend_inline

import numpy as np
import random

num_inputs = 2  ##特征数
num_examples = 1000  ##样本数
true_w = [2, -3.4]   ##weight
true_b = 4.2   ##bias
features = torch.randn(num_examples, num_inputs,
                       dtype=torch.float32)   ##特征矩阵
# print(features[:, 0],true_w[0] * features[:, 0])
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  ##标签（结果值）矩阵
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),   ##np.random.normal 为正态分布输出为函数抽取出的样本
                       dtype=torch.float32)

# print(labels)  ##此时labels为最终的标签矩阵


# def use_svg_display():
#     # 用矢量图显示
#     matplotlib_inline.backend_inline.set_matplotlib_formats('svg')
# #
# def set_figsize(figsize=(3.5, 2.5)):
#     use_svg_display()
#     # 设置图的尺寸
#     plt.rcParams['figure.figsize'] = figsize
# #
# # # # 在../d2lzh_pytorch里面添加上面两个函数后就可以这样导入
# # # import sys
# # # sys.path.append("..")
# # # from d2lzh_pytorch import *
# #
# set_figsize()
# plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()    ##有图可看出特征值与标签值成线性关系


#加载数据
import torch.utils.data as Data

batch_size = 10   ##每批加载的样本数
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# print(dataset)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)

for X, y in data_iter:
    print(X, y)
    break

#
#定义模型
from torch import nn
# class LinearNet(nn.Module):
#     def __init__(self, n_feature):
#         super(LinearNet, self).__init__()
#         self.linear = nn.Linear(n_feature, 1)
#     # forward 定义前向传播
#     def forward(self, x):
#         y = self.linear(x)
#         return y
#
# net = LinearNet(num_inputs)  ##定义线性网络
# print(net.children()) # 使用print可以打印出网络的结构



#可以向nn.sequential容器中添加大量网络
# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )
#
# # 写法二
# net = nn.Sequential()
# net.add_module('linear', nn.Linear(num_inputs, 1))
# # net.add_module ......
#
# # 写法三
# from collections import OrderedDict
# net = nn.Sequential(OrderedDict([
#           ('linear', nn.Linear(num_inputs, 1))
#           # ......
#         ]))
#
# print(net)
print(net[0])
for param in net.parameters():
    print(param)
#初始化网络
from torch.nn import init

init.normal_(net[0].weight, mean=0, std=0.01)  #权重初始化为均值为0，标准差为0.01的正态随机数
init.constant_(net[0].bias, val=0)  # bias初始化为0


loss = nn.MSELoss()  ##定义损失函数


#定义优化器
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.3)  ##选用SGD算法
print(optimizer)


num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)   ##output为预测标签矩阵
        l = loss(output, y.view(-1, 1)) ##输入为预测值和真实值
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()   ##求损失函数的梯度
        optimizer.step()  ##求SGD
    print('epoch %d, loss: %f' % (epoch, l.item()))


dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)