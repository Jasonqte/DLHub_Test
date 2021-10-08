import torch
from torch import nn

#卷积计算规则定义类（卷积运算后+偏差）
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super(Conv2D, self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))
        # print(self.weight,self.bias)

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias


def corr2d(X, K):  #卷积核计算函数
    h, w = K.shape #输出数组的行数和列数
    # print(h,w)
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    # print(Y)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

# X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
# K = torch.tensor([[0, 1], [2, 3]])
# print(corr2d(X, K))

# X = torch.ones(6, 8)
# X[:, 2:6] = 0
# K = torch.tensor([[1, -1]])
# Y = corr2d(X, K)

#卷积层运算模拟
# conv2d = Conv2D(kernel_size=(1, 2)) #kernel_size定义卷积核大小（1行2列）
# step = 20
# lr = 0.01
# for i in range(step):
#     Y_hat = conv2d(X) #卷积预测值
#     l = ((Y_hat - Y) ** 2).sum()
#     l.backward()
# #
# #     # 梯度下降
#     conv2d.weight.data -= lr * conv2d.weight.grad
#     conv2d.bias.data -= lr * conv2d.bias.grad
# #
# #     # 梯度清0
#     conv2d.weight.grad.fill_(0)
#     conv2d.bias.grad.fill_(0)
#     if (i + 1) % 5 == 0:
#         print('Step %d, loss %.3f' % (i + 1, l.item()))


# #填充和步长（超参）
# import torch
# from torch import nn
#
# # 定义一个函数来计算卷积层。它对输入和输出做相应的升维和降维
# def comp_conv2d(conv2d, X):
#     # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1
#     X = X.view((1, 1) + X.shape)
#     print(X)
#     Y = conv2d(X)
#     print(Y,Y.view(Y.shape[2:]))
#     return Y.view(Y.shape[2:])  # 排除不关心的前两维：批量和通道
#
# # 注意这里是两侧分别填充1行或列，所以在两侧一共填充2行或列
# conv2d = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
#
# X = torch.rand(8, 8)
# comp_conv2d(conv2d, X).shape


#多通道输入/输出
import torch
from torch import nn
import sys
sys.path.append("..")
import d2l

def corr2d_multi_in(X, K): #多通道输入求卷积层
    # 沿着X和K的第0维（通道维）分别计算再相加
    res = d2l.corr2d(X[0, :, :], K[0, :, :])
    for i in range(1, X.shape[0]):
        res += d2l.corr2d(X[i, :, :], K[i, :, :])
    return res

def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输入X做互相关计算。所有结果使用stack函数合并在一起
    return torch.stack([corr2d_multi_in(X, k) for k in K])
K = torch.tensor([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])
# print(K+1)
K = torch.stack([K, K + 1, K + 2]) #构造多卷积核
print(K)
print(K.shape) # torch.Size([3, 2, 2, 2])