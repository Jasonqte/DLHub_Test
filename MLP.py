import torchvision
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib_inline
import torch.utils.data
import time
import numpy as np
import sys
from matplotlib_inline import backend_inline
import d2l
sys.path.append("..") # 为了导入上层目录的d2lzh_pytorch
##读取数据集
mnist_train = torchvision.datasets.FashionMNIST(root='/Users/renyiming/Neural\ Network/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='/Users/renyiming/Neural\ Network/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())
feature, label = mnist_train[0]
# print(feature, label)  #每个feature为一张图片

def use_svg_display():
    # 用矢量图显示
    matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

# X, y = [], []
# for i in range(10):
#     X.append(mnist_train[i][0])
#     y.append(mnist_train[i][1])
# show_fashion_mnist(X, get_fashion_mnist_labels(y))

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

batch_size = 256  #共256个样本,每个样本包含28*28个特征值
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
# for x,y in train_iter:
#     print(len(x),len(x[0]),len(y))
#     print(len(x[0][0]),y)
#     print(x,y)
#     break
num_inputs, num_outputs, num_hiddens = 784, 10, 256


from torch import nn
net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens, num_outputs),
        )
W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)
b1 = torch.zeros(num_hiddens, dtype=torch.float)
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)

loss = nn.CrossEntropyLoss()  ##定义损失函数
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=0.5)  ##选用SGD算法
print(optimizer)


num_epochs = 5
# for epoch in range(1, num_epochs + 1):
#     train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
#     for X, y in train_iter:
#         output = net(X)   ##output为预测标签矩阵
#         l = loss(output, y.view(-1, 1)).sum() ##输入为预测值和真实值
#         optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
#         l.backward()   ##求损失函数的梯度
#         optimizer.step()  ##求SGD
#
#         train_l_sum += l.item()
#         train_acc_sum += (output.argmax(dim=1) == y).sum().item()
#         n += y.shape[0]
#     test_acc = evaluate_accuracy(test_iter, net)
#     print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
#           % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))
#     # print('epoch %d, loss: %f' % (epoch, l.item()))
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            # if optimizer is not None:
            optimizer.zero_grad()
            # elif params is not None and params[0].grad is not None:
            #     for param in params:
            #         param.grad.data.zero_()

            l.backward()
            # if optimizer is None:
            #     d2l.sgd(params, lr, batch_size)
            # else:
            optimizer.step()  # “softmax回归的简洁实现”一节将用到


            train_l_sum += l.item()
            # train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f,  test acc %.3f'
              % (epoch + 1, train_l_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
