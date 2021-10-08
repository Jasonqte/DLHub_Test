import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch import distributions as distrib
from torch.distributions import transforms as dist_transforms

# torch.manual_seed(1)
batch_size = 512

# Define grids of points (for later plots)
x = np.linspace(-5, 5, 1000) #生成数据数量为1000,上下限为-5，5的样本
# print(np.array(np.meshgrid(x, x)))
z = np.array(np.meshgrid(x, x)).transpose(1, 2, 0)  # 默认transpose(0,1,2)  当前状态表示换轴即xyz变为yzx
#最终生成（1000，1000，2）的三维矩阵
z = np.reshape(z, [z.shape[0] * z.shape[1], -1])  # 生成(1000*1000, 2)的二维矩阵
x2_dist = distrib.Normal(loc=0., scale=4.) #loc均值，scale标准差，创建以上两个参数的正太分布
x2_samples = x2_dist.sample((batch_size,))  #高斯分布随机采样
x1_dist = distrib.Normal(loc=0.25*torch.mul(x2_samples, x2_samples), scale=torch.ones((batch_size,)))
#拟合x1分布
x1_samples = x1_dist.sample()  # (512,)
samples = torch.stack([x1_samples, x2_samples], dim=1) #连接x1，x2采样
plt.scatter(samples[:, 0], samples[:, 1], s=10, color='red')
plt.xlim([-5, 30])
plt.ylim([-10, 10])
# plt.show()

def target_density(z): #求样本在不同密度函数下的目标概率密度
    pi = torch.Tensor([math.pi])
    z1, z2 = torch.chunk(z, chunks=2, dim=1)
    # print(z1,z2)
    part1 = 1. / torch.sqrt(2 * pi) * torch.exp(-torch.mul(z1 - 0.25*z2*z2, z1 - 0.25*z2*z2) / 2.)
    part2 = 1. / 4 * torch.sqrt(2 * pi) * torch.exp(-torch.mul(z2, z2) / 32.)

    return part1*part2

# Plot it
# print(z) #z为样本全集
plt.hexbin(z[:,0], z[:,1], C=target_density(torch.Tensor(z)).numpy().squeeze(), cmap='rainbow')
plt.title('Target density', fontsize=18)


# Base class
class Flow(dist_transforms.Transform, nn.Module):
    def __init__(self):
        dist_transforms.Transform.__init__(self)
        nn.Module.__init__(self)

    def init_parameters(self):
        for param in self.parameters():
            param.data.normal_(0, 0.01)

    def __hash__(self):
        return nn.Module.__hash__(self)

# LeakyReLU
class PReLUFlow(Flow):
    def __init__(self, dim):
        super(PReLUFlow, self).__init__()
        self.alpha = nn.Parameter(torch.randn([1]))
        self.domain = torch.distributions.constraints.Constraint()
        self.codomain = torch.distributions.constraints.Constraint()

    def _call(self, z):
        return torch.where(z >= 0, z, torch.abs(self.alpha) * z)

    def _inverse(self, x):
        return torch.where(x >= 0, x, torch.abs(1. / self.alpha) * x)

    def log_abs_det_jacobian(self, z):
        I = torch.ones_like(z)
        J = torch.where(z >= 0, I, self.alpha * I)
        log_abs_det = torch.log(torch.abs(J) + 1e-5)
        return torch.sum(log_abs_det, dim=1)


# Affine function
class AffineFlow(Flow):
    def __init__(self, dim):
        super(AffineFlow, self).__init__()
        self.domain = torch.distributions.constraints.Constraint()
        self.codomain = torch.distributions.constraints.Constraint()
        self.weight = nn.Parameter(torch.randn(dim, dim))
        self.shift = nn.Parameter(torch.randn(dim, ))
        nn.init.orthogonal_(self.weight)

    def _call(self, z):
        return self.shift + z @ self.weight

    def _inverse(self, x):
        return (x - self.shift) @ torch.inverse(self.weight)

    def log_abs_det_jacobian(self, z):
        return torch.slogdet(self.weight)[-1].unsqueeze(0).repeat(z.size(0), 1).squeeze()


# Normalizing flows
class NormalizingFlow(nn.Module):
    def __init__(self, dim, blocks, num_layers, base_density):
        super().__init__()
        bijectors = []
        for n in range(num_layers):
            for b in blocks:
                bijectors.append(b(dim))

        bijectors.pop()  # Remove the last ReLU block
        self.transforms = dist_transforms.ComposeTransform(bijectors)
        self.bijectors = nn.ModuleList(bijectors)
        self.base_density = base_density
        self.proj_density = distrib.TransformedDistribution(base_density, self.transforms)
        self.log_abs_det_jacobians = []

    def get_fz(self, z):
        # Get the outputs of medium layers
        fz = []
        fz.append(z)
        for i in range(len(self.bijectors)):
            z = self.bijectors[i](z)
            fz.append(z)

        return fz

    def forward(self, z):
        self.log_abs_det_jacobians = []
        for i in range(len(self.bijectors)):
            ladj = self.bijectors[i].log_abs_det_jacobian(z)
            self.log_abs_det_jacobians.append(ladj)
            z = self.bijectors[i](z)
        return z, self.log_abs_det_jacobians

my_blocks = [
    AffineFlow,
    PReLUFlow
    # Flow
]
base_density = distrib.MultivariateNormal(torch.zeros(2), torch.eye(2))  #建立基础概率密度为多元高斯分布
nf = NormalizingFlow(dim=2, blocks=my_blocks, num_layers=6, base_density=base_density)
# print(nf.bijectors)

sources = base_density.sample((512, ))
# print(sources)
fz = nf.get_fz(sources)
print(fz)
f, arr = plt.subplots(1, len(fz), figsize=(4 * (len(fz)), 4))
X0 = fz[0].detach().numpy()
for i in range(len(fz)):
    X1 = fz[i].detach().numpy()
    idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
    arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='red')
    idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
    arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='green')
    idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
    arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue')
    idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
    arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='black')
    arr[i].set_xlim([-10, 10])
    arr[i].set_ylim([-10, 10])
plt.show()

optimizer = optim.Adam(nf.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99995)
# Define the loss function
def loss(density, zk, log_jacobians):
    '''
    density: target_density
    zk: the final output
    log_jacobians: a list of log_abs_det_jacobians
    # '''
    # print(1111111111111,zk)
    # print(22222222222222,log_jacobians)
    sum_of_log_jacobians = sum(log_jacobians)
    return (-sum_of_log_jacobians - torch.log(density(zk) + 1e-9).squeeze()).mean()  #squeeze删除shape数为1的维度，
    # 使jacobian与密度矩阵维度相同可正常加和

id_figure = 2
plt.figure(figsize=(16, 18))
plt.subplot(3, 4, 1)
plt.hexbin(z[:, 0], z[:, 1], C = target_density(torch.Tensor(z)).numpy().squeeze(), cmap='rainbow')
plt.title('Target density', fontsize=15)

# Train
for it in range(10001):
    # Draw a sample batch from base_density
    sources = base_density.sample((512, ))
    # Evaluate flow of transforms
    zk, log_jacobians = nf(sources)
    # Evaluate loss and backprop
    optimizer.zero_grad()
    loss_v = loss(target_density, zk, log_jacobians)
    loss_v.backward()
    optimizer.step()
    scheduler.step()

    if (it % 1000 == 0):
        print('Loss (it. %i) : %f'%(it, loss_v.item()))
        # Draw random samples from base_density
        sources = base_density.sample((int(1e5), ))
        # Evaluate flow and plot
        zk, _ = nf(sources)
        zk = zk.detach().numpy()
        # Plot
        plt.subplot(3, 4, id_figure)
        plt.hexbin(zk[:, 0], zk[:, 1], cmap='rainbow')
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        plt.title('Iter.%i'%(it), fontsize=15);
        id_figure += 1


sources = base_density.sample((512, ))
fz = nf.get_fz(sources)
f, arr = plt.subplots(1, len(fz), figsize=(4 * (len(fz)), 4))
X0 = fz[0].detach().numpy()
for i in range(len(fz)):
    X1 = fz[i].detach().numpy()
    idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
    arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='red')
    idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
    arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='green')
    idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
    arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue')
    idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
    arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='black')
    arr[i].set_xlim([-10, 10])
    arr[i].set_ylim([-10, 10])
plt.savefig('toy2d_flow.png', dpi=300)

samples = nf.proj_density.sample((1000,))
plt.scatter(samples[:, 0], samples[:, 1], s=10, color='red')
plt.xlim([-5, 30])
plt.ylim([-10, 10])
plt.savefig('toy2d_out.png', dpi=300)

plt.show()