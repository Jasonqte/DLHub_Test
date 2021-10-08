import math
import torch
from torch import nn
import numpy as np
from torch import optim
from torch import distributions as distrib
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler

# Here some modern flows were implemented
from nflows.base import FlowSequential
from nflows.realnvp import LinearMaskedCoupling
from nflows.bn import BatchNormFlow

batch_size = 100
n_samples = 1000
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
X, y = noisy_moons
X = StandardScaler().fit_transform(X)
samples = torch.tensor(X).type(torch.float)
plt.scatter(samples[:, 0], samples[:, 1], s=10, color='red')


class RealNVP(nn.Module):
    '''
    Construct flows from target density (train data) to source density (known density i.e., Gaussian distribution.)
    x indicates train data while z indicates source samples
    target density -> source density
              q(x) -> p(z)
    '''

    def __init__(self, n_blocks, input_size, hidden_size, n_hidden, source_density):
        super(RealNVP, self).__init__()

        bijectors = []
        mask = torch.arange(input_size).float() % 2
        for i in range(n_blocks):
            bijectors += [LinearMaskedCoupling(input_size, hidden_size, n_hidden, mask)]
            mask = 1 - mask
            bijectors += [BatchNormFlow(input_size)]

        self.bijectors = bijectors
        self.source_density = source_density
        self.flows = FlowSequential(*bijectors)

    def forward(self, x):
        '''
        T(x) = z
        '''
        zs, sum_log_abs_det_jacobians = self.flows(x)
        return zs, sum_log_abs_det_jacobians

    def inverse(self, z):
        '''
        T^{-1}(z) = x
        '''
        xs, sum_log_abs_det_jacobians = self.flows.inverse(z)
        return xs, sum_log_abs_det_jacobians

    def log_prob(self, x):
        '''
        \log q(x) = \log p(z) + log_abs_det_jacobian
        '''
        zs, sum_log_abs_det_jacobians = self.forward(x)
        return self.source_density.log_prob(zs[-1]) + sum_log_abs_det_jacobians

source_density = distrib.MultivariateNormal(torch.zeros(2), torch.eye(2))
nf = RealNVP(n_blocks=4, input_size=2, hidden_size=4, n_hidden=4, source_density=source_density)


def plot_flows(outputs):
    '''
    outputs (list): list of tensors
    '''
    f, arr = plt.subplots(1, len(outputs), figsize=(4 * (len(outputs)), 4))

    X0 = outputs[0].detach().numpy()
    for i in range(len(outputs)):
        X1 = outputs[i].detach().numpy()
        idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='red')

        idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='green')

        idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue')

        idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
        arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='black')

        arr[i].set_xlim([-3, 3])
        arr[i].set_ylim([-3, 3])


# train distribution -> known distribution
# q(x) -> p(z)
zs, _ = nf(samples)
plot_flows(zs)

optimizer = optim.Adam(nf.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99994)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7000], gamma=0.1)

# Train
for it in range(20001):
    # samples of built toy distribution as train data
    loss = - nf.log_prob(samples).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if it % 1000 == 0:
        print(it, loss.item())


# train distribution -> known distribution
# q(x) -> p(z)
zs, _, = nf(samples)
plot_flows(zs)

sources = source_density.sample((512, ))
xs, _, = nf.inverse(sources)
plot_flows(xs)


sources = source_density.sample((1000, ))
xs, _, = nf.inverse(sources)
tmp = xs[-1].detach().numpy()
plt.scatter(tmp[:, 0], tmp[:, 1], color='red');