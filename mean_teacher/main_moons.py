import math

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sklearn.datasets
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class FullyConnectedNet(nn.Module):
    def __init__(self):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)


def main():
    x_data, y_data = sklearn.datasets.make_moons(n_samples=2000, shuffle=False, noise=0.1)
    x_data = x_data.astype(np.float32)
    # TODO: Normalize dataset
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_data), torch.from_numpy(y_data))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=32,
                                               shuffle=True,
                                               )

    x1_min, x2_min = np.min(x_data, axis=0)
    x1_max, x2_max = np.max(x_data, axis=0)
    h = 0.02
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
                           np.arange(x2_min, x2_max, h),
                           )
    xx1 = xx1.astype(np.float32)
    xx2 = xx2.astype(np.float32)

    net_fc = FullyConnectedNet()

    net_fc.apply(xavier)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net_fc.parameters(), lr=5e-4)

    net_fc.train()
    num_epochs = 200
    for i in range(num_epochs):
        for j, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = net_fc(x)

            loss = criterion(y_pred, y)
            loss.backward()

            optimizer.step()

            if j % 40 == 0:
                print(loss)

        print('Epoch: {}'.format(i))

    net_fc.eval()

    xx_in = np.hstack([xx1.reshape(-1, 1), xx2.reshape(-1, 1)])

    zz = net_fc(torch.from_numpy(xx_in))
    # Just need one prob, the other is 1 - p0
    # Check this detach thing out...
    zz = zz.detach().numpy()
    zz = zz[:, 1]

    ax = plt.subplot(111)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax.contourf(xx1, xx2, zz.reshape(xx1.shape), cmap=cm, alpha=.8)
    #print(x_data.shape)
    ax.scatter(x_data[:, 0], x_data[:, 1], c=y_data, cmap=cm_bright)
    #ax.contourf(xx1, xx2, zz.reshape(xx1.shape), alpha=.8)
    plt.show()


if __name__ == '__main__':
    main()
