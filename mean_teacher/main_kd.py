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


class FullyConnectedNet(nn.Module):
    def __init__(self):
        super(FullyConnectedNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # NOTE: Check padding and size, want SAME padding
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def get_data_loader():
    batch_size = 32
    use_cuda = False
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    mnist_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))]
                                         )
    mnist_train = datasets.MNIST('../data', train=True, download=True, transform=mnist_transform)
    mnist_test = datasets.MNIST('../data', train=False, transform=mnist_transform)

    print(mnist_train.train_data.size())
    print(mnist_train.train_labels.size())
    print('')

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader


def xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)


class RunningAverage(object):
    def __init__(self, alpha, params):
        self.alpha = alpha
        self.params = []

        for p in params:
            self.params.append(torch.zeros_like(p))

    # TODO: Benchmark for speed
    # TODO: Compare to the following
    #       https://discuss.pytorch.org/t/how-to-apply-exponential-moving-average-decay-for-variables/10856/4
    #       https://discuss.pytorch.org/t/running-average-of-parameters/902
    def update_params(self, params):
        # Is this a good way to do things in_place?
        for p_run, p_curr in zip(self.params, params):
            p_run.mul_(self.alpha)
            p_run.add_(p_curr * (1 - self.alpha))


def main():
    x, y = sklearn.datasets.make_moons(n_samples=1000, shuffle=False, noise=0.1)
    net_conv = ConvNet()
    net_fc = FullyConnectedNet()
    net_conv = net_fc

    net_conv.apply(xavier)

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net_conv.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net_conv.parameters(), lr=1e-4)
    train_loader, test_loader = get_data_loader()

    run_ave = RunningAverage(0.9, net_conv.parameters())

    net_conv.train()
    num_epochs = 5
    for i in range(num_epochs):
        for j, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = net_conv(x)

            loss = criterion(y_pred, y)
            loss.backward()

            optimizer.step()

            run_ave.update_params(net_conv.parameters())

            if j % 20 == 0:
                print(loss)


if __name__ == '__main__':
    main()
