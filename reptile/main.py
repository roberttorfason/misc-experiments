import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.xavier_uniform_(m.weight)


class FullyConnectedNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 40)
        self.fc2 = nn.Linear(40, 40)
        self.fc3 = nn.Linear(40, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def generate_data():
    a = 1.0  # a in [0.1, 0.5]
    b = np.pi  # b in [0, 2*pi]
    x = np.arange(start=-5, stop=5, step=0.01, dtype=np.float32)  # x in [-5.0, 5.0]
    y = a * np.sin(x + b)

    return torch.from_numpy(np.expand_dims(x, -1)), torch.from_numpy(np.expand_dims(y, -1))


def get_mean_state_dict(state_dicts):
    base_keys = set(state_dicts[0].keys())
    assert all([base_keys == set(state_dict.keys()) for state_dict in state_dicts]), 'Expected all state dictionaries' \
                                                                                     ' to have the same keys.'

    result = {k: torch.zeros_like(v) for k, v in state_dicts[0].items()}

    for state_dict in state_dicts:
        for k, v in state_dict.items():
            result[k].add_(v)

    for k in result:
        result[k].div_(len(state_dicts))

    return result


def main():
    x, y = generate_data()

    # plt.scatter(x, y)
    # plt.show()

    net = FullyConnectedNet()
    net.apply(init_weights)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6000, gamma=0.2)

    train = torch.utils.data.TensorDataset(x, y)
    train_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)

    net_states = []
    net.train()
    for epoch in range(40):
        # scheduler.step()
        loss_avg = 0.0
        counter = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            outputs = net(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            loss_avg += loss.item()

            optimizer.step()
            counter += 1

        if epoch % 10 == 0:
            print(loss_avg / counter)
            net_states.append(copy.deepcopy(net.state_dict()))

    net.eval()
    net_states_mean = get_mean_state_dict(net_states)
    net.load_state_dict(net_states_mean)

    for net_state in net_states:
        net.load_state_dict(net_state)
        plt.plot(x.data.numpy(), net(x).data.numpy(), 'r')
        plt.plot(x.data.numpy(), y.data.numpy())
        plt.show()


if __name__ == '__main__':
    main()
