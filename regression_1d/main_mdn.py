import logging

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from torch import nn
import torch.nn.functional as F
import torch
import torch.utils.data


def _init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)


class FcNetwork(nn.Module):
    def __init__(self, input_size, dropout_probability):
        # TODO(robert): Get the basic network to train!
        #               probably wrong intialization for tanh
        # TODO(robert): Try with different activation functions
        num_outputs = 9
        super(FcNetwork, self).__init__()
        self._linear0 = nn.Linear(input_size, 20)
        self._linear1 = nn.Linear(20, num_outputs)

        self._mixture_soft = nn.Softmax()

        self.apply(_init_weights)

    def forward(self, x):
        out = self._linear0(x)
        out = torch.tanh(out)
        out = self._linear1(out)

        mixture = self._mixture_soft(out[0:3])
        mean = out[3:6]
        var = torch.exp(out[6:9])

        return out


def _generate_data(invert=False):
    amplitude = 0.3
    freq = 1.0
    t = np.arange(start=0, stop=1, step=0.001, dtype=np.float32)
    y = amplitude * np.sin(2 * np.pi * freq * t)
    # drift = 0.2 * t
    drift = 1.0 * t
    sequence_len = t.shape[0]

    noise = np.random.uniform(low=-0.1, high=0.1, size=sequence_len)

    y_hat = y + drift + noise

    t = t.astype(np.float32)
    y_hat = y_hat.astype(np.float32)

    if invert:
        t, y_hat = y_hat, t

    return np.expand_dims(t, -1), np.expand_dims(y_hat, -1)


def main():
    t, y = _generate_data(invert=True)

    plt.scatter(t, y)
    plt.show()

    net = FcNetwork(input_size=1, dropout_probability=0.1)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6000, gamma=0.2)

    train = torch.utils.data.TensorDataset(torch.from_numpy(t), torch.from_numpy(y))
    train_loader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)

    net.train()
    for epoch in range(400):
        # scheduler.step()
        loss_avg = 0.0
        counter = 0
        for t_batch, y_batch in train_loader:
            optimizer.zero_grad()

            outputs = net(t_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            loss_avg += loss.item()

            optimizer.step()
            counter += 1

        if epoch % 200 == 0:
            print(loss_avg / counter)

    t_extended = np.arange(start=-1, stop=2, step=0.01, dtype=np.float32)
    t_extended = np.expand_dims(t_extended, -1)
    net.eval()
    plt.scatter(t, y)
    plt.plot(t_extended, net(torch.from_numpy(t_extended)).data.numpy(), 'r')
    plt.show()

    num_samples = 5
    output_samples = np.zeros((t_extended.shape[0], num_samples))

    net.eval_with_dropout()
    for i in range(num_samples):
        output_samples[:, i] = net(torch.from_numpy(t_extended)).data.numpy().ravel()

    plt.scatter(t, y)
    # plt.scatter(t_extended, output_samples, 'r')
    plt.plot(t_extended, output_samples)
    plt.show()

    # net.eval_with_dropout()
    # plt.scatter(t, y)
    # plt.plot(t_extended, net(torch.from_numpy(t_extended)).data.numpy(), 'r')
    # plt.show()


if __name__ == '__main__':
    main()
