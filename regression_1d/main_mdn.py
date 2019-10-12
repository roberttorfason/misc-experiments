import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from regression_1d import input_data, models


class LLWithVar(nn.Module):
    def forward(self, mean, beta, value):
        mean_squared = (value - mean)**2
        mean_squared_scaled = beta * mean_squared
        norm = torch.log(beta)
        ll_tot = norm - mean_squared_scaled
        ll_ave = -torch.mean(ll_tot)

        return ll_ave


def main():
    t, y = input_data.generate_data(invert=False)

    # plt.scatter(t, y)
    # plt.plot(t, y)
    # plt.show()

    net = models.FcNetwork(input_size=1)
    # input_test = torch.from_numpy(np.array([[0.5], [0.1]], dtype=np.float32))
    # result = net(input_test)

    criterion = LLWithVar()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6000, gamma=0.2)

    train = torch.utils.data.TensorDataset(torch.from_numpy(t), torch.from_numpy(y))
    train_loader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)

    net.train()
    for epoch in range(1200):
        loss_avg = 0.0
        counter = 0
        for t_batch, y_batch in train_loader:
            optimizer.zero_grad()

            mean_batch, beta_batch = net(t_batch)
            loss = criterion(mean_batch, beta_batch, y_batch)
            loss.backward()

            loss_avg += loss.item()

            for name, param in net.named_parameters():
                if "beta" in name:
                    # print(param)
                    pass

            optimizer.step()
            counter += 1

        if epoch % 100 == 0:
            print(loss_avg / counter)

    t_extended = np.arange(start=-0.5, stop=1.5, step=0.01, dtype=np.float32)
    t_extended = np.expand_dims(t_extended, -1)
    net.eval()

    plt.figure()
    plt.scatter(t, y)
    plt.plot(t_extended, net(torch.from_numpy(t_extended))[0].data.numpy(), 'r')
    plt.show()

    plt.figure()
    plt.scatter(t, y)
    plt.plot(t_extended, net(torch.from_numpy(t_extended))[1].data.numpy(), 'r')
    plt.show()


if __name__ == '__main__':
    main()
