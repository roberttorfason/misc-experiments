import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn


def gaussian_1d(value, u, beta):
    return 1 / (2 * np.pi) * beta * torch.exp(-beta**2 * (u - value)**2 / 2)


def _generate_data(invert=False):
    amplitude = 0.3
    freq = 1
    t = np.linspace(0.0, 1.0, num=1000)

    y = amplitude * np.sin(2 * np.pi * freq * t)
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
    plt.plot(t, y)
    plt.show()

    net = FcNetwork(input_size=1)
    input_test = torch.from_numpy(np.array([[0.5], [0.1]], dtype=np.float32))
    result = net(input_test)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6000, gamma=0.2)

    train = torch.utils.data.TensorDataset(torch.from_numpy(t), torch.from_numpy(y))
    train_loader = torch.utils.data.DataLoader(train, batch_size=8, shuffle=True)

    net.train()
    for epoch in range(4000):
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

    t_extended = np.arange(start=-2, stop=2, step=0.01, dtype=np.float32)
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

    net.eval_without_dropout()
    plt.scatter(t, y)
    plt.plot(t_extended, net(torch.from_numpy(t_extended)).data.numpy(), 'r')
    plt.show()



if __name__ == '__main__':
    main()
