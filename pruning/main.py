import math
import time

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class MnistCustom(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size

        x, y = torch.load(data_path)
        # Where is the best place to make them floats?
        self.x = (x.float() - 128.0) / 128.0
        self.y = y.long()

        self.loc = 0

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {'images': self.x[idx], 'labels': self.y[idx]}

    def __iter__(self):
        return self

    def __next__(self):
        if self.loc >= self.__len__():
            raise StopIteration
        else:
            x_batch = self.x[self.loc:self.loc+self.batch_size]
            y_batch = self.y[self.loc:self.loc+self.batch_size]
            self.loc += self.batch_size

        return x_batch, y_batch


class FullyConnectedNet(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc2 = nn.Linear(400, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def test(net, dataset_test):
    net.eval()
    test_loss = 0
    correct = 0.0
    tot_num_examples = 0.0
    with torch.no_grad():
        for x, y in dataset_test:
            y_pred_soft = net(x)
            y_pred = y_pred_soft.max(1, keepdim=True)[1] # get the index of the max log-probability
            # sm = torch.nn.Softmax()
            # print(sm(y_pred_soft))
            # print(y_pred)
            # import pdb; pdb.set_trace()
            # print(y_pred.eq(y.view_as(y_pred)).sum().item())
            correct += y_pred.eq(y.view_as(y_pred)).sum().item()
            tot_num_examples += y_pred.view(-1).size()[0]

    print('\nTest set: Accuracy: {}/{} ({:.1f}%)\n'.format(
        correct, tot_num_examples, 100.0 * correct / tot_num_examples))

    print(tot_num_examples)
    print(correct)

def Xavier(m):
    if m.__class__.__name__ == 'Linear':
        fan_in, fan_out = m.weight.data.size(1), m.weight.data.size(0)
        std = 1.0 * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        m.weight.data.uniform_(-a, a)
        m.bias.data.fill_(0.0)

def main():
    torch.manual_seed(11)
    net = Net()
    net.apply(Xavier)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    batch_size = 10
    test_batch_size = 10

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

    # x_in = mnist_train.train_data[0:20].float()
    # tmp_data = net(x_in)
    # tmp_data_0 = net(mnist_train.train_data[0:10].float())
    # tmp_data_1 = net(mnist_train.train_data[10:20].float())

    # tmp_label = mnist_train.train_labels[0:20].long()
    # tmp_label_0 = mnist_train.train_labels[0:10].long()
    # tmp_label_1 = mnist_train.train_labels[10:20].long()

    # print(tmp_label)
    # print(criterion(tmp_data, tmp_label))

    # step_size = 20
    # p = next(net.parameters())
    # optimizer.zero_grad()
    # counter = -1
    # for i, idx in enumerate(range(0, 100, step_size)):
    #     tmp_data = net(mnist_train.train_data[idx:idx+step_size].float())
    #     tmp_label = mnist_train.train_labels[idx:idx+step_size].long()
    #     loss = criterion(tmp_data, tmp_label)
    #     loss.backward()
    #     counter = i + 1
    #     print(p.grad.sum() / counter)

    # print(p.grad.sum())
    # optimizer.zero_grad()
    # print(p.grad.sum())
    # tmp_data = net(mnist_train.train_data[0:100].float())
    # tmp_label = mnist_train.train_labels[0:100].long()
    # loss = criterion(tmp_data, tmp_label)
    # loss.backward()
    # print(p.grad.sum())
    # print(p.grad.dtype)

    # tmp_data = net(mnist_train.train_data[idx:idx+step_size].float())
    # tmp_label = mnist_train.train_labels[idx:idx+step_size].long()
    # 1/0
    # def test_hook(module, in_grad, out_grad):
    #        in_grad = my_new_gradient
    #        idx = torch.ones_like(p.grad, dtype=torch.uint8)
    #        # print(idx.size())
    #        p.grad[idx] = 0

    # datasets.ImageFolder
    list_of_masks = []
    for p in net.parameters():
        list_of_masks.append(torch.ones_like(p.data, dtype=torch.float))

    running_loss = 0.0
    loss = 0
    for epoch in range(4):
        net.train()
        # dataloader = DataLoader(MnistCustom('data/mnist_train.pt', batch_size = 6), batch_size=10, shuffle=True, num_workers=4)
        # dataloader = MNIST('data/mnist_train.pt', batch_size = 6)
        for i, (x, y) in enumerate(train_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred = net(x)
            loss = criterion(y_pred, y)
            loss.backward()

            start = time.time()
            for p, mask in zip(net.parameters(), list_of_masks):
                if p.grad is not None:
                    # print('before grad: {}'.format(p.grad.sum()))
                    # idx = torch.ones_like(p.grad, dtype=torch.uint8)
                    # print(idx.size())
                    p.grad.mul_(mask)
                    #print('after grad: {}'.format(p.grad.sum()))

            #p_tmp = next(net.parameters())
            #print('before data: {}'.format(p_tmp.data.sum()))
            optimizer.step()
            print(time.time() - start)
            #print('after data: {}'.format(p_tmp.data.sum()))

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        for i, (x, y) in enumerate(train_loader):
            if i % 2000 == 0:
                print(i)
                print(y)

        test(net, test_loader)

        print('epoch {}'.format(epoch))

    print('Finished Training')


if __name__ == '__main__':
    main()
