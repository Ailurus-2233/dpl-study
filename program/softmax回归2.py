import torch
import torchvision
from torch import nn
from torch.utils import data
from torchvision import transforms


W = torch.normal(0, 0.01, (784, 10), requires_grad=True)
b = torch.zeros(10, requires_grad=True)
lr = 0.1

def load_data_fashion_mnist(batch_size, resize=None, root='../data'):
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root=root, train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=root, train=False, transform=trans, download=True
    )
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(mnist_test, batch_size, shuffle=True, num_workers=4))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)


def current_num(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def evaluate_accuracy(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(current_num(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(l.sum(), current_num(y_hat, y), y.numel())
    return metric[0]/metric[2], metric[1]/metric[2]


def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(
            f'epoch {epoch + 1}: loss {train_metrics[0]:.3f}, train acc {train_metrics[1]:.3f}, test acc {test_acc:.3f}')
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


def main():
    batch_size = 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net.apply(init_weights)
    loss = nn.CrossEntropyLoss(reduction="none")
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    num_epochs = 10
    train(net, train_iter, test_iter, loss, num_epochs, trainer)


if __name__ == '__main__':
    main()