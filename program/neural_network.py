from statistics import mode
import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from tools.show import show_model


# 简单的神经网路
class NeuralNetwork(nn.Module):
    def __init__(self, learning_rate=0.01):
        super(NeuralNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 100)
        )

        self.learning_rate = learning_rate
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=learning_rate)

    def forward(self, x):
        return self.net(x)

    def train(self, x, y, wd=0.0):
        if wd != 0.0:
            params = []
            for net in self.net:
                if type(net) == nn.Linear:
                    params.append({'params': net.weight, 'weight_decay': wd})
                    params.append({'params': net.bias})
            self.optimizer = torch.optim.SGD(params, lr=self.learning_rate)
        self.optimizer.zero_grad()
        predict = self.net(x)
        loss = self.loss(predict, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def load_data(root='../data', batch_size=64):
    training_data = datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    return (
        DataLoader(training_data, batch_size=batch_size, shuffle=True),
        DataLoader(test_data, batch_size=batch_size, shuffle=True)
    )


def main():
    device = "cuda"
    model = NeuralNetwork().to(device)
    show_model(model)
    # learning_rate = 1e-3
    # batch_size = 64
    # epochs = 10

    # train_data, test_data = load_data(batch_size=batch_size)

    # for i in range(epochs):
    #     for X, y in train_data:
    #         X, y = X.to(device), y.to(device)
    #         loss = model.train(X, y, wd=0.2)
        


if __name__ == '__main__':
    main()
