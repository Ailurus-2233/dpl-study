import numpy as np
import torch
from torch.utils import data
from torch import nn


def synthetic_data(w, b, num_samples):
    X = torch.normal(0, 1, (num_samples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


true_w = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)
true_b = torch.tensor(4.0, dtype=torch.float)
features, labels = synthetic_data(true_w, true_b, num_samples=1000)
batch_size = 10
data_iter = load_array((features, labels), batch_size)

net = nn.Sequential(nn.Linear(3, 1))
loss = nn.MSELoss()

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0.0)

optimizer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差', true_w-w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差', true_b-b)
