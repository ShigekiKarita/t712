"""
ref: https://github.com/pytorch/examples/blob/master/mnist/main.py

with GTX1080
pytorch: 4.8 sec/epoch
theano: 4.5 sec/epoch
"""

import numpy
import theano
import theano.tensor as tt
from theano.tensor.signal.pool import pool_2d
import torch
from torchvision import datasets, transforms

import t721.leaf as L
import t721.optimizer as O


batch_size = 512
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)


class CNN(L.Leaf):
    def __init__(self, n_output):
        # stateful (shared) variables
        self.conv1 = L.Conv2D(1, 10, kernel=(5, 5))
        self.conv2 = L.Conv2D(10, 20, kernel=(5, 5))
        self.dropout = L.Dropout()
        self.fc1 = L.Linear(320, 50)
        self.fc2 = L.Linear(50, n_output)
        self.optimizer = O.Adam(lr=1e-3)

        # computation graphs
        x = tt.tensor4(name="x")
        t = tt.lvector(name="t")
        act = tt.nnet.relu
        h = self.conv1(x)
        h = pool_2d(h, (2, 2), mode="max", ignore_border=True)
        h = act(h)
        h = self.conv2(h)
        h = self.dropout(h)
        h = pool_2d(h, (2, 2), mode="max", ignore_border=True)
        h = act(h)
        h = h.reshape([x.shape[0], -1])
        h = act(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        y = tt.nnet.softmax(h)
        loss = tt.sum(tt.nnet.categorical_crossentropy(y, t))
        acc = tt.sum(tt.eq(tt.argmax(y, axis=1), t))

        # functions
        self.train = theano.function([x, t], [loss, acc],
                                     updates=self.optimize(loss, self.optimizer),
                                     givens={self.is_train: numpy.int8(True)})
        self.test = theano.function([x, t], [loss, acc],
                                    givens={self.is_train: numpy.int8(False)})


model = CNN(10)
for e in range(10):
    for k, loader in [("train", train_loader), ("test", test_loader)]:
        n = 0
        sum_loss = 0
        sum_acc = 0
        for i, (xs, ts) in enumerate(loader):
            xs = xs.numpy()
            ts = ts.numpy()
            l, a = getattr(model, k)(xs, ts)
            sum_loss += l
            sum_acc += a
            n += len(ts)
        print("{} loss: {}, acc {}".format(k, sum_loss / n, sum_acc / n))
