"""
ref: https://github.com/pytorch/examples/blob/master/mnist/main.py

with GTX1080
pytorch: 4.8 sec/epoch
theano: 4.5 sec/epoch
"""

import theano
import theano.tensor as tt
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


class MLP(L.Leaf):
    def __init__(self, n_input, n_hidden, n_output):
        # stateful (shared) variables
        self.l1 = L.Linear(n_input, n_hidden)
        self.l2 = L.Linear(n_hidden, n_hidden)
        self.l3 = L.Linear(n_hidden, n_hidden)
        self.fc = L.Linear(n_hidden, n_output)
        self.optimizer = O.Adam(lr=1e-3)

        # computation graphs
        x = tt.matrix(name="x")
        t = tt.lvector(name="t")
        act = tt.nnet.relu
        h = act(self.l1(x))
        h = act(self.l2(h))
        h = act(self.l3(h))
        y = tt.nnet.softmax(self.fc(h))
        loss = tt.sum(tt.nnet.categorical_crossentropy(y, t))
        acc = tt.sum(tt.eq(tt.argmax(y, axis=1), t))

        # functions
        self.train = theano.function([x, t], [loss, acc], updates=self.optimizer.updates(self.get_params(), loss))
        self.test = theano.function([x, t], [loss, acc])


model = MLP(28*28, 320, 10)
for e in range(10):
    for k, loader in [("train", train_loader), ("test", test_loader)]:
        n = 0
        sum_loss = 0
        sum_acc = 0
        for i, (xs, ts) in enumerate(loader):
            xs = xs.numpy().reshape(-1, 28 * 28)
            ts = ts.numpy()
            l, a = getattr(model, k)(xs, ts)
            sum_loss += l
            sum_acc += a
            n += len(ts)
        print("{} loss: {}, acc {}".format(k, sum_loss / n, sum_acc / n))
