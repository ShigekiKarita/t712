import matplotlib
import matplotlib.pyplot as plt
import numpy
import theano
import theano.tensor as tt
import torch
from torchvision import datasets, transforms


import t721.leaf as L
import t721.optimizer as O
import t721.initializer as I


floatX = getattr(numpy, theano.config.floatX)


class BNGenerator(L.Leaf):
    def __init__(self, n_input, n_hidden, n_output):
        pass

class GAN(L.Leaf):
    def __init__(self, lr=1e-3, activation=tt.nnet.sigmoid, n_layers=3, weight_init=I.XavierUniform(),
                 n_rand=128, n_g_hidden=512, n_d_hidden=32, n_input=2, n_d_output=1):
        self.n_rand = n_rand
        self.rng = tt.shared_randomstreams.RandomStreams()
        self.g = L.MLP(n_rand, n_g_hidden, n_input, n_layers=n_layers,
                       activation=activation, weight_init=weight_init)
        self.d = L.MLP(n_input, n_d_hidden, n_d_output, n_layers=n_layers,
                       activation=activation, weight_init=weight_init)

        n_sample = tt.lscalar("n_sample")
        self.generate = theano.function([n_sample], self.g(self.rng.normal((n_sample, n_rand))))

        xs = tt.matrix("xs")
        d_loss = self.d_loss(xs)
        d_optimizer = O.Adam(lr)
        self.train_d = theano.function([xs], d_loss, updates=self.d.optimize(d_loss, d_optimizer))

        g_loss = self.g_loss(n_sample)
        g_optimizer = O.Adam(lr)
        self.train_g = theano.function([n_sample], g_loss, updates=self.g.optimize(g_loss, g_optimizer))

    def loss(self, xs, ts):
        ys = tt.nnet.sigmoid(self.d(xs))
        loss = tt.mean(tt.nnet.binary_crossentropy(ys.reshape([-1]), ts))
        return loss

    def d_loss(self, xs):
        n_batch = xs.shape[0]
        gs = tt.nnet.sigmoid(self.g(self.rng.normal((n_batch, self.n_rand))))
        gxs = tt.concatenate([xs, gs])
        gts = tt.concatenate([tt.ones([n_batch], dtype=numpy.int64),
                              tt.zeros([n_batch], dtype=numpy.int64)])
        return self.loss(gxs, gts)

    def g_loss(self, n_batch):
        xs = tt.nnet.sigmoid(self.g(self.rng.normal((n_batch, self.n_rand))))
        ts = tt.ones([n_batch], dtype=numpy.int64)
        return self.loss(xs, ts)


model = GAN(lr=1e-4, n_rand=128, n_input=28*28, n_g_hidden=512, n_d_hidden=256, n_layers=4)
zs = tt.matrix("zs")
decode = theano.function([zs], model.g(zs))


def test(r=2, nn=10, random=True):
    if random:
        xs = model.generate(nn * nn)
    else:
        zs = numpy.array([(z1, z2)
                          for z1 in numpy.linspace(-r, r, nn)
                          for z2 in numpy.linspace(-r, r, nn)]).astype('float32')
        xs = decode(zs)

    xs = numpy.bmat([[xs.reshape(-1, 28, 28)[i + j * nn] for i in range(nn)] for j in range(nn)])
    matplotlib.rc('axes', **{'grid': False})
    plt.figure(figsize=(10, 10))
    plt.imshow(xs, interpolation='none', cmap='gray')
    plt.show()


# test()
n_batch = 256
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()
                   ])),
    batch_size=n_batch, shuffle=True)

for epoch in range(1, 100 + 1):
    d_loss = 0
    g_loss = 0
    sum_n = 0
    for i, (xs, ts) in enumerate(train_loader):
        xs = xs.numpy().reshape(-1, 28*28) / 255
        n = len(xs) * 2
        d_loss += model.train_d(xs) * n
        g_loss += model.train_g(n) * n
        sum_n += n
        if numpy.isnan(d_loss + g_loss):
            print("nan detected!")
            break
    if epoch % 10 == 0:
        print("Epoch: {},\tDiscriminator: {},\tGenerator: {}".format(epoch, d_loss / sum_n, g_loss / sum_n))

test()
plt.show()
