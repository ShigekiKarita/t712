import matplotlib
import matplotlib.pyplot as plt
import numpy
import theano
import theano.tensor as tt
from torchvision import datasets

import t721.leaf as L
import t721.optimizer as O
import t721.initializer as I


floatX = theano.config.floatX


def gaussian_mixture_circle(batchsize, num_cluster=8, scale=1, std=1):
    rand_indices = numpy.random.randint(0, num_cluster, size=batchsize)
    base_angle = numpy.pi * 2 / num_cluster
    angle = rand_indices * base_angle - numpy.pi / 2
    mean = numpy.zeros((batchsize, 2), dtype=floatX)
    mean[:, 0] = numpy.cos(angle) * scale
    mean[:, 1] = numpy.sin(angle) * scale
    return numpy.random.normal(mean, std**2, (batchsize, 2)).astype(floatX)


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
        gs = self.g(self.rng.normal((n_batch, self.n_rand)))
        gxs = tt.concatenate([xs, gs])
        gts = tt.concatenate([tt.ones([n_batch], dtype=numpy.int64),
                              tt.zeros([n_batch], dtype=numpy.int64)])
        return self.loss(gxs, gts)

    def g_loss(self, n_batch):
        xs = self.g(self.rng.normal((n_batch, self.n_rand)))
        ts = tt.ones([n_batch], dtype=numpy.int64)
        return self.loss(xs, ts)


n_sample = 10000
n_cluster = 8
scale = 10

model = GAN(lr=1e-3)
n_batch = 32
gs_init = model.generate(n_sample)

n_iters = 50000
n_interval = n_iters // 10
d_loss = 0
g_loss = 0
for i in range(1, n_iters):
    xs = gaussian_mixture_circle(n_batch, n_cluster, scale)
    d_loss += model.train_d(xs)
    g_loss += model.train_g(n_batch * 2)
    if numpy.isnan(d_loss + g_loss):
        print("nan detected!")
        break
    if i % n_interval == 0:
        print("Discriminator: {},\tGenerator: {}".format(d_loss / i, g_loss / i))

real = gaussian_mixture_circle(n_sample, num_cluster=n_cluster, scale=scale)
gs = model.generate(n_sample)

plt.scatter(real[:, 0], real[:, 1], alpha=0.1, marker=".")
plt.scatter(gs_init[:, 0], gs_init[:, 1], alpha=0.1, marker=".")
plt.scatter(gs[:, 0], gs[:, 1], alpha=0.1, marker=".")
plt.legend(["real", "inital", "final"])
plt.show()
