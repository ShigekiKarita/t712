"""
convolutional VAE example with PyMC3
https://github.com/pymc-devs/pymc3/blob/master/docs/source/notebooks/convolutional_vae_keras_advi.ipynb
"""
from collections import OrderedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy
import pymc3 as pm
import theano
import theano.tensor as tt
from torchvision import datasets, transforms

import t721.leaf as L
from t721.logger import logger


class VAE(L.Leaf):
    def __init__(self, n_latent, n_hidden=128):
        self.act = tt.nnet.relu

        # params for encode
        self.fc1 = L.Linear(28*28, n_hidden)
        self.fc2 = L.Linear(n_hidden, n_latent * 2)

        # params for decode
        self.dfc1 = L.Linear(n_latent, n_hidden)
        self.dfc2 = L.Linear(n_hidden, 28*28)

    def encode(self, xs):
        h = xs.reshape([xs.shape[0], -1])
        h = self.act(self.fc1(h))
        encoded = self.fc2(h)
        return tt.split(encoded, [n_latent] * 2, 2, axis=1)

    def decode(self, zs):
        h = self.act(self.dfc1(zs))
        h = self.act(self.dfc2(h))
        h = h.reshape([zs.shape[0], 1, 28, 28])
        return tt.nnet.sigmoid(h)


class ConvVAE(L.Leaf):
    def __init__(self, n_latent, n_hidden=128, n_filters=64):
        self.act = tt.nnet.relu
        self.n_filters = n_filters

        # params for encode
        self.conv1 = L.Conv2D(1, n_filters, kernel=(3, 3))
        self.conv2 = L.Conv2D(n_filters, n_filters, kernel=(3, 3))
        self.pool = L.MaxPooling2D(kernel=(2, 2))
        self.conved_shape = L.get_output_shape((1, 1, 28, 28), [self.conv1, self.conv2])
        print(self.conved_shape)
        n_out = numpy.prod(self.conved_shape)
        self.fc1 = L.Linear(n_out, n_hidden)
        self.fc2 = L.Linear(n_hidden, n_latent * 2)

        # params for decode
        self.dfc1 = L.Linear(n_latent, n_hidden)
        self.dfc2 = L.Linear(n_hidden, n_out)
        self.dconv1 = L.Deconv2D(n_filters, n_filters, kernel=(3, 3))
        self.dconv2 = L.Deconv2D(n_filters, 1, kernel=(3, 3))
        n_revert = L.get_output_shape(self.conved_shape, [self.dconv1, self.dconv2])
        print(n_revert)

    def encode(self, xs):
        h = self.conv1(xs)
        h = self.act(h)
        h = self.conv2(h)
        h = self.act(h)
        h = h.reshape([xs.shape[0], -1])
        h = self.act(self.fc1(h))
        encoded = self.fc2(h)
        return tt.split(encoded, [n_latent] * 2, 2, axis=1)

    def decode(self, zs):
        h = self.act(self.dfc1(zs))
        h = self.act(self.dfc2(h))
        h = h.reshape([zs.shape[0], *self.conved_shape[1:]])
        h = self.act(self.dconv1(h))
        h = self.dconv2(h)
        return tt.nnet.sigmoid(h)


logger.info("loading dataset")
batch_size = 128
train_mnist = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
x_train = train_mnist.train_data.numpy()
data = pm.floatX(x_train.reshape(-1, 1, 28, 28))
data /= numpy.max(data)


logger.info("defining symbols")
n_latent = 2
vae = VAE(n_latent)
xs = tt.tensor4("xs")
xs.tag.test_value = numpy.zeros((batch_size, 1, 28, 28)).astype('float32')

logger.info("building model")
with pm.Model() as model:
    zs = pm.Normal("zs", mu=0, sd=1, shape=(batch_size, n_latent),
                   dtype=theano.config.floatX, total_size=len(data))
    xs_ = pm.Normal("xs_", mu=vae.decode(zs), sd=0.1, observed=xs,
                    dtype=theano.config.floatX, total_size=len(data))

local_RVs = OrderedDict({zs: vae.encode(xs)})
xs_t_minibatch = pm.Minibatch(data, batch_size)

logger.info("fitting model")
with model:
    approx = pm.fit(15000, local_rv=local_RVs, more_obj_params=list(vae.get_params()),
                    obj_optimizer=pm.adam(learning_rate=1e-3),
                    more_replacements={xs: xs_t_minibatch})
plt.plot(approx.hist)

# evaluate analogy
dec_zs = tt.matrix()
dec_fun = theano.function([dec_zs], theano.clone(vae.decode(zs), {zs: dec_zs}))


def test():
    nn = 10
    zs = numpy.array([(z1, z2)
                      for z1 in numpy.linspace(-2, 2, nn)
                      for z2 in numpy.linspace(-2, 2, nn)]).astype('float32')
    xs = dec_fun(zs)[:, 0, :, :]
    xs = numpy.bmat([[xs[i + j * nn] for i in range(nn)] for j in range(nn)])
    matplotlib.rc('axes', **{'grid': False})
    plt.figure(figsize=(10, 10))
    plt.imshow(xs, interpolation='none', cmap='gray')
    plt.show()


test()
