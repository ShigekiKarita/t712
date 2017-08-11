"""
sine wave fitting test for optimizers
"""
from time import time

import numpy
import theano
import theano.tensor as T

import t721.initializer as I
import t721.leaf as L
import t721.optimizer as O


def log(*args, **kwargs):
    if __name__ == '__main__':
        print(*args, **kwargs)


class RNN(L.Leaf):
    def __init__(self, optimizer):
        self.n_hidden = 100
        self.rnn1 = L.GRU(1, self.n_hidden)
        self.rnn2 = L.GRU(self.n_hidden, self.n_hidden)
        self.lin = L.Linear(self.n_hidden, 1)
        self.h1 = None
        self.h2 = None

        dropout = L.Dropout(0.5)
        xs = T.tensor3(name="xs")
        h1 = T.matrix("h1")
        h2 = T.matrix("h2")
        hs1 = self.rnn1(xs, h1)
        hsd1 = dropout(hs1)
        hs2 = self.rnn2(hsd1, h2)
        hsd2 = dropout(hs2)
        ys = T.tanh(L.sequence_apply(self.lin, hsd2))

        log("compiling rnn_predict ...")
        self.predict_fun = theano.function([xs, h1, h2], [ys, hs1[-1], hs2[-1]],
                                           givens={self.is_train: numpy.int8(False)})

        ts = T.tensor3(name="ts")
        loss = (ys - ts).norm(2) / ts.size
        log("compiling rnn_train ...")
        self.train_fun = theano.function([xs, h1, h2, ts], [loss, hs1[-1], hs2[-1]],
                                         updates=optimizer.updates(self.get_params(), loss),
                                         givens={self.is_train: numpy.int8(True)})

    def reset_state(self, n_batch):
        self.h1 = I.Constant(0.0)([n_batch, self.n_hidden])
        self.h2 = I.Constant(0.0)([n_batch, self.n_hidden])

    def train(self, xs, ts, init_states=None):
        if init_states is None:
            self.reset_state(xs.shape[1])
        else:
            self.h1, self.h2 = init_states
        loss, self.h1, self.h2 = self.train_fun(xs, self.h1, self.h2, ts)
        return loss

    def predict(self, xs, init_states=None):
        if init_states is None:
            self.reset_state(xs.shape[1])
        else:
            self.h1, self.h2 = init_states
        ys, self.h1, self.h2 = self.predict_fun(xs, self.h1, self.h2)
        return ys


n_step = 0.01
n_batch = 4
n_seq = 50
times = numpy.arange(n_seq*n_batch, dtype=theano.config.floatX).reshape(n_batch, n_seq, 1).transpose(1, 0, 2)
nxs = numpy.sin(times / n_step)

for o in [O.SGD(lr=1e-1, momentum=0.95, nestrov=True), O.Adadelta(lr=1.0), O.Adam(lr=1e-4)]:
    log(o)
    model = RNN(o)
    initial_predict = model.predict(nxs[:-1])
    initial_loss = None
    start = time()
    for i in range(100):
        x_data = nxs[:-1]
        t_data = nxs[1:]
        l = model.train(x_data, t_data)
        if initial_loss is None:
            initial_loss = l
        if i % 10 == 0:
            log("loss", l)

    log("time: ", time() - start)


if __name__ == '__main__':
    from matplotlib import pyplot
    pyplot.plot(times[1:, 0].ravel(), initial_predict[:, 0].ravel())
    pyplot.plot(times[1:, 0].ravel(), model.predict(nxs[:-1])[:, 0].ravel())
    pyplot.plot(times[1:, 0].ravel(), nxs[1:, 0].ravel(), linestyle="--")
    pyplot.show()
