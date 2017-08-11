import pickle
import tempfile
from collections import OrderedDict

import numpy
import theano
import theano.tensor as tt

import t721.leaf as L
import t721.initializer as I
import t721.optimizer as O


def rec_gen(xs):
    for x in xs:
        if isinstance(x, int):
            yield x
        elif isinstance(x, list):
            yield from rec_gen(x)
        else:
            raise RuntimeError("unknown type {}".format(type(x)))


def test_rec_gen():
    xs = [1, 2, [3, [4]], 5, []]
    assert list(rec_gen(xs)) == [1, 2, 3, 4, 5]


class A(L.Leaf):
    def __init__(self):
        self.x = theano.shared(1)
        self.y = theano.shared(numpy.array([1, 2, 3]))
        self.z = None


class B(L.Leaf):
    def __init__(self):
        self.x = theano.shared(0)
        self.a = A()


class C(L.Leaf):
    def __init__(self, x):
        self.a = A()
        self.b = B()
        self.params = {x, self.b}  # manually identify


def test_leaf_get_params():
    a = A()
    assert set(a.get_params()) == {a.x, a.y}
    b = B()
    assert set(b.get_params()) == {b.x, b.a.x, b.a.y}, "recursively accumulated params"
    x = theano.shared(0)
    c = C(x)
    assert set(c.get_params()) == {x} | set(c.b.get_params()), "only manually selected params"


class U(L.Leaf):
    def __init__(self):
        x = theano.shared(1)
        self.updates = OrderedDict([(x, x + 1)])


class UU(L.Leaf):
    def __init__(self):
        self.u = U()

class UUU(L.Leaf):
    def __init__(self):
        self.u = U()
        self.uu = UU()


class V(L.Leaf):
    def __init__(self):
        a = theano.shared(1)
        values, self.updates = theano.scan(lambda: {a: a + 1}, n_steps=10)
        self.uuu = UUU()


def test_leaf_get_updates():
    u = U()
    assert set(u.get_updates()) == set(u.updates.items())
    uu = UU()
    assert set(uu.get_updates()) == set(uu.u.updates.items())
    uuu =UUU()
    assert set(uuu.get_updates()) == set(uuu.u.get_updates()) | set(uuu.uu.get_updates())
    v = V()
    assert set(v.get_updates()) == set(v.updates.items()) | set(v.uuu.get_updates())


cudnn_decimal = 6


def test_leaf_gru():
    from theano.gpuarray import gpuarray_shared_constructor
    from theano.gpuarray.tests.config import mode_with_gpu

    n_time = 5
    n_batch = 3
    n_input = 2
    n_output = 3
    xs_data = I.Normal()([n_time, n_batch, n_input])
    h0_data = I.Normal()([n_batch, n_output])
    ts_data = I.Normal()([n_time, n_batch, n_output])
    xs = tt.tensor3("xs")
    h0 = tt.matrix("h0")
    ts = tt.tensor3("ts")
    givens = {xs: xs_data, h0: h0_data, ts: ts_data}

    def forward(fun):
        return theano.function([], fun(xs, h0), givens=givens,
                               on_unused_input='ignore', mode=mode_with_gpu)()

    # NOTE: n_batch won't affect rnnblock (!?)
    gru = L.GRU(n_input, n_output, n_batch=1, impl=L.RNNImpl.auto)
    assert gru.impl != L.RNNImpl.auto
    gru.impl = L.RNNImpl.ref
    ref_ys = forward(gru.ref_forward)

    fused_ys = forward(gru.fused_forward)
    for r, f in zip(ref_ys, fused_ys):
        numpy.testing.assert_array_almost_equal(r, f)

    def backward(fun, params):
        ys = fun(xs, h0)[0]
        cost = tt.mean((ts - ys)**2)
        grad = tt.grad(cost, [xs, h0] + params)
        return theano.function([], grad, givens=givens,
                               on_unused_input='ignore', mode=mode_with_gpu)()

    ref_grad = backward(gru.ref_forward, gru.params)
    fused_grad = backward(gru.fused_forward, gru.params)
    for r, f in zip(ref_grad, fused_grad):
        numpy.testing.assert_array_almost_equal(r, f)

    gru.impl = L.RNNImpl.cudnn
    assert gru.params == list(gru.get_params())
    cudnn_ys = forward(gru.cudnn_forward)
    for r, c in zip(ref_ys, cudnn_ys):
        numpy.testing.assert_array_almost_equal(r, c, decimal=cudnn_decimal)
    cudnn_grad = backward(gru.cudnn_forward, gru.params)
    cudnn_grad = cudnn_grad[:2] + gru._rnn_block.split_params(
        gpuarray_shared_constructor(cudnn_grad[2]), 0, [n_batch, n_input])
    for r, f in zip(ref_grad, cudnn_grad):
        numpy.testing.assert_array_almost_equal(r, f, decimal=cudnn_decimal)

    gru.impl = L.RNNImpl.fused
    fused_ys = forward(gru.fused_forward)
    for r, f in zip(ref_ys, fused_ys):
        numpy.testing.assert_array_almost_equal(r, f)


def test_leaf_lstm():
    from theano.gpuarray import gpuarray_shared_constructor
    from theano.gpuarray.tests.config import mode_with_gpu

    n_time = 5
    n_batch = 3
    n_input = 2
    n_output = 3
    xs_data = I.Normal()([n_time, n_batch, n_input])
    ts_data = I.Normal()([n_time, n_batch, n_output])
    h0_data = I.Normal()([n_batch, n_output])
    c0_data = I.Normal()([n_batch, n_output])
    xs = tt.tensor3("xs")
    ts = tt.tensor3("xs")
    h0 = tt.matrix("h0")
    c0 = tt.matrix("c0")
    givens = {xs: xs_data, ts: ts_data, h0: h0_data, c0: c0_data}

    def forward(fun):
        return theano.function([], fun(xs, h0, c0), givens=givens,
                               on_unused_input='ignore', mode=mode_with_gpu)()

    lstm = L.LSTM(n_input, n_output, impl=L.RNNImpl.ref)
    ref_lstm = forward(lstm.ref_forward)

    fused_lstm = forward(lstm.fused_forward)
    for r, f in zip(ref_lstm, fused_lstm):
        numpy.testing.assert_array_almost_equal(r, f)

    def backward(fun, params):
        ys = fun(xs, h0, c0)[0]
        cost = tt.mean((ts - ys) ** 2)
        grad = tt.grad(cost, [xs, h0, c0] + params)
        return theano.function([], grad, givens=givens,
                               on_unused_input='ignore', mode=mode_with_gpu)()

    fused_grad = backward(lstm.fused_forward, lstm.params)
    ref_grad = backward(lstm.ref_forward, lstm.params)
    for r, f in zip(ref_grad, fused_grad):
        numpy.testing.assert_array_almost_equal(r, f)

    lstm.impl = L.RNNImpl.cudnn  # TODO: do this in cudnn_forward
    cudnn_lstm = forward(lstm.cudnn_forward)
    for r, c in zip(ref_lstm, cudnn_lstm):
        numpy.testing.assert_array_almost_equal(r, c, decimal=cudnn_decimal)

    cudnn_grad = backward(lstm.cudnn_forward, lstm.params)
    cudnn_grad = cudnn_grad[:3] + lstm._rnn_block.split_params(
        gpuarray_shared_constructor(cudnn_grad[3]), 0, [n_batch, n_input])
    for r, c in zip(ref_grad, cudnn_grad):
        numpy.testing.assert_array_almost_equal(r, c, decimal=cudnn_decimal)


class MLP(L.Leaf):
    def __init__(self, n_input, n_hidden, n_output):
        # stateful (shared) variables
        self.l1 = L.Linear(n_input, n_hidden)
        self.l2 = L.Linear(n_hidden, n_hidden)
        self.l3 = L.Linear(n_hidden, n_hidden)
        self.fc = L.Linear(n_hidden, n_output)
        # self.optimizer = O.SGD2(list(self.get_params()), lr=1e-3, momentum=0.95, nestrov=True)
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
        self.test = theano.function([x], y)


def test_leaf_pickle():
    mlp = MLP(2, 3, 4)
    x = I.Normal()([5, 2])
    y = mlp.test(x)

    fname = "/tmp/ppppp"
    with open(fname, "wb") as f:
        numpy.savez(f, *mlp.state_list())
    with open(fname, "rb") as f:
        l = numpy.load(f)
        mlp1 = MLP(2, 3, 4)
        mlp1.load_state_list((l[k] for k in l.files))
    y1 = mlp1.test(x)
    numpy.testing.assert_array_almost_equal(y, y1)

    with open(fname, "wb") as f:
        pickle.dump(mlp, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(fname, "rb") as f:
        mlp2 = pickle.load(f)
    y2 = mlp2.test(x)
    numpy.testing.assert_array_almost_equal(y, y2)


def test_leaf_conv2d():
    x_data = I.Normal()([1, 1, 28, 28])

    x = tt.tensor4("x")
    conv = L.Conv2D(1, 10, (5, 5), use_bias=False)
    y = conv(x)
    bn = L.BatchNorm(conv.get_output_shape(x_data.shape))
    y = bn(y)
    assert conv.filter.get_value().shape == (10, 1, 5, 5)

    f = theano.function([x], y, givens={L.Leaf.is_train: numpy.int8(True)})
    y_data = f(x_data)
    assert y_data.shape == conv.get_output_shape(x_data.shape)

    f = theano.function([x], y, givens={L.Leaf.is_train: numpy.int8(False)})
    y_data = f(x_data)
    assert y_data.shape == conv.get_output_shape(x_data.shape)


def test_leaf_deconv2d():
    x_data = I.Normal()([1, 1, 28, 28])

    x = tt.tensor4("x")
    conv = L.Conv2D(1, 10, (5, 5), use_bias=False)
    y = conv(x)
    deconv = L.Deconv2D(10, 1, (5, 5), use_bias=False)
    dx = deconv(y)
    assert dx.eval({x: x_data}).shape == x_data.shape

    deconv.filter = conv.filter
    dx = deconv(y)
    assert dx.eval({x: x_data}).shape == x_data.shape
    loss = dx.norm(2)
    gf = tt.grad(loss, deconv.filter)
    gf.eval({x: x_data})  # just confirm no throw
