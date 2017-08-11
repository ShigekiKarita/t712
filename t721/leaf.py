import math
import sys
from collections import OrderedDict
from enum import Enum

import numpy
import theano
import theano.tensor as tt
from theano.ifelse import ifelse
from theano.tensor.shared_randomstreams import RandomStreams
from theano.tensor.sharedvar import SharedVariable
from theano.tensor.signal import pool

from t721.initializer import XavierNormal, Constant, Uniform
from t721.logger import logger
from t721.optimizer import Optimizer


def attrs_of(self, target_type):
    attrs = map(lambda s: getattr(self, s), dir(self))
    return filter(lambda x: isinstance(x, target_type), attrs)


# workaround for https://github.com/Theano/Theano/issues/689
sys.setrecursionlimit(40000)


# http://deeplearning.net/software/theano/library/tensor/shared_randomstreams.html
rng = RandomStreams()


def search_shared(root):
    for node in root:
        if isinstance(node, SharedVariable):
            yield node
        elif isinstance(node, Leaf):
            yield from node.get_params()


def search_updates(node):
    # for node in attrs_of(root, Leaf):
    if hasattr(node, "updates"):
        if hasattr(node.updates, "items"):
            yield from node.updates.items()
        else:
            yield from node.updates
    for child in attrs_of(node, Leaf):
        yield from search_updates(child)


class Leaf:
    """
    special fields
    + params : white list of params to optimize
    + updates : white list of updates to optimize
    """
    is_train = tt.bscalar()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("implement in a derived class")

    def get_params(self):
        if hasattr(self, "params"):
            return search_shared(self.params)
        return search_shared(getattr(self, s) for s in dir(self))

    def get_updates(self):
        return search_updates(self)

    def optimize(self, cost: tt.Variable, optimizer: Optimizer):
        updates = optimizer.updates(self.get_params(), cost)
        updates.update(self.get_updates())
        return updates

    def state_list(self):
        return list(p.get_value() for p in self.get_params())

    def load_state_list(self, state):
        for p, s in zip(self.get_params(), state):
            p.set_value(s)


def get_output_shape(x_shape, layers):
    o = x_shape
    for l in layers:
        o = l.get_output_shape(o)
    return o


class Linear(Leaf):
    def __init__(self, n_input, n_output, weight_init=XavierNormal(1.0), bias_init=Constant(0.0)):
        self.n_input = n_input
        self.n_output = n_output
        self.weight = theano.shared(weight_init((n_input, n_output)), name="weight")
        self.bias = theano.shared(bias_init(n_output), name="bias")

    def __call__(self, x):
        return x.dot(self.weight) + self.bias


class Conv1D(Leaf):
    def __init__(self, n_input_ch, n_output_ch, kernel, stride=1, pad=0, dilation=1, input_shape=None,
                 weight_init=XavierNormal(), bias_init=Constant(0.0), use_bias=True):
        self.n_input_ch = n_input_ch
        self.n_output_ch = n_output_ch
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.dilation = dilation
        self.input_shape = input_shape
        w = weight_init((n_input_ch, n_output_ch, kernel, 1)).transpose(1, 0, 2, 3)
        self.filter = theano.shared(w)
        self.use_bias = use_bias
        if use_bias:
            self.bias = theano.shared(bias_init(n_output_ch))

    def get_output_shape(self, input_shape):
        n_batch, n_input_ch, n_in_time = input_shape
        assert n_input_ch == self.n_input_ch
        n_out_time = math.floor((n_in_time + 2 * self.pad - self.dilation * (self.kernel - 1) - 1) / self.stride + 1)
        return n_batch, self.n_output_ch, n_out_time

    def __call__(self, x):
        """

        Args:
            x: (n_batch, n_input_ch, n_time)

        Returns:
            h: (n_batch, n_output_ch, n_time)
        """
        in_shape = None if self.input_shape is None else (*self.input_shape, 1)
        fx = tt.nnet.conv2d(x, self.filter, border_mode=(self.pad, 0), subsample=(self.stride, 1),
                            input_shape=in_shape, filter_shape=self.filter.get_value().shape,
                            filter_dilation=(self.dilation, 1))
        fx = fx.reshape(fx.shape[:3])
        if self.use_bias:
            fx = fx + self.bias.dimshuffle('x', 0, 'x')
        return fx


class Conv2D(Leaf):
    """
    References
        http://deeplearning.net/software/theano/library/tensor/nnet/conv.html
        http://deeplearning.net/tutorial/lenet.html
        http://sinhrks.hatenablog.com/entry/2014/12/07/203048
    """
    def __init__(self, n_input_ch, n_output_ch, kernel, stride=(1, 1), pad=(0, 0), dilation=(1,1), input_shape=None,
                 weight_init=XavierNormal(), bias_init=Constant(0.0), use_bias=True):
        assert len(kernel) == 2
        assert len(stride) == 2
        self.n_input_ch = n_input_ch
        self.n_output_ch = n_output_ch
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.dilation = dilation
        self.input_shape = input_shape
        w = weight_init((n_input_ch, n_output_ch, *kernel)).transpose(1, 0, 2, 3)
        self.filter = theano.shared(w)
        self.use_bias = use_bias
        if use_bias:
            self.bias = theano.shared(bias_init(n_output_ch))

    def get_output_shape(self, input_shape):
        assert len(input_shape) == 4
        assert input_shape[1] == self.n_input_ch
        os = [input_shape[0], self.n_output_ch]
        image_shape = input_shape[2:]
        for i in range(2):
            o = (image_shape[i] + 2 * self.pad[i] - self.dilation[i] * (self.kernel[i] - 1) - 1) / self.stride[i] + 1
            os.append(math.floor(o))
        return tuple(os)

    def __call__(self, x: tt.TensorVariable):
        assert x.ndim == 4
        fx = tt.nnet.conv2d(x, self.filter, border_mode=self.pad, subsample=self.stride,
                            input_shape=self.input_shape, filter_shape=self.filter.get_value().shape,
                            filter_dilation=self.dilation)
        if self.use_bias:
            return fx + self.bias.dimshuffle('x', 0, 'x', 'x')
        return fx


class Deconv2D(Leaf):
    """
    References
        http://deeplearning.net/software/theano/library/tensor/nnet/conv.html
        http://deeplearning.net/tutorial/lenet.html
        http://sinhrks.hatenablog.com/entry/2014/12/07/203048
    """
    def __init__(self, n_input_ch, n_output_ch, kernel, stride=(1, 1), pad=(0, 0), dilation=(1,1), input_shape=None,
                 weight_init=XavierNormal(), bias_init=Constant(0.0), use_bias=True):
        assert len(kernel) == 2
        assert len(stride) == 2
        self.n_input_ch = n_input_ch
        self.n_output_ch = n_output_ch
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.dilation = dilation
        self.input_shape = input_shape
        self.filter = theano.shared(weight_init((n_input_ch, n_output_ch) + kernel))
        self.use_bias = use_bias
        if use_bias:
            self.bias = theano.shared(bias_init(n_output_ch))

    def get_output_shape(self, output_shape):
        n_batch = None if isinstance(output_shape[0], tt.Variable) else output_shape[0]
        os = [n_batch, self.n_output_ch]
        image_shape = output_shape[2:]
        for i in range(2):
            o = (image_shape[i] - 1) * self.stride[i] + 1 + self.dilation[i] * (self.kernel[i] - 1) - 2 * self.pad[i]
            o = tt.cast(o, "int64") if isinstance(image_shape[i], tt.Variable) else math.floor(o)
            os.append(o)
        return tuple(os)

    def __call__(self, x: tt.TensorVariable):
        assert x.ndim == 4
        fx = tt.nnet.conv2d_grad_wrt_inputs(
            x, self.filter, input_shape=self.get_output_shape(x.shape), filter_shape=None,
            subsample=self.stride, border_mode=self.pad, filter_dilation=self.dilation)
        if self.use_bias:
            return fx + self.bias.dimshuffle('x', 0, 'x', 'x')
        return fx


class Pooling2D(Leaf):
    def __init__(self, mode, kernel, stride=None, pad=(0, 0)):
        if stride is None:
            stride = kernel
        assert len(kernel) == 2
        assert len(stride) == 2
        assert len(pad) == 2
        self.mode = mode
        self.kernel = kernel
        self.stride = stride
        self.pad = pad

    def __call__(self, x):
        return pool.pool_2d(input=x, ws=self.kernel, stride=self.stride, pad=self.pad,
                            mode=self.mode, ignore_border=True)

    def get_output_shape(self, input_shape):
        assert len(input_shape) == 4
        os = list(input_shape[:2])
        image_shape = input_shape[2:]
        for i in range(2):
            o = math.floor((image_shape[i] + 2 * self.pad[i] - (self.kernel[i] - 1) - 1) / self.stride[i] + 1)
            os.append(o)
        return tuple(os)


class MaxPooling2D(Pooling2D):
    def __init__(self, kernel, stride=None, pad=(0, 0)):
        super().__init__(mode="max", kernel=kernel, stride=stride, pad=pad)


class SumPooling2D(Pooling2D):
    def __init__(self, kernel, stride=None, pad=(0, 0)):
        super().__init__(mode="sum", kernel=kernel, stride=stride, pad=pad)


class AveragePooling2D(Pooling2D):
    def __init__(self, kernel, stride=None, pad=(0, 0), count_include_pad=True):
        if count_include_pad:
            mode = "average_inc_pad"
        else:
            mode = "average_exc_pad"
        super().__init__(mode=mode, kernel=kernel, stride=stride, pad=pad)


class Unpooling2D(Leaf):
    def __init__(self, mode, kernel):
        assert len(kernel) == 2
        self.mode = mode
        self.kernel = kernel

    def __call__(self, x):
        assert x.ndim == 4
        a, b = self.kernel
        upscaled = x
        if self.mode == 'repeat':
            if b > 1:
                upscaled = tt.extra_ops.repeat(upscaled, b, 3)
            if a > 1:
                upscaled = tt.extra_ops.repeat(upscaled, a, 2)
        return upscaled

    def get_output_shape(self, input_shape):
        assert len(input_shape) == 4
        return input_shape[:2] + [input_shape[2] * self.kernel[0], input_shape[3] * self.kernel[1]]


class BatchNorm(Leaf):
    """
    References
        http://deeplearning.net/software/theano/library/tensor/nnet/bn.html
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
    """
    def __init__(self, input_shape, eps=1e-4, momentum=0.1, estimate=True, axes="per-activation"):
        """

        Args:
            input_shape:
            eps:
            momentum:
            estimate:
            axes: ('per-activation', 'spatial' or a tuple of ints)
                The axes along which the input should be normalized. 'per-activation' normalizes per activation
                and is equal to axes=(0,). 'spatial' shares normalization factors across spatial dimensions
                (i.e., all dimensions past the second), which for 4D inputs would be equal to axes=(0, 2, 3).
        """
        self.axes = axes
        self.eps = eps
        self.momentum = momentum
        self.estimate = estimate
        self.running_mean = Constant(0.0)(input_shape)
        self.running_var = Constant(1.0)(input_shape)
        if self.estimate:
            self.gamma = Uniform(0.0, 1.0)(input_shape)
            self.beta = Constant(0.0)(input_shape)

    def train(self, x):
        result = tt.nnet.bn.batch_normalization_train(x, self.gamma, self.beta,
                                                      axes=self.axes, epsilon=self.eps,
                                                      running_average_factor=self.momentum,
                                                      running_mean=self.running_mean, running_var=self.running_var)
        out, mean, invstd, self.running_mean, self.running_var = result
        return out

    def test(self, x):
        return tt.nnet.bn.batch_normalization_test(x, self.gamma, self.beta,
                                                   self.running_mean, self.running_var, self.axes, self.eps)

    def __call__(self, x):
        return ifelse(self.is_train, self.train(x), self.test(x))


class RNNImpl(Enum):
    auto = 0
    ref = 1
    fused = 2
    cudnn = 3


def sequence_apply(fun, xs):
    # NOTE: this function allows batch_first or time_first
    n_time, n_batch, n_feat = xs.shape
    return fun(xs.reshape([n_time, n_batch, n_feat])).reshape([n_time, n_batch, -1])


class Recurrent(Leaf):
    def __init__(self, n_input, n_output, activation=tt.tanh, weight_init=XavierNormal(1.0), bias_init=Constant(0.0), impl=RNNImpl.auto):
        self.impl = impl
        self.n_input = n_input
        self.n_output = n_output
        self.activation = activation
        self.weight_hx = theano.shared(weight_init((n_input, n_output)), name="weight_hx")
        self.weight_hh = theano.shared(weight_init((n_output, n_output)), name="weight_hh")
        self.bias = theano.shared(bias_init(n_output), name="bias")
        self.state = None

    def ref_forward(self, xs, h0):
        def forward_one(x, h_prev):
            return self.activation(x.dot(self.weight_hx) + h_prev.dot(self.weight_hh) + self.bias)

        hs, self.updates = theano.scan(fn=forward_one, sequences=xs, outputs_info=h0)
        return hs

    def fused_forward(self, xs, h0):
        def forward_one(x, h_prev):
            return self.activation(x + h_prev.dot(self.weight_hh))

        hxs = sequence_apply(lambda x: x.dot(self.weight_hx) + self.bias, xs)
        hs, self.updates = theano.scan(fn=forward_one, sequences=hxs, outputs_info=h0)
        return hs

    def cudnn_forward(self, xs, h0):
        raise NotImplementedError("TODO: impl with theano.gpuarray.dnn.RNNBlock")

    def __call__(self, xs, h0):
        """

        Args:
            xs: (n_time, n_batch, n_input) input sequence
            h0: (n_batch, n_output) recurrent initial state

        Returns:
            hs: (n_time, n_batch, n_output) output sequence

        """
        if self.impl == RNNImpl.auto:
            return self.fused_forward(xs, h0)
        return getattr(self, self.impl.name + "_forward")(xs, h0)


class RNNBase(Leaf):
    """
    Base class for RNN with CUDNN APIs

    Notes:
        - do not use this class directly
        - currently RNN-RELU and RNN-TANH are not implemented (theano.gpuarray.dnn.RNNBlock might not have that?)
    """
    def __init_subclass__(cls, rnn_type):
        cls.rnn_type = rnn_type
        cls.non_cudnn_params = []
        cls.params = []
        cls.n_batch = 1
        cls.hidden_dim = 0
        cls.input_dim = 0
        cls._impl = None
        cls._rnn_block = None  # for cudnn

    def _params_to_cudnn(self):
        from theano.gpuarray import dnn
        from theano.gpuarray.type import gpuarray_shared_constructor
        assert dnn.dnn_available(None)
        self._rnn_block = dnn.RNNBlock(theano.config.floatX, self.hidden_dim, num_layers=1,
                                       input_mode="linear", rnn_mode=self.rnn_type, direction_mode="unidirectional")
        param_size = self._rnn_block.get_param_size([self.n_batch, self.input_dim])  # TODO: study about n_batch
        self.params = [gpuarray_shared_constructor(Constant(0.0)(param_size))]
        cs = self._rnn_block.split_params(self.params[0], layer=0, input_size=[self.n_batch, self.input_dim])   # TODO: multi layer support
        for c, p in zip(cs, self.non_cudnn_params):
            c[:] = p.get_value(borrow=True, return_internal_type=True)

    def _params_from_cudnn(self):
        assert self._rnn_block is not None, "[BUG] you should call self._params_to_cudnn() before"
        assert len(self.params) == 1
        cudnn_params = self._rnn_block.split_params(self.params[0], 0, [self.n_batch, self.input_dim])
        for np, cp in zip(self.non_cudnn_params, cudnn_params):
            np.set_value(cp)
        self.params = self.non_cudnn_params

    def cudnn_forward(self, xs, h0, c0=None):
        # self.impl = RNNImpl.cudnn
        if self.rnn_type == "lstm":
            assert c0 is not None
            c0 = c0.reshape([1, *h0.shape])
        h0 = h0.reshape([1, *h0.shape])
        results = self._rnn_block.apply(self.params[0], xs, hx=h0, cx=c0)  # TODO: multi layer support
        if self.rnn_type == "lstm":
            ys, h_next, c_next = results
            return ys, h_next[0], c_next[0]  # TODO: multi layer support
        else:
            ys, h_next = results
            return ys, h_next[0]  # TODO: multi layer support

    @property
    def impl(self):
        return self._impl

    @impl.setter
    def impl(self, v):
        if v == RNNImpl.auto:
            try:
                self.impl = RNNImpl.cudnn
            except AssertionError as e:
                logger.warning("cudnn is not available: {}".format(e))
                self.impl = RNNImpl.fused
            except ImportError as e:
                logger.warning("warning: cuda or libgpuarray is not available: {}".format(e))
                self.impl = RNNImpl.fused
        else:
            if self._impl == RNNImpl.cudnn and v != RNNImpl.cudnn:
                self._params_from_cudnn()
            elif self._impl != RNNImpl.cudnn and v == RNNImpl.cudnn:
                self._params_to_cudnn()
            self._impl = v


class GRU(RNNBase, rnn_type="gru"):
    def __init__(self, input_dim, output_dim, weight_init=XavierNormal(), bias_init=Constant(0.0), name="",
                 impl=RNNImpl.auto, n_batch=1):
        self.n_batch = n_batch
        self.name = name
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.output_dim = output_dim
        self.params = []  # TODO: cuDNN conversion
        self.non_cudnn_params = []

        def register(init, shape, name):
            v = theano.shared(init(shape), name=name)
            self.params.append(v)
            self.non_cudnn_params.append(v)
            return v

        # NOTE: do not change this initialization order because of cuDNN transfer
        self.W_r = register(weight_init, (input_dim, output_dim), name=name + ".W_r")
        self.b_wr = register(bias_init, (output_dim,), name=name + ".b_wr")

        self.W_i = register(weight_init, (input_dim, output_dim), name=name + ".W_i")
        self.b_wi = register(bias_init, (output_dim,), name=name + ".b_wi")

        self.W_h = register(weight_init, (input_dim, output_dim), name=name + ".W_h")
        self.b_wh = register(bias_init, (output_dim,), name=name + ".b_wh")

        self.R_r = register(weight_init, (output_dim, output_dim), name=name + ".R_r")
        self.b_rr = register(bias_init, (output_dim,), name=name + ".b_rr")

        self.R_i = register(weight_init, (output_dim, output_dim), name=name + ".R_i")
        self.b_ru = register(bias_init, (output_dim,), name=name + ".b_ru")

        self.R_h = register(weight_init, (output_dim, output_dim), name=name + ".R_h")
        self.b_rh = register(bias_init, (output_dim,), name=name + ".b_rh")

        self.impl = impl  # NOTE this should be set after all the initialization of params

    def ref_forward(self, xs, h0):
        def ref_forward_one(x, h_prev):
            i_t = tt.nnet.sigmoid(
                tt.dot(x, self.W_i) + tt.dot(h_prev, self.R_i) + self.b_wi + self.b_ru)
            r_t = tt.nnet.sigmoid(
                tt.dot(x, self.W_r) + tt.dot(h_prev, self.R_r) + self.b_wr + self.b_rr)

            h_hat_t = tt.tanh(
                tt.dot(x, self.W_h) + (r_t * (tt.dot(h_prev, self.R_h) + self.b_rh)) + self.b_wh)

            h_curr = ((1.0 - i_t) * h_hat_t) + (i_t * h_prev)
            return h_curr

        states, self.updates = theano.scan(fn=ref_forward_one, sequences=xs, outputs_info=h0)
        return states, states[-1]

    def fused_forward(self, xs, h0):
        # pre fused gemm
        Ws = tt.concatenate([self.W_i, self.W_r, self.W_h], axis=1)
        bs = tt.concatenate([self.b_wi + self.b_ru, self.b_wr + self.b_rr, self.b_wh])
        hxs = sequence_apply(lambda x: tt.dot(x, Ws) + bs, xs)
        output_sizes = [self.output_dim * 2, self.output_dim]
        hxs = tt.split(hxs, splits_size=output_sizes, n_splits=len(output_sizes), axis=2)
        Rs = tt.concatenate([self.R_i, self.R_r, self.R_h], axis=1)

        def fused_forward_one(hx_ir_t, hx_hat_t, h_prev):
            # second fused gemm
            hhs = tt.dot(h_prev, Rs)
            hh_ir_t, hh_hat_t = tt.split(hhs, output_sizes, len(output_sizes), axis=1)
            ir_t = tt.nnet.sigmoid(hx_ir_t + hh_ir_t)
            i_t, r_t = tt.split(ir_t, [self.output_dim] * 2, 2, axis=1)
            h_hat_t = tt.tanh(hx_hat_t + (r_t * (hh_hat_t + self.b_rh)))
            h_curr = ((1.0 - i_t) * h_hat_t) + (i_t * h_prev)
            return h_curr

        states, self.updates = theano.scan(fn=fused_forward_one, sequences=hxs, outputs_info=h0)
        return states, states[-1]

    def __call__(self, xs, h0):
        """

        Args:
            xs: (n_time, n_batch, n_input)
            h0: (n_batch, n_output)

        Returns:
            hs: (n_time, n_batch, n_output)
            h_next: ()
        """
        return getattr(self, self._impl.name + "_forward")(xs, h0)


class LSTM(RNNBase, rnn_type="lstm"):
    def __init__(self, input_dim, output_dim, weight_init=XavierNormal(), bias_init=Constant(0.0), name="",
                 impl=RNNImpl.auto, n_batch=1):
        self.n_batch = n_batch
        self.input_dim = input_dim
        self.hidden_dim = output_dim
        self.output_dim = output_dim
        self.params = []  # TODO: cuDNN conversion
        self.non_cudnn_params = []

        def register(init, shape, name):
            v = theano.shared(init(shape), name=name)
            self.params.append(v)
            self.non_cudnn_params.append(v)
            return v

        self.W_i = register(weight_init, (input_dim, output_dim), name=name + ".W_i")
        self.b_wi = register(bias_init, (output_dim,), name=name + ".b_wi")

        self.W_f = register(weight_init, (input_dim, output_dim), name=name + ".W_f")
        self.b_wf = register(bias_init, (output_dim,), name=name + ".b_wf")

        self.W_c = register(weight_init, (input_dim, output_dim), name=name + ".W_c")
        self.b_wc = register(bias_init, (output_dim,), name=name + ".b_wc")

        self.W_o = register(weight_init, (input_dim, output_dim), name=name + ".W_o")
        self.b_wo = register(bias_init, (output_dim,), name=name + ".b_wo")

        self.R_i = register(weight_init, (output_dim, output_dim), name=name + ".R_i")
        self.b_ri = register(bias_init, (output_dim,), name=name + ".b_ri")

        self.R_f = register(weight_init, (output_dim, output_dim), name=name + ".R_f")
        self.b_rf = register(bias_init, (output_dim,), name=name + ".b_rf")

        self.R_c = register(weight_init, (output_dim, output_dim), name=name + ".R_c")
        self.b_rc = register(bias_init, (output_dim,), name=name + ".b_rc")

        self.R_o = register(weight_init, (output_dim, output_dim), name=name + ".R_o")
        self.b_ro = register(bias_init, (output_dim,), name=name + ".b_ro")

        self.impl = impl  # NOTE this should be set after all the initialization of params

    def ref_forward(self, xs, h_init, c_init):
        # self.impl = RNNImpl.ref
        def ref_forward_one(x_t, h_tm1, c_tm1):
            i_t = tt.nnet.sigmoid(
                tt.dot(x_t, self.W_i) + tt.dot(h_tm1, self.R_i) + self.b_wi + self.b_ri)
            f_t = tt.nnet.sigmoid(
                tt.dot(x_t, self.W_f) + tt.dot(h_tm1, self.R_f) + self.b_wf + self.b_rf)
            o_t = tt.nnet.sigmoid(
                tt.dot(x_t, self.W_o) + tt.dot(h_tm1, self.R_o) + self.b_ro + self.b_wo)

            c_hat_t = tt.tanh(
                tt.dot(x_t, self.W_c) + tt.dot(h_tm1, self.R_c) + self.b_wc + self.b_rc)
            c_t = f_t * c_tm1 + i_t * c_hat_t
            h_t = o_t * tt.tanh(c_t)
            return h_t, c_t

        [hs, cs], self.updates = theano.scan(fn=ref_forward_one, sequences=xs, outputs_info=[h_init, c_init])
        return hs, hs[-1], cs[-1]

    def fused_forward(self, xs, h_init, c_init):
        # self.impl = RNNImpl.fused
        Ws = tt.concatenate([self.W_i, self.W_f, self.W_o, self.W_c], axis=1)
        bs = tt.concatenate([self.b_wi + self.b_ri, self.b_wf + self.b_rf, self.b_ro + self.b_wo, self.b_rc])
        hxs = sequence_apply(lambda x: x.dot(Ws) + bs, xs)
        Rs = tt.concatenate([self.R_i, self.R_f, self.R_o, self.R_c], axis=1)

        def fused_forward_one(hx, h_prev, c_prev):
            hs = hx + tt.dot(h_prev, Rs)
            h_ifo, h_hat = tt.split(hs, [self.output_dim * 3, self.output_dim], 2, axis=1)

            i_t, f_t, o_t = tt.split(tt.nnet.sigmoid(h_ifo), [self.output_dim] * 3, 3, axis=1)
            c_hat_t = tt.tanh(h_hat)

            c_t = f_t * c_prev + i_t * c_hat_t
            h_t = o_t * tt.tanh(c_t)
            return h_t, c_t

        [hs, cs], self.updates = theano.scan(fn=fused_forward_one, sequences=hxs, outputs_info=[h_init, c_init])
        return hs, hs[-1], cs[-1]

    # def cudnn_forward(self, xs, h0, c0):
    #     # TODO: multi layer support
    #     from theano.gpuarray import dnn
    #     from theano.gpuarray.type import gpuarray_shared_constructor
    #     assert dnn.dnn_available(None)
    #     rnnb = dnn.RNNBlock(theano.config.floatX, self.hidden_dim, num_layers=1,
    #                         input_mode="linear", rnn_mode="lstm", direction_mode="unidirectional")
    #     param_size = rnnb.get_param_size([self.n_batch, self.input_dim])  # TODO: study about n_batch
    #     param_cudnn = gpuarray_shared_constructor(Constant(0.0)(param_size))
    #     cs = rnnb.split_params(param_cudnn, layer=0, input_size=[self.n_batch, self.input_dim])
    #     for c, p in zip(cs, self.non_cudnn_params):
    #         c[:] = p.get_value(borrow=True, return_internal_type=True)
    #     self.cudnn_params = [param_cudnn]
    #
    #     # for multi layer (depth, batch, hidden)
    #     h0 = h0.reshape([1, *h0.shape])
    #     c0 = c0.reshape([1, *c0.shape])
    #     ys, h_next, c_next = rnnb.apply(param_cudnn, xs, h0, c0)  # TODO: multi layer support
    #     return ys, h_next[0], c_next[0]  # TODO: multi layer support

    def __call__(self, xs, h_init, c_init):
        if self.impl == RNNImpl.auto:
            try:
                return self.cudnn_forward(xs, h_init, c_init)
            except AssertionError:
                print("warning: cudnn is not available")
            except ImportError:
                print("warning: cuda or libgpuarray is not available")
            return self.fused_forward(xs, h_init, c_init)
        return getattr(self, self.impl.name + "_forward")(xs, h_init, c_init)


class ConvLSTM(Leaf):
    def __init__(self, n_input_ch, n_output_ch, kernel, weight_init=XavierNormal(), bias_init=Constant(0.0)):
        assert kernel % 2 == 1
        self.kernel = kernel
        self.pad = (kernel - 1) // 2
        self.n_output_ch = n_output_ch
        self.conv_wx = Conv1D(n_input_ch, n_output_ch * 4, kernel=self.kernel, pad=self.pad,
                              weight_init=weight_init, bias_init=bias_init)
        self.conv_wh = Conv1D(n_output_ch, n_output_ch * 4, kernel=self.kernel, pad=self.pad,
                              weight_init=weight_init, bias_init=bias_init)

    def __call__(self, xs, h0, c0):
        """

        Args:
            xs: (n_batch, n_input_ch, n_freq, n_time)
            h0: (n_batch, n_input_ch, n_freq)
            c0: (n_batch, n_input_ch, n_freq)

        Returns:
            hs (n_batch, n_output_ch, n_freq, n_time)
        """
        n_batch, n_input_ch, n_freq, n_time = xs.shape
        xs_batch = xs.transpose([3, 0, 1, 2]).reshape([n_time * n_batch, n_input_ch, n_freq])
        hxs = self.conv_wx(xs_batch)                                    # (n_time * n_batch, n_output_ch * 4, n_freq)
        hxs = hxs.reshape([n_time, n_batch, self.n_output_ch * 4, -1])  # (n_time, n_batch, n_output_ch * 4, n_freq)

        def step(hx, h_prev, c_prev):
            hss = hx + self.conv_wh(h_prev)  # (n_batch, n_output_ch, n_freq)
            h_ifo, h_hat = tt.split(hss, [self.n_output_ch * 3, self.n_output_ch], 2, axis=1)
            i, f, o = tt.split(tt.nnet.sigmoid(h_ifo), [self.n_output_ch] * 3, 3, axis=1)
            c_hat = tt.tanh(h_hat)
            c = f * c_prev + i * c_hat
            h = o * tt.tanh(c)
            return h, c

        hs, self.updates = theano.scan(step, sequences=hxs, outputs_info=[h0, c0])  # (n_time, n_batch, n_output_ch, n_freq)
        return hs.transpose([1, 2, 3, 0])


class ConvGRU(Leaf):
    def __init__(self, n_input_ch, n_output_ch, kernel, weight_init=XavierNormal(), bias_init=Constant(0.0)):
        assert kernel % 2 == 1
        self.kernel = kernel
        self.pad = (kernel - 1) // 2
        self.n_output_ch = n_output_ch
        self.conv_wx = Conv1D(n_input_ch, n_output_ch * 3, kernel=self.kernel, pad=self.pad,
                              weight_init=weight_init, bias_init=bias_init)
        self.conv_wh = Conv1D(n_output_ch, n_output_ch * 3, kernel=self.kernel, pad=self.pad,
                              weight_init=weight_init, bias_init=bias_init)

    def __call__(self, xs, h0):
        """

        Args:
            xs: (n_batch, n_input_ch, n_freq, n_time)
            h0: (n_batch, n_input_ch, n_freq)

        Returns:
            hs (n_batch, n_output_ch, n_freq, n_time)
        """
        n_batch, n_input_ch, n_freq, n_time = xs.shape
        xs_batch = xs.transpose([3, 0, 1, 2]).reshape([n_time * n_batch, n_input_ch, n_freq])
        hxs = self.conv_wx(xs_batch)                                    # (n_time * n_batch, n_output_ch * 3, n_freq)
        hxs = hxs.reshape([n_time, n_batch, self.n_output_ch * 4, -1])  # (n_time, n_batch, n_output_ch * 3, n_freq)
        ss = (self.n_output_ch * 2, self.n_output_ch)
        hxs = tt.split(hxs, splits_size=ss, n_splits=len(ss), axis=3)

        def step(hx_ir, hx_hat, h_prev):
            hhs = self.conv_wh(h_prev)
            hh_ir, hh_hat = tt.split(hhs, splits_size=ss, n_splits=len(ss), axis=1)
            ir = tt.nnet.sigmoid(hx_ir + hh_ir)
            i, r = tt.split(ir, [self.n_output_ch] * 2, 2, axis=1)
            hat = tt.tanh(hx_hat + r * hh_hat)
            return (1.0 - i) * hat + i * h_prev

        hs, self.updates = theano.scan(step, sequences=hxs, outputs_info=h0)  # (n_time, n_batch, n_output_ch)
        return hs.transpose([1, 2, 0])


class Embed(Leaf):
    """
    ref: http://deeplearning.net/tutorial/rnnslu.html#word-embeddings
    """

    def __init__(self, n_vocab, n_output, weight_init=XavierNormal()):
        self.n_vocab = n_vocab
        self.n_output = n_output
        self.weight = theano.shared(weight_init((n_vocab, n_output)))

    def __call__(self, ids):
        return self.weight[ids].reshape([ids.shape[0], self.n_output])


class Dropout(Leaf):
    # http://deeplearning.net/tutorial/lstm.html
    def __init__(self, connect_prob=0.5):
        self.connect_prob = connect_prob

    def __call__(self, x):
        return ifelse(self.is_train,
                      x * rng.binomial(x.shape, p=self.connect_prob, n=1, dtype=x.dtype),
                      x * self.connect_prob)


class MLP(Leaf):
    def __init__(self, n_input, n_hidden, n_output, n_layers, activation=tt.nnet.relu, last_activation=False, **kwargs):
        assert n_layers > 0
        self.last_activation = last_activation
        self.n_layers = n_layers
        self.activation = activation
        for n in range(1, self.n_layers + 1):
            n_in = n_input if n == 1 else n_hidden
            n_out = n_output if n == n_layers else n_hidden
            l = Linear(n_in, n_out, **kwargs)
            setattr(self, "l%d" % n, l)

    def __call__(self, x):
        for n in range(1, self.n_layers):
            x = getattr(self, "l%d" % n)(x)
            x = self.activation(x)

        x = getattr(self, "l%d" % self.n_layers)(x)
        if self.last_activation:
            x = self.activation(x)
        return x


class Sequential(Leaf):
    def __init__(self, *args, **kwargs):
        self.leaf_dict = None
        if len(args) == 1 and len(kwargs) == 0:
            self.leaf_dict = args[0]
        elif len(args) == 0 and len(kwargs) != 0:
            if sys.version_info.major == 3 and sys.version_info.minor >= 6:
                raise AssertionError("Python >= 3.6 is required to preserve kwargs order. Use OrderedDict instead.")
            self.leaf_dict = OrderedDict(("leaf%d" % i, l) for i, l in enumerate(args))
        elif len(kwargs) == 0 and len(args) != 0:
            self.leaf_dict = kwargs
        else:
            raise RuntimeError("you can use only args or kwargs, cannot use both.")

        for k, v in self.leaf_dict:
            setattr(self, k, v)

    def __call__(self, x):
        for v in self.leaf_dict.values():
            x = v(x)
        return x
