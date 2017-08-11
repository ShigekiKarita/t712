from typing import List

import numpy
import theano


floatX = theano.config.floatX


class Initializer:
    def __call__(self, shape: List[int]) -> numpy.ndarray:
        pass


class Constant(Initializer):
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, shape: List[int]) -> numpy.ndarray:
        return numpy.zeros(shape, dtype=floatX) + self.value


class Uniform(Initializer):
    def __init__(self, a=1.0, b=None):
        if b is None:
            b = -a
        if b < a:
            b, a = a, b
        self.lower = a
        self.upper = b

    def __call__(self, shape: List[int]) -> numpy.ndarray:
        return numpy.random.uniform(-self.lower, self.upper, shape).astype(floatX)


class Normal(Initializer):
    def __init__(self, mean=0.0, stddev=1.0):
        self.stddev = stddev
        self.mean = mean

    def __call__(self, shape: List[int]) -> numpy.ndarray:
        return numpy.random.normal(self.mean, self.stddev, shape).astype(floatX)


class XavierUniform(Initializer):
    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, shape: List[int]) -> numpy.ndarray:
        fan_in, fan_out = calculate_fan_in_and_fan_out(shape)
        std = self.gain * (2.0 / (fan_in + fan_out)) ** 0.5
        a = 3.0 ** 0.5 * std
        return numpy.random.uniform(-a, a, shape).astype(floatX)


class XavierNormal(Initializer):
    def __init__(self, gain=1.0):
        self.gain = gain

    def __call__(self, shape: List[int]) -> numpy.ndarray:
        fan_in, fan_out = calculate_fan_in_and_fan_out(shape)
        std = self.gain * (2.0 / (fan_in + fan_out)) ** 0.5
        return numpy.random.normal(0.0, std, shape).astype(floatX)


def calculate_gain(nonlinearity):
    """Return the recommended gain value for the given nonlinearity function. The values are as follows:
    ============ ==========================================
    nonlinearity gain
    ============ ==========================================
    linear       :math:`1`
    conv{1,2,3}d :math:`1`
    sigmoid      :math:`1`
    tanh         :math:`5 / 3`
    relu         :math:`\sqrt{2}`
    leaky_relu   :math:`\sqrt{2 / (1 + negative\_slope^2)}`
    ============ ==========================================

    Args:
        nonlinearity: the nonlinear function (`nn.functional` name)
        param: optional parameter for the nonlinear function
    """
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return 2.0 ** 0.5
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))


def calculate_fan_in_and_fan_out(shape):
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with less than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        num_input_fmaps = shape[0]
        num_output_fmaps = shape[1]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = numpy.prod(shape[2:])
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out