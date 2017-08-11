from collections import OrderedDict

import theano
import theano.tensor as tt
from theano.ifelse import ifelse

from t721.initializer import Constant


# TODO: make lr shared?
class Optimizer:
    def __init_subclass__(cls):
        cls.lr = 1.0
        cls.state_keys = []

    def update_one(self, param, grad, state_dict):
        raise NotImplementedError("implement in a derived class")

    def updates(self, params, cost: tt.Variable) -> OrderedDict:
        updates = OrderedDict()
        for p in params:
            d = None
            if hasattr(self, "state_keys"):
                d = {k: theano.shared(Constant(0.0)(p.get_value().shape)) for k in self.state_keys}

            g = tt.grad(cost=cost, wrt=p)
            us = self.update_one(p, g, d)
            updates.update(us)
        return updates


class SGD(Optimizer):
    def __init__(self, lr=1e-3, momentum=0.0, nestrov=False):
        self.lr = lr
        self.momentum = momentum
        self.nestrov = nestrov
        self.state_keys = []
        if self.momentum != 0.0:
            self.state_keys = ["v"]

    def update_one(self, param, grad, state_dict):
        if self.momentum == 0.0:
            return (param, param - self.lr * grad),

        v = state_dict["v"]
        next_v = self.momentum * v - self.lr * grad
        if self.nestrov:
            next_p = param - self.momentum * v + (1 + self.momentum) * next_v
        else:
            next_p = param + next_v
        return (v, next_v), (param, next_p)


class Adadelta(Optimizer):
    """
    ref: http://www.matthewzeiler.com/pubs/googleTR2012/googleTR2012.pdf
    """

    def __init__(self, lr=1.0, decay_rate=0.95, epsilon=1e-6):
        self.lr = lr
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.state_keys = ["g", "s"]

    def update_one(self, param, grad, state_dict):
        s, g = state_dict["s"], state_dict["g"]
        next_g = self.decay_rate * g + (1 - self.decay_rate) * grad ** 2
        next_param = param - self.lr * grad * (s + self.epsilon) ** 0.5 / (next_g + self.epsilon) ** 0.5
        next_s = self.decay_rate * s + (1.0 - self.decay_rate) * (next_param - param) ** 2
        return (param, next_param), (s, next_s), (g, next_g)


class Adam(Optimizer):
    """
    References:
         http://arxiv.org/abs/1412.6980
         http://cs231n.github.io/neural-networks-3/
    """

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.state_keys = ["m", "v"]

    def update_one(self, param, grad, state_dict):
        m = state_dict["m"]
        v = state_dict["v"]
        next_m = m + (1.0 - self.beta1) * (grad - m)
        next_v = v + (1.0 - self.beta2) * (grad ** 2 - v)
        next_p = param - self.lr * next_m / (next_v ** 0.5 + self.epsilon)
        return (param, next_p), (m, next_m), (v, next_v)


class L1Penalty(Optimizer):
    def __init__(self, base_optimizer : Optimizer, decay_rate=1e-3):
        """

        Args:
            base_optimizer: should be instantiated
            decay_rate: float value
        """
        assert isinstance(base_optimizer, Optimizer)
        self.base_optimizer = base_optimizer
        self.decay_rate = decay_rate

    def update_one(self, param, grad, state_dict):
        updates = OrderedDict(self.base_optimizer.update_one(param, grad, state_dict))
        updates[param] -= self.base_optimizer.lr * self.decay_rate * tt.sgn(param)
        return updates


class L2Penalty(Optimizer):
    """
    a.k.a. weight decay
    """
    def __init__(self, base_optimizer : Optimizer, decay_rate=1e-3):
        assert isinstance(base_optimizer, Optimizer)
        self.base_optimizer = base_optimizer
        self.state_keys = base_optimizer.state_keys
        self.decay_rate = decay_rate

    def update_one(self, param, grad, state_dict):
        updates = OrderedDict(self.base_optimizer.update_one(param, grad, state_dict))
        updates[param] -= self.base_optimizer.lr * self.decay_rate * param
        return updates


class GradientClip(Optimizer):
    def __init__(self, base_optimizer: Optimizer, threshold=1.0, norm=2):
        self.norm = norm
        self.base_optimizer = base_optimizer
        self.state_keys = base_optimizer.state_keys
        self.threshold = threshold

    def updates(self, params, cost: tt.Variable) -> OrderedDict:
        updates = OrderedDict()
        params = list(params)
        sum_grad = 0.0
        for p in params:
            d = None
            if hasattr(self, "state_keys"):
                d = {k: theano.shared(Constant(0.0)(p.get_value().shape)) for k in self.state_keys}

            g = tt.grad(cost=cost, wrt=p)
            sum_grad += g.norm(self.norm)
            us = self.base_optimizer.update_one(p, g, d)
            updates.update(us)

        scale = ifelse(sum_grad > self.threshold, self.threshold / sum_grad, 1.0)
        for p in params:
            updates[p] *= scale
        return updates


    # class YellowFin(Optimizer):
#     def __init__(self, lr=0.1, mu=0.0, clip_thresh=None, weight_decay=0.0, beta=0.999,
#                  curv_win_width=20, zero_debias=True, delta_mu=0.0, auto_clip_fac=None):
#         self.base_optimizer = SGD(lr=lr, momentum=)
#         self.state_keys =