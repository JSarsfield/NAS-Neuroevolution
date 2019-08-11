r"""
Defines different activation functions for evolving the CPPN which in turn produces an ANN architecture when decoded

Code taken from neat-python https://github.com/CodeReclaimers/neat-python/blob/master/neat/activations.py
Refactored to TensorFlow activation functions - for creating computational graphs with AutoGraph
Modified by:
__author__ = "Joe Sarsfield"
__email__ = "joe.sarsfield@gmail.com"
"""
import torch
import random
import math

"""
def sigmoid_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 1.0 / (1.0 + math.exp(-z))


def tanh_activation(z):
    z = max(-60.0, min(60.0, 2.5 * z))
    return math.tanh(z)


def sin_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return math.sin(z)


def cos_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return math.cos(z)


def gauss_activation(z):
    z = max(-3.4, min(3.4, z))
    return math.exp(-5.0 * z ** 2)


def relu_activation(z):
    return z if z > 0.0 else 0.0


def elu_activation(z):
    return z if z > 0.0 else math.exp(z) - 1


def lelu_activation(z):
    leaky = 0.005
    return z if z > 0.0 else leaky * z


def selu_activation(z):
    lam = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return lam * z if z > 0.0 else lam * alpha * (math.exp(z) - 1)


def softplus_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 0.2 * math.log(1 + math.exp(z))


def identity_activation(z):
    return z


def clamped_activation(z):
    return max(-1.0, min(1.0, z))


def inv_activation(z):
    try:
        z = 1.0 / z
    except ArithmeticError:  # handle overflows
        return 0.0
    else:
        return z


def log_activation(z):
    z = max(1e-7, z)
    return math.log(z)


def exp_activation(z):
    z = max(-60.0, min(60.0, z))
    return math.exp(z)


def abs_activation(z):
    return abs(z)


def hat_activation(z):
    return max(0.0, 1 - abs(z))


def square_activation(z):
    return z ** 2


def cube_activation(z):
    return z ** 3


class InvalidActivationFunction(TypeError):
    pass


def validate_activation(function):
    if not isinstance(function,
                      (types.BuiltinFunctionType,
                       types.FunctionType,
                       types.LambdaType)):
        raise InvalidActivationFunction("A function object is required.")

    if function.__code__.co_argcount != 1:  # avoid deprecated use of `inspect`
        raise InvalidActivationFunction("A single-argument function is required.")
"""


def step(x):
    return torch.tensor(1, dtype=torch.float32) if x > 0.5 else torch.tensor(0, dtype=torch.float32)


def gaussian(z):
    z = max(-3.4, min(3.4, z))
    return torch.tensor(math.exp(-5.0 * z ** 2), dtype=torch.float32)


class ActivationFunctionSet(object):
    """
    Contains the list of current valid activation functions,
    including methods for adding and getting them.
    """

    def __init__(self):
        self.functions = {}
        #self.add('step', step)
        self.add('gauss', gaussian)
        #self.add('tanh', torch.tanh)
        #self.add('sigmoid', torch.sigmoid)
        self.add('sin', torch.sin)
        self.add('cos', torch.cos)
        #self.add('log', torch.log)
        #self.add('abs', torch.abs) # unbalanced function
        # self.add('exp', torch.exp)  # unbalanced function creates loads of nodes - investigate
        #self.add('square', torch.softmax)
        #self.add('relu', torch.nn.relu)
        #self.add('elu', tf.nn.elu)
        #self.add('lelu', torch.nn.LeakyReLU)
        #self.add('crelu', tf.nn.crelu)
        #self.add('softplus', tf.nn.softplus)
        #self.add('identity', tf.math.identity)
        #self.add('clamped', tf.math.clamped)
        #self.add('inv', tf.math.inv)
        #self.add('hat', tf.math.hat)
        #self.add('cube', tf.math.cube)
        #self.add('square', tf.math.softmax)

    def add(self, name, function):
        self.functions[name] = function

    def get(self, name):
        return self.functions.get(name)

    def get_random_activation_func(self):
        return self.functions[random.choice(list(self.functions.keys()))]

    def is_valid(self, name):
        return name in self.functions


def dot(inputs, weights, bias):
    return torch.dot(inputs, weights) + bias


def diff(inputs, weights, bias):
    return (inputs[0]-inputs[1]) + bias


class NodeFunctionSet(object):
    """ function to apply to data going into node before going through activation function """

    def __init__(self):
        self.functions = {}
        self.add('dot', dot)
        self.add('diff', diff)

    def add(self, name, function):
        self.functions[name] = function

    def get(self, name):
        return self.functions.get(name)

    def get_random_activation_func(self):
        return self.functions[random.choice(list(self.functions.keys()))]