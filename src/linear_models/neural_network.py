from dataclasses import dataclass
from typing import List, Callable
from functools import partial
from linear_models.SGD import _BaseSGD
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum


def _linear(x):
    return x


def _relu(x):
    return x if x > 0 else 0


def _sigmoid(x):
    return 1 / (1 + np.e ** -x)


def _tanh(x):
    return 2 / (1 + np.e ** (-2 * x)) - 1


def _linear_dx(x):
    return np.ones_like(x)


def _relu_dx(x):
    return (x > 0).astype(int)


def _sigmoid_dx(x):
    return np.e ** x / (np.e ** -x + 1) ** 2


def _tanh_dx(x):
    return 4 * np.e ** (2 * x) / (np.e ** -(2 * x) + 1) ** 2


class ActivationFunction(Enum):
    RELU = partial(_relu)
    SIGMOID = partial(_sigmoid)
    TANH = partial(_tanh)
    LINEAR = partial(_linear)

    @property
    def derivative(self):
        if self == ActivationFunction.RELU:
            return _relu_dx
        elif self == ActivationFunction.SIGMOID:
            return _sigmoid_dx
        elif self == ActivationFunction.TANH:
            return _tanh_dx
        elif self == ActivationFunction.LINEAR:
            return _linear_dx

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


@dataclass
class Node:
    activation: Callable
    is_intercept: bool = False
    incoming_weights: np.ndarray = None
    input_val: int = 0
    output_val: int = 0

    def __init__(self, activation, intercept=False):
        self.activation = activation
        if intercept:
            self.is_intercept = True
            self.input_val = 1
            self.output_val = 1

    def activate(self):
        self.output_val = self.activation(self.input_val)


@dataclass
class MetaLayer(ABC):
    nodes: List[Node]
    activation: ActivationFunction
    is_output_layer: bool

    @abstractmethod
    def activate(self):
        pass

    @abstractmethod
    def get_outputs(self):
        pass

    @abstractmethod
    def get_inputs(self):
        pass

    @abstractmethod
    def get_bias(self):
        pass


@dataclass
class Layer(MetaLayer):
    prev_layer: MetaLayer
    is_output_layer = False

    def __init__(self, layer_size, activation, prev_layer):
        self.activation = activation
        self.nodes = [Node(activation, intercept=True)]
        self.nodes = self.nodes + [Node(activation) for _ in range(layer_size)]
        self.prev_layer = prev_layer
        if self.prev_layer is not None:
            self.prev_layer.next_layer = self

    def activate(self):
        for node in self.nodes:
            node.activate()

    def get_outputs(self):
        if self.is_output_layer:
            return np.asarray([node.output_val for node in self.nodes if not node.is_intercept])

        return np.asarray([node.output_val for node in self.nodes])\
            .reshape((1, len(self.nodes)))

    def get_inputs(self):
            return np.asarray([node.input_val for node in self.nodes if not node.is_intercept])\
                .reshape(1, len(self.nodes) - 1)

    def update_weights(self, weights):
        for node, weight in zip(self.nodes[1:], weights):
            node.incoming_weights = weight

    def get_bias(self):
        return self.nodes[1].incoming_weights[0]  # This can be any node that isn't the intercept node


@dataclass
class NeuralNetwork(_BaseSGD):
    layers: List[Layer]

    def __init__(
            self,
            nodes: List[int],
            activations: List[callable],
            max_iters_no_change: int = 5,
            learning_rate=1e-3,
            batch_size: int = 1,
            epsilon=1e-4,
            max_iters: int = 1000):
        super(NeuralNetwork, self).__init__()
        self.max_iters_no_change = max_iters_no_change
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.layers = []

        if len(activations) != len(nodes):
            print('Length of activation functions must be equal to the number of hidden layers.')
            return
        # Copy and add 1 to layer sizes to have an output layer
        layer_sizes = nodes.copy()
        layer_sizes.append(1)
        activations.append(ActivationFunction.LINEAR)
        prev_layer = None
        for cur_size, act in zip(layer_sizes, activations):
            cur_layer = Layer(cur_size, act, prev_layer)
            self.layers.append(cur_layer)
            prev_layer = cur_layer

        self.layers[-1].is_output_layer = True

    def _update_rule(self, x, y, w):
        """
        This is the method that updates the list of weight matrices for gradient descent

        :param x: An mini-batch of observations that we want to make predictions about
        :param y: The expected prediction for the observation(s)
        :param w: The list of weight matrices for all layers in the model

        :return: Returns the value that determines how much the weight matrices should change
        """
        total_weight_updates = []
        for x0, y0 in zip(x, y.T):
            weight_updates = []
            x0 = x0.reshape(1, len(x0))
            y0 = y0.reshape(1, len(y0))
            # print(f'y: {y0}')
            # print(f'x: {x0}')
            # print(f'Prediction: {self.predict(x0)}')
            predictions = [pred[0] for pred in self.predict(x0)]
            delta_l1 = np.subtract(predictions, y0)
            weights_l1 = None
            for weights_l, l in zip(w[::-1], self.layers[::-1]):
                activation_derivative = l.activation.derivative
                z_l = l.get_inputs()
                # print(f'z(l): {z_l}')
                weights_l1 = np.asarray([1]) if l.is_output_layer else weights_l1[:, 1:]
                # print(f'w(l+1): {weights_l1}')
                g_primes = activation_derivative(z_l)
                # print(f'g\'(z(l)): {g_primes}')
                #
                # print(f'd(l1): {delta_l1}')
                # print(f'dw: {np.dot(delta_l1, weights_l1)}')
                delta_l = np.multiply(
                    np.dot(
                        delta_l1,
                        weights_l1),
                    g_primes)
                # print(f'd(l+1) * w(l+1): {np.dot(delta_l1, weights_l1)}')
                # print(f'd(l): {delta_l}')

                h_l1 = l.prev_layer.get_outputs() \
                    if l.prev_layer is not None \
                    else np.insert(x0, 0, 1).reshape(1, x0.shape[1] + 1)
                # print(f'h(l-1): {h_l1}')
                updates = np.outer(h_l1, delta_l)
                # print(f'w(l): {weights_l}')
                # print(f'∂C/∂w(l): {updates.T}')
                weights_l1 = weights_l
                delta_l1 = delta_l
                weight_updates.append(updates.T)
                # print()

            if total_weight_updates is not None:
                total_weight_updates = weight_updates
            else:
                total_weight_updates = \
                    [np.add(w1, w2)
                     for w1, w2
                     in zip(total_weight_updates, weight_updates)]

        # Reverse weights updates since we are iterating through the network in a backward direction
        return total_weight_updates[::-1]

    def _fit(self, x, y, **kwargs):
        return super()._fit(x, y, **kwargs)

    def feed_forward(self, x):
        input_layer = np.insert(x, 0, 1)
        for layer in self.layers:
            for node in layer.nodes:
                if node.is_intercept:
                    node.input_val = 1
                else:
                    node.input_val = np.dot(input_layer.flatten(), node.incoming_weights.T)
                node.activate()

            input_layer = layer.get_outputs()
        return self.layers[-1].get_outputs()

    def predict(self, x):
        return [self.feed_forward(v) for v in x]

    def initialize_weights(self, x_shape):
        # We add 1 to the shape of the weights to account for the weight associated with the bias term
        for layer in self.layers:
            if layer.prev_layer is None:
                for node in layer.nodes:
                    if not node.is_intercept:
                        l1_len = x_shape[1]
                        node.incoming_weights = np.insert(np.random.randn(l1_len) * np.sqrt(2/l1_len), 0, 0)
            else:
                for node in layer.nodes:
                    if not node.is_intercept:
                        l1_len = len(layer.prev_layer.nodes) - 1
                        node.incoming_weights = np.insert(np.random.randn(l1_len) * np.sqrt(2/l1_len), 0, 0)
        return self

    def _loss(self, x, y):
        predicted = [pred[0] for pred in self.predict(x)]
        return 0.5 * np.mean((np.subtract(predicted, y)) ** 2)

    def get_weights(self):
        weights = []
        for layer in self.layers:
            weights.append(self.get_weights_at_layer(layer))

        return weights

    @staticmethod
    def get_weights_at_layer(layer):
        return np.asarray([node.incoming_weights for node in layer.nodes if not node.is_intercept])

    def update_model_params(self, weights):
        for layer, weights in zip(self.layers, weights):
            layer.update_weights(weights)

    def pprint(self):
        for l in range(len(self.layers)):
            layer = self.layers[l]
            layer_weights = self.get_weights_at_layer(layer)[:, 1:]
            layer_bias = layer.get_bias()
            print(f'Layer {l}')
            print(f'    weights:')
            print(f'      {layer_weights}')
            print(f'    bias:'
                  f'      {layer_bias}')
