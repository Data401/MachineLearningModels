from dataclasses import dataclass
from typing import List, Callable
from timeit import default_timer as timer
import numpy as np


@dataclass
class Node:
    activation: Callable
    incoming_weights: np.ndarray[int]
    outgoing_weights: np.ndarray[int]
    input_val: int = 0
    output_val: int = 0

    def __init__(self, input_size, activation, intercept=False):
        self.incoming_weights = np.random.uniform(-1, 1, input_size).reshape((1, input_size))
        self.outgoing_weights = np.random.uniform(-1, 1, input_size).reshape((1, input_size))

        self.activation = activation
        if intercept:
            self.input_value = 1
            self.output_val = 1

    def activate(self):
        self.output_val = self.activation(self.input_val)


@dataclass
class Layer:
    activation: Callable
    nodes: List[Node]
    input_size: int

    def __init__(self, input_size, layer_size, activation):
        self.activation = activation
        self.input_size = input_size
        self.nodes = [Node(input_size, activation, intercept=True)]
        self.nodes = [Node(input_size, activation) for _ in range(layer_size)]

    def activate(self):
        for node in self.nodes:
            node.activate()

    def get_outputs(self):
        return np.ndarray(
            buffer=(node.output_val for node in self.nodes),
            shape=(1, len(self.nodes)))


@dataclass
class NeuralNetwork():
    layers: List[Layer]

    def __init__(self,
                 layers,
                 nodes,
                 activations,
                 max_iters_no_change=5,
                 learning_rate=1e-3,
                 batch_size=1,
                 epsilon=1e-4,
                 max_iters = 1e3):
        self.max_iters_no_change = max_iters_no_change
        self.learning_rate = learning_rate
        self.batch_size = batch_size,
        self.epsilon = epsilon
        self.max_iters = max_iters,
        self.layers = []

        if len(layers) != len(nodes):
            print('Length of nodes must be equal to the number of layers.')
            return
        if len(activations) != len(nodes):
            print('Length of activation functions must be equal to the number of hidden layers.')
            return
        layer_sizes = nodes
        layer_sizes.insert(0, 0)
        for prev_size, cur_size, act in zip(layer_sizes, layer_sizes[1:], activations):
            self.layers.append(Layer(prev_size, cur_size, act))

    def _change_in_loss(self, x, y, prior_weights):
        return NotImplementedError

    def _fit_by_back_prop(self, x, y):
        # """
        #         Calculates the gradient of the loss function for linear regression.
        #
        #         :param x: Column vector of explanatory variables
        #         :param y: Column vector of dependent variables
        #         :return: Vector of parameters for Linear Regression
        #         """
        # assert len(x) == len(y)
        #
        # # Reset model error calculations
        # self.errors = []
        # self.iterations = []
        #
        # # Setup Debugging/ Graphing
        # start_time = timer()
        # train_time = 0
        #
        # y0 = y.reshape(len(y), 1)  # Convert y to a column vector
        #
        # betas = np.zeros((len(x0[0]), 1))  # Makes a column vector of zeros
        #
        # n_iter = 0
        # n_iters_no_change = 0
        #
        # while n_iters_no_change < self.max_iters_no_change and n_iter < self.max_iters:
        #     permutation = np.random.permutation(x0.shape[0])
        #     x0 = x0[permutation]
        #     y0 = y0[permutation]
        #
        #     pre_epoch_betas = betas
        #     # Iterate through all (X, Y) pairs where X is a vector of predictor variables [x1, x2, x3, ...]
        #     # and Y is a vector containing the response variable
        #     with np.errstate(invalid='raise'):
        #         for v, w in zip(x0, y0):
        #             v = v.reshape(1, len(v))
        #             w = w.reshape(1, len(w))
        #             prior_betas = betas
        #             try:
        #                 loss_change = self.learning_rate * self._change_in_loss(v, w, prior_betas)
        #                 betas = np.subtract(prior_betas, loss_change)
        #             except FloatingPointError:
        #                 raise ConvergenceError()
        #
        #     total_error = np.sqrt(np.sum(np.subtract(betas, pre_epoch_betas) ** 2))
        #     n_iters_no_change = n_iters_no_change + 1 if total_error < self.epsilon else 0
        #     n_iter += 1
        #     train_time = timer() - start_time
        #     if verbose > 0:
        #         print(
        #             f'-- Epoch {n_iter}\n'
        #             f'Total training time: {round(train_time, 3)}')
        #         if verbose > 1:
        #             print(f'Equation:\n'
        #                   f'y = {np.round(betas[1:][0][0], 3)}(x1) + {np.round(betas[1:][1][0], 3)}(x2) + {np.round(betas[0][0], 3)}')
        #         if verbose > 2:
        #             print(
        #                 f'Pre Epoch Betas:\n{pre_epoch_betas}\n'
        #                 f'Post Epoch Betas:\n{betas}\n')
        #     self.iterations.append(n_iter)
        #     self.errors.append(total_error)
        #
        # self.coef_ = betas[1 if self.fit_intercept else 0:]
        # self.intercept_ = betas[0][0] if self.fit_intercept else 0  # betas[0] gives a series with a single value
        # if verbose > 0:
        #     print(f'SGD converged after {n_iter} epochs.\n'
        #           f'Total Training Time: {round(train_time, 3)} sec.')
        #
        # if n_iter == self.max_iters and self.errors[-1] > self.epsilon:
        #     print(f'SGD did not converge after {self.max_iters} epochs. Increase max_iters for a better model.')
        #
        # return self

    def feed_forward(self, x):
        input_layer = x.insert(0, 1)
        for layer in self.layers:
            for node in layer.nodes:
                node.input_val = np.dot(input_layer, node.incoming_weights)
            input_layer = layer.get_outputs()

        return input_layer

    def predict(self, x):
        return self.feed_forward(x)

    def relu(self, x):
        return np.max(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.e ** -x)

    def softmax(self, x):
        return NotImplementedError

    def tanh(self, x):
        return 2 / (1 + np.e**(-2*x)) - 1

    def relu_dx(self, x):
        return 0 if x < 0 else 1

    def sigmoid_dx(self, x):
        return np.e**x / (np.e**x + 1)**2

    def softmax_dx(self, x):
        return NotImplementedError

    def tanh_dx(self, x):
        return 4 * np.e**(2 * x) / (np.e**-(2 * x) + 1) ** 2
