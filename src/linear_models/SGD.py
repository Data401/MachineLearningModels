import numpy as np

from linear_models.base_model import BaseModel
from abc import abstractmethod
from timeit import default_timer as timer
from errors.convergence import ConvergenceError
from linear_models.base_model import _convert_dataframe
import matplotlib.pyplot as plt

from plotting_utils import plot_2d_decision_boundary, plot_contours, make_meshgrid


class _BaseSGD(BaseModel):

    @abstractmethod
    def _change_in_loss(self, x, y, b):
        pass

    def _fit(self, x, y, **kwargs):
        return self._fit_by_sgd(x, y, **kwargs)

    def _fit_by_sgd(self, x, y, verbose=0, plot_iterations=False):
        """
        Calculates the gradient of the loss function for linear regression.

        :param x: Column vector of explanatory variables
        :param y: Column vector of dependent variables
        :return: Vector of parameters for Linear Regression
        """
        assert len(x) == len(y)

        # Reset model error calculations
        self.errors = []
        self.iterations = []

        # Setup Debugging/ Graphing
        start_time = timer()
        train_time = 0

        # Copy input DataFrame so we don't modify original (may need to change if copy is too expensive)
        intercept_terms = np.ones((x.shape[0], 1))
        x0 = np.hstack((intercept_terms, x.copy()))
        y0 = y.reshape(len(y), 1)  # Convert y to a column vector

        betas = np.zeros((len(x0[0]), 1))  # Makes a column vector of zeros

        n_iter = 0
        n_iters_no_change = 0

        while n_iters_no_change < self.max_iters_no_change and n_iter < self.max_iters:
            permutation = np.random.permutation(x0.shape[0])
            x0 = x0[permutation]
            y0 = y0[permutation]

            pre_epoch_betas = betas
            # Iterate through all (X, Y) pairs where X is a vector of predictor variables [x1, x2, x3, ...]
            # and Y is a vector containing the response variable
            with np.errstate(invalid='raise'):
                for v, w in zip(x0, y0):
                    v = v.reshape(1, len(v))
                    w = w.reshape(1, len(w))
                    prior_betas = betas
                    try:
                        loss_change = self.learning_rate * self._change_in_loss(v, w, prior_betas)
                        betas = np.subtract(prior_betas, loss_change)
                    except FloatingPointError:
                        raise ConvergenceError()

            total_error = np.sqrt(np.sum(np.subtract(betas, pre_epoch_betas) ** 2))
            n_iters_no_change = n_iters_no_change + 1 if total_error < self.epsilon else 0
            n_iter += 1
            train_time = timer() - start_time
            if verbose > 0:
                print(
                    f'-- Epoch {n_iter}\n'
                    f'Total training time: {round(train_time, 3)}')
                if verbose > 1:
                    print(f'Equation:\n'
                          f'y = {np.round(betas[1:][0][0], 3)}(x1) + {np.round(betas[1:][1][0], 3)}(x2) + {np.round(betas[0][0], 3)}')
                if verbose > 2:
                    print(
                        f'Pre Epoch Betas:\n{pre_epoch_betas}\n'
                        f'Post Epoch Betas:\n{betas}\n')
            self.iterations.append(n_iter)
            self.errors.append(total_error)

        self.coef_ = betas[1:]
        self.intercept_ = betas[0][0] if self.fit_intercept else 0  # betas[0] gives a series with a single value
        if verbose > 0:
            print(f'SGD converged after {n_iter} epochs.\n'
                  f'Total Training Time: {round(train_time, 3)} sec.')

        if n_iter == self.max_iters and self.errors[-1] > self.epsilon:
            print(f'SGD did not converge after {self.max_iters} epochs. Increase max_iters for a better model.')

        return self


class SGDRegressor(_BaseSGD):
    def __init__(
            self,
            learning_rate=1e-4,
            epsilon=1e-3,
            penalty='l2',
            max_iters=1000,
            max_iters_no_change=5,
            fit_intercept=True,
            alpha=1e-4):
        super(SGDRegressor, self).__init__()
        self.method = 'SGD'
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.penalty = penalty
        self.max_iters = max_iters
        self.max_iters_no_change = max_iters_no_change
        self.fit_intercept = fit_intercept
        self.alpha = alpha

    def _change_in_loss(self, x, y, b):
        """
        Returns the value of the gradient of the loss function with a regularization term
        dL/dβ = -2η(X.T(y - Xb)
        Where X is a column vector of [x_1, x_2, ... , x_i] and
        y is a column vector of [y_1, y_2, ... , y_i] and
        b is the column vector of [β_1, β_2, ... , β_n]

        However, since this is stochastic, we randomly sample a single x_i, y_i
        :param x: Vector of predictor variables for a single observation
        :param y: Vector containing the response variable for a single observation
        :param b: vector of prior betas
        :return: vector of changes to apply to vector of betas
        """
        if self.penalty == 'l2':
            xb = x.dot(b)
            change_in_loss = np.multiply(-2 * self.learning_rate, x.T.dot(y - xb))
            p = np.multiply(2 * self.alpha, b)
            p[0] = 0  # Penalizing the intercept is no bueno and b = [b_0, b_1, ...]
            return change_in_loss + p
        if self.penalty is None:
            xb = x.dot(b)
            change_in_loss = np.multiply(-2 * self.learning_rate, x.T.dot(y - xb))
            return change_in_loss
        else:
            print(f'Other penalty types are not supported yet.')
            return NotImplementedError


class SGDClassifier(_BaseSGD):
    def __init__(
            self,
            learning_rate=1e-4,
            epsilon=1e-3,
            penalty='l2',
            max_iters=1000,
            max_iters_no_change=5,
            fit_intercept=False,
            alpha=1e-4,
            loss='hinge',
            C=1):
        super(SGDClassifier, self).__init__()
        self.method = 'SGD'
        self.learning_rate = learning_rate
        self.loss = loss
        self.epsilon = epsilon
        self.penalty = penalty
        self.max_iters = max_iters
        self.max_iters_no_change = max_iters_no_change
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.C = C

    def _change_in_loss(self, x, y, b):
        """
        Returns the value of the gradient of the loss function with a L2 regularization term
        The gradient that is calculated by this method is determined by self.loss
        Possible values:
        'log' - Compute the gradient for logistic regression
        'hinge' - Compute the gradient for SVM

        :param x0: Vector of predictor variables for a single observation
        :param y0: Vector containing the response variable for a single observation
        :param b: vector of prior betas
        :return: COLUMN vector of changes to apply to COLUMN vector of betas
        """
        if self.loss == 'log':
            xb = -(x.dot(b))
            exp = np.exp(xb)
            p = 1 / (1 + exp)
            reg_term = np.multiply(self.alpha / len(b), b)
            reg_term[0] = 0

            change_in_loss = -np.dot(np.subtract(y, p), x).T
            return change_in_loss + reg_term

        elif self.loss == 'hinge':
            xb = x.dot(b)[0][0]
            return b - self.C * y.dot(x).T if y.dot(xb) < 1 else b

    def predict(self, x):
        x0 = _convert_dataframe(x)
        return self.classes[(self.decision_function(x0) > 0).astype(int)].T.flatten()

    def score(self, x, y, metric=None):
        x0 = _convert_dataframe(x)
        y0 = _convert_dataframe(y)
        return np.mean(self.predict(x0) == y0)

    def decision_function(self, x):
        x = _convert_dataframe(x)
        return self.intercept_ + np.dot(x, self.coef_)

    def decision_boundary(self, x, c=0):
        b1 = self.coef_[0]
        b2 = self.coef_[1]
        return (b1*x + self.intercept_ - c)/-b2

    def _fit(self, x, y, **kwargs):
        assert len(np.unique(y) == 2)
        y0 = np.asarray(y).flatten()
        self.classes = np.unique(y0)
        if self.loss == 'hinge':
            y0 = np.asarray([-1 if val == self.classes[0] else 1 for val in y0])
        elif self.loss == 'log':
            y0 = np.asarray([0 if val == self.classes[0] else 1 for val in y0])
        else:
            return NotImplementedError

        return super()._fit(x, y0, **kwargs)

    def generate_2d_plot(self, x, y):
        ax = plt.gca()
        xx, yy = make_meshgrid(x, y)
        plot_contours(self, ax, xx, yy)
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap='winter', edgecolors='k')





