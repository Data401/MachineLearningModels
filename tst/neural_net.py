from linear_models.neural_network import NeuralNetwork, ActivationFunction
from sklearn.datasets import load_iris, load_boston, make_regression
from sklearn.model_selection import train_test_split
from pprint import PrettyPrinter
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore

pp = PrettyPrinter()

iris, classes = make_regression(n_samples=100, n_features=1)
iris = np.apply_along_axis(zscore, arr=iris, axis=0)

np.random.seed(42)
x_train, x_test, y_train, y_test = train_test_split(iris, classes, test_size=0.33)

x = np.asarray([[1, 2, 3], [2, 4, 6]])
y = np.asarray([10, -42])
nn = NeuralNetwork(
    nodes=[1],
    learning_rate=1e-3,
    batch_size=1,
    epsilon=1e-10,
    max_iters=1000,
    activations=[
        ActivationFunction.RELU])

nn.fit(x, y)
#pp.pprint(nn.score(x_test, y_test, metric='r2'))
print(f'Prediction: {nn.predict(x)}')
pp.pprint(nn)

print(nn.errors)

plt.subplot(311)
plt.scatter(iris, classes)
plt.subplot(312)
nn.plot_loss()
plt.subplot(313)
nn.plot_error()
plt.show()

print(nn.loss)
#
# from sklearn.linear_model import LinearRegression
#
# lr = LinearRegression()
#
# lr.fit(x_train, y_train)
# print(lr.score(x_test, y_test))




