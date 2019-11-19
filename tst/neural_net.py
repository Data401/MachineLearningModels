from linear_models.neural_network import NeuralNetwork, ActivationFunction
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
df = pd.read_csv('~/Downloads/house_dataset.csv').drop(['timestamp'], axis=1)

print(df.head())
x = scaler.fit_transform(df.drop(['price_doc'], axis=1))
y = scaler.fit_transform(df[['price_doc']])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

nn = NeuralNetwork(
    nodes=[6, 3, 4],
    learning_rate=1e-8,
    batch_size=1,
    epsilon=1e-10,
    max_iters=100,
    activations=[
        ActivationFunction.RELU,
        ActivationFunction.SIGMOID,
        ActivationFunction.RELU, ])

nn.fit(x_train, y_train, verbose=1)
print(f'Score: {nn.score(x_test, y_test, metric="r2")}')
print(f'Prediction: {nn.predict(x_test)}')
print(f'Actual: {y_test}')
nn.pprint()

plt.subplot(211)
# nn.plot_loss()
plt.subplot(212)
nn.plot_error()
plt.show()

print(f'MSE: {np.mean((nn.predict(x_test) - y_test)**2)}')






