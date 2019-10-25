from linear_models.linear_regression import LinearRegressionModel
from linear_models.SGD import SGDRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor as sklSGD
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

linear_model = LinearRegressionModel()
sgd_model = SGDRegressor(epsilon=1e-3)
sklearn_model = sklSGD(
    max_iter=1000,
    learning_rate='constant',
    penalty=None)

# df = pd.read_csv('data/aggregated.csv').dropna().head(500)
# print(df.columns)
# x = df.drop(['County', 'Store.Location', 'Category.Name'], axis=1).apply(zscore)
# print(x)
# y = df[['Per.Capita']].apply(zscore)

# df = pd.read_csv('data/test_data.csv')
# print(df.columns)
# x = df[['X1', 'X2']]
# y = df[['Y']]

df = pd.read_csv('data/iris_setosa.csv')
x = df[['X']]
y = df[['Y']]
# x = pd.DataFrame([i for i in range(100)])
# y = pd.DataFrame([np.random.normal(5, 1) for i in range(100)])

iris, classes = load_iris(True)

# pca = PCA(n_components=2)
# transformed = pca.fit_transform(iris, classes)
# filter = zip(classes == 0, classes == 1)
# filter = [val[0] or val[1] for val in filter]
# x = pd.DataFrame(transformed[filter][:, 0])
# y = pd.DataFrame(transformed[filter][:, 1])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

sklearn_model.fit(x_train, np.ravel(y_train))
linear_model.fit(x_train, y_train)
sgd_model.fit(x_train, y_train)
print(sgd_model.coef)
plt.subplot(311)
sgd_model.generate_2d_plot(x, y)
plt.subplot(312)
sgd_model.plot_error()
plt.subplot(313)
predictions = sklearn_model.predict(x)
plt.plot(x, predictions)
plt.scatter(x, y)
plt.text(np.mean(x).iloc[0], np.max(predictions),
         f'y = {np.round(sklearn_model.intercept_, 3)} + {np.round(sklearn_model.coef_[0], 3)}x',
         horizontalalignment='center',
         verticalalignment='center',
         bbox=dict(facecolor='blue', alpha=0.25))
plt.title('Line of Best Fit (Sklearn)')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()


print(f'Adj. R^2: {linear_model.score(x_test, y_test)}')
print(f'Sklearn: \n'
      f'  Model: \n {sklearn_model}\n'
      f'  Intercept: {sklearn_model.intercept_}\n'
      f'  Slopes: {sklearn_model.coef_}\n'
      f'  Score: {sklearn_model.score(x_test, y_test)}')



