from src.linear_models import SGDClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import pandas as pd

iris, classes = load_iris(True)

pca = PCA(n_components=2)

transformed = pca.fit_transform(iris, classes)


plt.scatter(transformed[classes == 0][:, 0], transformed[classes == 0][:, 1], c='r')
plt.scatter(transformed[classes == 1][:, 0], transformed[classes == 1][:, 1], c='k')
plt.show()
log_classifier = SGDClassifier(max_iters=5000)

filter = zip(classes == 0, classes == 1)
filter = [val[0] or val[1] for val in filter]
x = pd.DataFrame(transformed[filter][:, 0])
y = pd.DataFrame(transformed[filter][:, 1])

x = pd.DataFrame(iris[filter])
y = pd.DataFrame(classes[filter])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

log_classifier.fit(x_train, y_train)
predictions = log_classifier.predict(x_test)

