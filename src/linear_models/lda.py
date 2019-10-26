from linear_models.base_model import BaseModel
import numpy as np


class LDAClassifier(BaseModel):
    def __init__(self):
        super(LDAClassifier, self).__init__()
        self.method = 'MLE'
        self.weights = None
        self.mean_0 = None
        self.mean_1 = None
        self.classes = None

    def __str__(self):
        return f'Linear Regression Model\n' \
               f'Coefficients: \n{self.coef}\n' \
               f'Intercept: {self.intercept}\n' \


    def _fit(self, data, y, **kwargs):
        assert len(data) == len(y)
        assert len(np.unique(y) == 2)

        self.classes = np.unique(y)
        # separate the data
        X_0 = data[y == self.classes[0]]
        X_1 = data[y == self.classes[1]]

        mu_0 = X_0.mean(axis=0)
        mu_1 = X_1.mean(axis=0)

        # find B
        diff = mu_0 - mu_1
        B = np.outer(diff, diff)

        # find S
        S_0 = np.apply_along_axis(lambda x: np.outer(x, x), 1, X_0 - mu_0).sum(axis=0)
        S_1 = np.apply_along_axis(lambda x: np.outer(x, x), 1, X_1 - mu_1).sum(axis=0)
        S = S_0 + S_1

        # solve
        values, vectors = np.linalg.eig(np.matmul(np.linalg.inv(S), B))
        w = vectors[:, np.argmax(values)]

        self.weights = w
        self.mean_0 = w.dot(mu_0)
        self.mean_1 = w.dot(mu_1)
        return self

    def predict(self, x):
        scores = x.dot(self.weights)

        distance_to_0 = np.abs(scores - self.mean_0)
        distance_to_1 = np.abs(scores - self.mean_1)

        return self.classes[(distance_to_1 < distance_to_0).astype(int)]






