import numpy as np
from src.linear_models import BaseModel


class LinearRegressionModel(BaseModel):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.method = 'MLE'

    def __str__(self):
        return f'Linear Regression Model\n' \
               f'Coefficients: \n{self.coef}\n' \
               f'Intercept: {self.intercept}\n' \


    def fit(self, x, y, **kwargs):
        """
           This method actually builds the model.

           :parameter x - Pandas DataFrame containing explanatory variables
           :parameter y - Pandas DataFrame containing response variables
        """
        return self._fit_by_mle(x, y)

    def _fit_by_mle(self, x, y):
        # Using qr factorization
        # qr = x and z = Q^T y yields r betas = z
        q, r = np.linalg.qr(x)
        z = np.dot(q.T, y)
        
        # use back subsitution for to solve for betas
        betas = np.linalg.solve(r, z)

        self.intercept = 0
        self.coef = betas

        return self
