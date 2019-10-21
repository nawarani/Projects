from sklearn.base import TransformerMixin
import numpy as np
class psi_func(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y= None):
        return self

    def transform(self, X, y= None):
        return np.array([int(round(float(x.strip('%')),0)) for x in X])

    def fit_transform(self, X, y= None):
        self.fit(X)
        return self.transform(X)

class company_func(TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y= None):
        return self

    def transform(self, X, y= None):
        return np.where(X == 'U.S.A.', 1, 0)

    def fit_transform(self, X, y= None):
        self.fit(X)
        return self.transform(X)
