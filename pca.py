# Method signatures from CS 4641 HW3


import numpy as np


class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) ->None:
        X = X - np.average(X, axis=0)	# Subtract the averages
        self.U, self.S, self.V = np.linalg.svd(X, full_matrices=False)

    def transform(self, data: np.ndarray, K: int=2) ->np.ndarray:
        X_new = self.U[:, :K] * self.S[:K]
        
        return X_new

    def transform_rv(self, data: np.ndarray, retained_variance: float=0.99
        ) ->np.ndarray:
        cumsum = np.cumsum(self.S**2)
        cumsum = cumsum / cumsum[-1]
        K = np.argmax(cumsum >= retained_variance) + 1
        
        return self.transform(data, K)