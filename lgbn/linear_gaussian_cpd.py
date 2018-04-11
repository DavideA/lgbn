import numpy as np
from cpd import CPD
from sklearn.linear_model import LinearRegression

epsilon = np.finfo(float).eps


class LinearGaussian(CPD):

    def __init__(self, var_idx, parents):
        super(LinearGaussian, self).__init__(var_idx, parents)

        self.linear_model = LinearRegression()
        self.std = None

    def fit(self, observations, targets):
        n_samples, n_parents = observations.shape
        assert n_parents == self.parents.size

        # Linear Gaussian
        self.linear_model.fit(X=observations, y=targets)
        self.std = np.sqrt(self.linear_model._residues / n_samples)

    def log_likelihood(self, observations, targets):
        assert self.is_fit, 'Model is not fit. Did you call LinearGaussian.fit()?'

        n_samples, n_parents = observations.shape

        llk = - n_samples * np.log(self.std * np.sqrt(2 * np.pi) + epsilon)

        mus = self.linear_model.predict(observations)
        llk -= 0.5 * np.sum(np.square((targets - mus) / (self.std + epsilon)))

        return llk

    @property
    def n_params(self):
        return self.parents.size + 1 + 1  # bias and std

    @property
    def is_fit(self):
        return hasattr(self.linear_model, 'coef_') and \
               hasattr(self.linear_model, 'intercept_') \
               and self.std is not None
