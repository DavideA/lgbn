import numpy as np
from cpd import CPD

epsilon = np.finfo(float).eps


class Gaussian(CPD):
    def __init__(self, var_idx, parents):
        super(Gaussian, self).__init__(var_idx, parents)

        self.mu = None
        self.std = None

    def fit(self, observations, targets):
        n_samples, n_parents = observations.shape
        assert n_parents == self.parents.size == 0

        # Fit Gaussian
        self.mu = np.mean(targets)
        self.std = np.std(targets)

    def log_likelihood(self, observations, targets):
        assert self.is_fit, 'Model is not fit. Did you call Gaussian.fit()?'

        n_samples, n_parents = observations.shape

        llk = - n_samples * np.log(self.std * np.sqrt(2 * np.pi) + epsilon)
        llk -= 0.5 * np.sum(np.square((targets - self.mu) / (self.std + epsilon)))

        return llk

    @property
    def n_params(self):
        return 1 + 1  # mean and std

    @property
    def is_fit(self):
        return self.mu is not None and self.std is not None
