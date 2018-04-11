from abc import ABCMeta
from abc import abstractmethod


class CPD:
    __metaclass__ = ABCMeta

    def __init__(self, var_idx, parents):
        super(CPD, self).__init__()

        self.var_idx = var_idx
        self.parents = parents

    @abstractmethod
    def fit(self, observations, targets):
        pass

    @abstractmethod
    def log_likelihood(self, observations, targets):
        pass

    @abstractmethod
    def n_params(self):
        pass


    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'var_idx={}, '.format(self.var_idx)
        s += 'parents={}'.format(self.parents)
        s += ')'

        return s
