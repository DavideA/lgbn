"""
This class models a Bayesian Net.
Each CPD is a Linear Gaussian model.
"""
import numpy as np
from linear_gaussian_cpd import LinearGaussian
from gaussian_cpd import Gaussian


class BayesianNetwork:

    def __init__(self, graph):
        assert graph.ndim == 2, \
            'Graph should be a 2-dimensional matrix'
        n, m = graph.shape
        assert n == m, \
            'Graph should be a square matrix.'
        assert np.array_equal(graph, graph.astype(bool)), \
            'Graph matrix should be binary.'
        assert np.count_nonzero(np.diag(graph)) == 0, \
            'Graph matrix is non zero on diagonals, no recurrent edges allowed.'
        assert np.count_nonzero(np.diag(np.linalg.matrix_power(graph, n))) == 0, \
            'Graph matrix is not acyclic!'
        assert np.linalg.det(graph) == 0, \
            'Graph matrix is not rooted!'

        self.graph = graph
        self.n_variables = n
        self.variables = np.arange(0, n)

        self.cpds = []
        for var_idx in range(0, self.n_variables):
            parents = self.variables[self.graph[var_idx].astype(np.bool)]
            if parents.size > 0:
                self.cpds.append(LinearGaussian(var_idx=var_idx, parents=parents))
            else:  # == 0
                self.cpds.append(Gaussian(var_idx=var_idx, parents=parents))

    def fit(self, data):
        for cpd, parents_idx in zip(self.cpds, self.graph):
            cpd_parents = data[:, self.graph[cpd.var_idx].astype(np.bool)]
            cpd_targets = data[:, cpd.var_idx]
            cpd.fit(observations=cpd_parents, targets=cpd_targets)

    def log_likelyhood(self, data):
        pass

    @property
    def is_fit(self):
        return np.all([cpd.is_fit for cpd in self.cpds])

    @property
    def independent_params(self):
        return np.sum([cpd.n_params for cpd in self.cpds])

    def __repr__(self):
        s = self.__class__.__name__
        s += '(\nCPDs:(\n'
        s += '\n'.join([str(cpd) for cpd in self.cpds])
        s += '\n),\nis_fit:{},'.format(self.is_fit)
        s += '\nindependent_params:{},'.format(self.independent_params)
        return s


n_variables = 64
n_samples = 10000
graph = np.tril(np.ones(shape=(n_variables, n_variables)), k=-1)
data = np.random.rand(n_samples, n_variables)
net = BayesianNetwork(graph)
net.fit(data)
print net
