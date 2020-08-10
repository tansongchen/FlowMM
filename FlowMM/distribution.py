'''
Combined base distribution
'''

import torch
from numpy import pi, log, prod
from scipy.stats import truncnorm
from nflows.flows import Flow
from nflows.distributions import Distribution
from nflows.transforms import Transform
from nflows.utils.torchutils import sum_except_batch
from torch import Tensor

# class MMBaseDistribution(Distribution):
#     '''
#     Base distribution for bonds, angles and dihedrals corresponding to a naive additive force field
#     '''

#     def __init__(self, graph):
#         super().__init__()
#         n = graph.number_of_nodes()
#         self.n = n
#         self.bond = DiagonalNormal([n - 1])
#         self.angle = BoxUniform(low=torch.zeros([n - 2]), high=torch.full([n - 2], pi))
#         self.dihedral = BoxUniform(low=torch.full([n - 3], -pi), high=torch.full([n - 3], pi))

#     def _log_prob(self, inputs, context=None):
#         bond, angle, dihedral = inputs
#         return self.bond.log_prob(bond) + self.angle.log_prob(angle) + self.dihedral.log_prob(dihedral)

#     def _sample(self, num_samples, context=None):
#         n = self.n
#         internal = torch.Tensor(num_samples, 3 * n - 6)
#         internal[:, : n - 1] = self.bond.sample(num_samples, context)
#         internal[:, n - 1 : 2 * n - 3] = self.angle.sample(num_samples, context)
#         internal[:, 2 * n - 3 :] = self.dihedral.sample(num_samples, context)
#         return internal

class IndependentUniformDistribution(Distribution):

    def __init__(self, lefts: Tensor, rights: Tensor):
        super().__init__()
        self._lefts = lefts
        self._rights = rights
        self._shape = lefts.shape
        self._log_prob_value = (rights - lefts).prod().log() * -1

    def _log_prob(self, inputs: Tensor, context=None):
        return torch.full([inputs.shape[0]], self._log_prob_value)

    def _sample(self, num_samples, context=None):
        if context is None:
            noise = torch.rand(num_samples, *self._shape)
            return self._lefts + noise * (self._rights - self._lefts)
        else:
            context_size = context.shape[0]
            noise = torch.rand(context_size, num_samples, *self._shape)
            return self._lefts + noise * (self._rights - self._lefts)

class IndependentNormalDistribution(Distribution):

    def __init__(self, mean: Tensor, std: Tensor):
        super().__init__()
        self._shape = mean.shape
        self._mean = mean
        self._std = std
        self._log_z = 0.5 * prod(self._shape) * log(2 * pi) + sum_except_batch(self._std.log())

    def _log_prob(self, inputs: Tensor, context=None):
        normalized = (inputs - self._mean) / self._std
        log_prob = -0.5 * sum_except_batch(normalized ** 2)
        log_prob -= self._log_z
        return log_prob

    def _sample(self, num_samples, context):
        if context is None:
            noise = torch.randn(num_samples, *self._shape)
        else:
            context_size = context.shape[0]
            noise = torch.randn(context_size, num_samples, *self._shape)
        return self._mean + noise * self._std

class IndependentTruncatedNormalDistribution(Distribution):

    def __init__(self, mean: Tensor, std: Tensor, maxdev: Tensor):
        super().__init__()
        self._shape = mean.shape
        self._mean = mean
        self._std = std
        self._minval = (-maxdev**2/2).exp()
        self._log_z = 0.5 * prod(self._shape) * log(2 * pi) + sum_except_batch(std.log()) + sum_except_batch((maxdev / 2**0.5).erf().log())

    def _log_prob(self, inputs: Tensor, context=None):
        normalized = (inputs - self._mean) / self._std
        log_prob = -0.5 * sum_except_batch(normalized ** 2)
        log_prob -= self._log_z
        return log_prob

    def _randtn(self, *shape):
        u1 = torch.rand(shape) * (1 - self._minval) + self._minval
        u2 = torch.rand(shape)
        return torch.sqrt(-2 * u1.log()) * torch.cos(2 * pi * u2)

    def _sample(self, num_samples, context):
        if context is None:
            noise = self._randtn(num_samples, *self._shape)
        else:
            context_size = context.shape[0]
            noise = self._randtn(context_size, num_samples, *self._shape)
        return self._mean + noise * self._std

class MMAngularConditionalBaseDistribution(IndependentUniformDistribution):
    '''
    Angular conditional on bond
    '''
    def __init__(self, graph):
        features = 2 * graph.number_of_nodes() - 5
        low = torch.full([features], -pi)
        high = torch.full([features], pi)
        super().__init__(low, high)

class MMBondDistribution(IndependentNormalDistribution):
    '''
    Bond
    '''
    def __init__(self, bond: Tensor):
        mean = bond.mean(0)
        std = bond.std(0)
        super().__init__(mean, std)

class ContextedDistribution(Distribution):
    def __init__(self, contextDistribution: Distribution, conditionalFlow: Flow, n):
        super().__init__()
        self.contextDistribution = contextDistribution
        self.conditionalFlow = conditionalFlow
        self.n = n

    def _log_prob(self, inputs, context=None):
        # n = self.coordinate.graph.number_of_nodes()
        # internal, logabsdet = self.coordinate.forward(inputs)
        n = self.n
        context = inputs[:, 0 : n - 1]
        feature = inputs[:, n - 1 : 3 * n - 6]
        context_prob = self.contextDistribution.log_prob(context)
        feature_prob = self.conditionalFlow.log_prob(feature, context)
        return context_prob + feature_prob

    def _sample(self, num_samples, context=None):
        # n = self.coordinate.graph.number_of_nodes()
        n = self.n
        data = torch.zeros(num_samples, 3 * n - 6)
        data[:, 0 : n - 1] = self.contextDistribution.sample(num_samples)
        data[:, n - 1 : 3 * n - 6] = self.conditionalFlow.sample(1, data[:, 0 : n - 1].clone()).squeeze(1)
        # cartesian, _ = self.coordinate.inverse(data)
        return data

if __name__ == '__main__':
    u = IndependentUniformDistribution(torch.zeros(3), torch.ones(3))
    print(u.log_prob(u.sample(4)))
    print(u.sample(4, context=torch.zeros(2)))
    normal = IndependentTruncatedNormalDistribution(torch.zeros(3), torch.ones(3), torch.ones(3))
    s = normal.sample(4)
    print(s, normal.log_prob(s))